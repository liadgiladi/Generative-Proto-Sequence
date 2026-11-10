"""Minimal causal Transformer decoder (minGPT-style) for conditional sequence generation.

Adapted from Andrej Karpathy's minGPT (https://github.com/karpathy/minGPT) and trimmed
down to the bare essentials needed to replace the MLP decoder of the VAE in
``dense_var_auto_encoder_vin_1.py``.

The key addition over a vanilla GPT is *latent conditioning*: the decoder is driven by a
continuous latent vector ``z`` (size ``decoder_input_size``). ``z`` is linearly projected
into the Transformer's embedding space and inserted as the **first token** of the
sequence -- a prefix conditioning token. Because attention is causal, every real token
can attend back to this prefix and thus to ``z``, which makes the whole sequence a
function of the latent code, exactly as a VAE decoder requires.

Architecture defaults are deliberately tiny (``n_layer=2``, ``n_head=4``, ``n_embd=32``)
because our action sequences are short (length 10), the vocabulary is small (5), and the
decoder is trained from scratch through RL -- a smaller model is both faster and easier to
optimise from sparse reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Hyper-parameters for the conditional GPT decoder.

    Attributes:
        vocab_size: Number of distinct tokens (4 actions + 1 EOS = 5).
        block_size: Maximum length of the *real* token sequence (10). The model
            internally works on ``block_size + 1`` positions to make room for the
            prefix conditioning token derived from ``z``.
        decoder_input_size: Dimensionality of the continuous latent vector ``z``.
        n_layer: Number of Transformer blocks.
        n_head: Number of attention heads (must divide ``n_embd``).
        n_embd: Embedding / hidden width of the Transformer.
        embd_pdrop: Dropout applied to the summed token + positional embeddings.
        resid_pdrop: Dropout applied to the residual paths inside each block.
        attn_pdrop: Dropout applied to the attention probabilities.
    """

    vocab_size: int = 5
    block_size: int = 10
    decoder_input_size: int = 16
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 32
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1


class CausalSelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer with a projection at the end.

    Each position may only attend to itself and earlier positions (causal masking),
    which is what makes the decoder autoregressive.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Combined linear projection producing queries, keys and values in one matmul.
        self.c_attn: nn.Linear = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection applied after attention.
        self.c_proj: nn.Linear = nn.Linear(config.n_embd, config.n_embd)

        # Regularisation. Attention dropout is applied inside the fused SDPA kernel (see
        # ``forward``) rather than by a separate nn.Dropout module.
        self.attn_pdrop: float = config.attn_pdrop
        self.resid_dropout: nn.Dropout = nn.Dropout(config.resid_pdrop)

        self.n_head: int = config.n_head
        self.n_embd: int = config.n_embd

        # Causal mask of shape (1, 1, T, T) where T = block_size + 1 (prefix token).
        # Registered as a buffer so it moves with ``.to(device)`` but is not a parameter.
        # Stored as bool, which is what scaled_dot_product_attention wants, so no dtype
        # conversion is needed on the hot path.
        max_len = config.block_size + 1
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)).view(1, 1, max_len, max_len)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Apply masked self-attention, optionally with a key/value cache.

        Args:
            x: Input of shape ``(B, Tq, n_embd)`` where ``Tq`` is the number of *new*
                query positions -- the whole sequence in the non-cached case, or a single
                token during incremental (cached) decoding.
            layer_past: Optional ``(k, v)`` from earlier decoding steps, each
                ``(B, n_head, T_past, head_dim)``; prepended to the current keys/values.
            use_cache: If ``True`` also return the updated ``(k, v)`` for reuse.

        Returns:
            ``(y, present)`` where ``y`` is ``(B, Tq, n_embd)`` and ``present`` is the
            extended ``(k, v)`` cache (or ``None`` when ``use_cache`` is ``False``).
        """
        B, Tq, C = x.size()  # batch, number of new query positions, embedding dim

        # Project and split into query, key, value for the new positions: each (B, Tq, C).
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape into heads: (B, n_head, Tq, head_dim).
        head_dim = C // self.n_head
        q = q.view(B, Tq, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, Tq, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, Tq, self.n_head, head_dim).transpose(1, 2)

        # Prepend cached keys/values from earlier decoding steps (KV cache).
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        # Scaled dot-product attention with causal masking, as a single fused kernel
        # instead of the matmul / scale / masked_fill / softmax / dropout / matmul chain.
        # At our sizes (n_embd=32, seq<=11) every one of those was launch-latency bound,
        # so collapsing them matters far more than the arithmetic does.
        #
        # When the keys include cached positions (Tk > Tq), the Tq new queries occupy
        # absolute rows [Tk - Tq, Tk); slicing the lower-triangular buffer this way lets
        # each new query attend to every key up to and including its own position. For the
        # full-sequence case Tk == Tq this reduces to the usual [:T, :T] mask. In the
        # single-token decode step (Tq == 1) the one query sits at the last position, so
        # every key is already in its past and no mask is needed at all.
        Tk = k.size(2)
        attn_mask = None if Tq == 1 else self.causal_mask[:, :, Tk - Tq:Tk, :Tk]

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_pdrop if self.training else 0.0,
        )  # (B, nh, Tq, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)  # re-assemble heads

        # Output projection + residual dropout.
        y = self.resid_dropout(self.c_proj(y))
        return y, present


class Block(nn.Module):
    """A single Transformer block: causal self-attention followed by an MLP.

    Uses the pre-LayerNorm formulation (LayerNorm before each sub-layer), which trains
    more stably than the original post-LayerNorm GPT.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1: nn.LayerNorm = nn.LayerNorm(config.n_embd)
        self.attn: CausalSelfAttention = CausalSelfAttention(config)
        self.ln_2: nn.LayerNorm = nn.LayerNorm(config.n_embd)

        # Position-wise feed-forward network (4x expansion is the GPT convention).
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPT(nn.Module):
    """A minimal GPT decoder conditioned on a continuous latent vector ``z``.

    The forward pass prepends a single conditioning token -- a linear projection of
    ``z`` -- to the (optional) token embeddings, then runs a causal Transformer. The
    logits aligned with the prefix predict the first action, and so on, giving an
    autoregressive ``p(sequence | z)`` factorisation.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config: GPTConfig = config
        self.block_size: int = config.block_size

        # --- Latent conditioning: project z into the Transformer's embedding space. ---
        # The result is treated as the first ("prefix") token of the sequence.
        self.z_proj: nn.Linear = nn.Linear(config.decoder_input_size, config.n_embd)

        # Token and (learned) positional embeddings. ``+ 1`` position accounts for the
        # prefix conditioning token sitting in front of the real sequence.
        self.tok_emb: nn.Embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb: nn.Parameter = nn.Parameter(
            torch.zeros(1, config.block_size + 1, config.n_embd)
        )
        self.drop: nn.Dropout = nn.Dropout(config.embd_pdrop)

        # Transformer trunk.
        self.blocks: nn.ModuleList = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f: nn.LayerNorm = nn.LayerNorm(config.n_embd)

        # Language-model head: project hidden states back to vocabulary logits.
        self.head: nn.Linear = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # --- Conditioning-aware init overrides (applied after the generic init) ---
        # 1. Position embeddings: the generic _init_weights does not touch nn.Parameter,
        #    so pos_emb would otherwise stay all-zeros. Give it the standard small-random
        #    init (as GPT-2 does) so positions are distinguishable from step 0.
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        # 2. Latent projection: initialise z_proj much larger than the 0.02-scale trunk so
        #    the z-prefix *dominates the (un-normalised) residual stream* at init -- i.e. z
        #    actually drives the decoder's outputs from the first step instead of being
        #    swamped by the near-random trunk. std = 1/sqrt(decoder_input_size) makes the
        #    prefix roughly unit-variance (~10x the rest). This scale is a tunable knob.
        nn.init.normal_(self.z_proj.weight, mean=0.0, std=1.0 / math.sqrt(config.decoder_input_size))
        nn.init.zeros_(self.z_proj.bias)

        # Loudly announce (once, at construction) that the minGPT Transformer decoder is
        # in use -- so a run can be confirmed to be using this file and not the old MLP.
        self._announce()

    def _announce(self) -> None:
        """Print a one-time banner and, if a wandb run is active, tag its config.

        This is purely a diagnostic so you can verify from the logs / wandb that the
        Transformer decoder (this file) is the one being instantiated.
        """
        cfg = self.config
        n_params = sum(p.numel() for p in self.parameters())
        print(
            "\n==================== [GPS] DECODER = minGPT GPT ====================\n"
            f"  mingpt_decoder.GPT active  |  trainable params = {n_params:,}\n"
            f"  n_layer={cfg.n_layer}  n_head={cfg.n_head}  n_embd={cfg.n_embd}\n"
            f"  vocab_size={cfg.vocab_size}  block_size={cfg.block_size}  "
            f"decoder_input_size={cfg.decoder_input_size}\n"
            "====================================================================\n",
            flush=True,
        )

        # Best-effort: record in wandb if (and only if) a run is already active.
        # Never let logging issues break model construction.
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update(
                    {
                        "decoder_type": "mingpt_gpt",
                        "decoder_n_params": n_params,
                        "decoder_n_layer": cfg.n_layer,
                        "decoder_n_head": cfg.n_head,
                        "decoder_n_embd": cfg.n_embd,
                        "decoder_vocab_size": cfg.vocab_size,
                        "decoder_block_size": cfg.block_size,
                        "decoder_input_size": cfg.decoder_input_size,
                    },
                    allow_val_change=True,
                )
                print("[GPS] tagged wandb run config with decoder_type=mingpt_gpt", flush=True)
        except Exception as exc:  # pragma: no cover - diagnostics must never crash training
            print(f"[GPS] (wandb tagging skipped: {exc})", flush=True)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-style weight initialisation."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        z: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the conditional decoder.

        Args:
            z: Latent conditioning vector of shape ``(B, decoder_input_size)``.
            idx: Token indices of shape ``(B, T)`` with ``T <= block_size``. These are
                the (teacher-forced) input tokens. If ``None``, the model runs on the
                prefix token alone -- useful as the first step of free-running
                generation.
            targets: Optional ground-truth token indices of shape ``(B, T)`` for which a
                cross-entropy loss is computed and returned.

        Returns:
            A tuple ``(logits, loss)`` where ``logits`` has shape
            ``(B, T_total, vocab_size)`` with ``T_total = (T or 0) + 1`` (the leading
            position corresponds to the prefix conditioning token), and ``loss`` is the
            cross-entropy if ``targets`` was provided, otherwise ``None``.
        """
        B = z.size(0)

        # Project the latent into a single prefix token: (B, 1, n_embd).
        prefix = self.z_proj(z).unsqueeze(1)

        if idx is not None:
            T = idx.size(1)
            assert T <= self.block_size, (
                f"Sequence length {T} exceeds block size {self.block_size}"
            )
            token_embeddings = self.tok_emb(idx)  # (B, T, n_embd)
            # Concatenate prefix in front of the real tokens.
            x = torch.cat([prefix, token_embeddings], dim=1)  # (B, T + 1, n_embd)
        else:
            x = prefix  # (B, 1, n_embd)

        t_total = x.size(1)
        position_embeddings = self.pos_emb[:, :t_total, :]  # (1, T + 1, n_embd)
        x = self.drop(x + position_embeddings)

        for block in self.blocks:
            x, _ = block(x)  # no cache on the (parallel) teacher-forced path
        x = self.ln_f(x)

        logits = self.head(x)  # (B, T + 1, vocab_size)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            # The logit at position i predicts token i. Position 0 is fed by the prefix
            # (predicts the first action); positions 1..T are fed by tokens 0..T-1. We
            # therefore align the first T logits with the T targets.
            logits_for_loss = logits[:, : targets.size(1), :]
            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    # ------------------------------------------------------------------ #
    # Cached incremental decoding (KV cache)
    #
    # The teacher-forced ``forward`` above processes the whole sequence in one parallel
    # pass. Free-running generation instead needs one step per token; the helpers below
    # keep a per-layer key/value cache so each step only computes the new token's
    # attention instead of re-encoding the entire growing prefix (O(T) instead of O(T^2)).
    # ------------------------------------------------------------------ #

    def _forward_trunk(
        self,
        embeddings: torch.Tensor,
        position_offset: int,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Run the Transformer trunk on a chunk of already-computed embeddings.

        Args:
            embeddings: ``(B, chunk_len, n_embd)`` prefix and/or token embeddings.
            position_offset: absolute index of the first embedding (to slice ``pos_emb``).
            past_key_values: optional list of per-layer ``(k, v)`` caches.
            use_cache: whether to return the extended per-layer caches.

        Returns:
            ``(logits, presents)`` with logits ``(B, chunk_len, vocab_size)`` and
            ``presents`` the updated cache list (or ``None``).
        """
        chunk_len = embeddings.size(1)
        position_embeddings = self.pos_emb[:, position_offset:position_offset + chunk_len, :]
        x = self.drop(embeddings + position_embeddings)

        presents: Optional[list] = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, layer_past=layer_past, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, presents

    def init_decode(
        self, z: torch.Tensor, use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Begin cached decoding: process the z-prefix and return the first-token logits.

        Args:
            z: Latent conditioning vector of shape ``(B, decoder_input_size)``.
            use_cache: whether to build the key/value cache (set ``False`` to disable).

        Returns:
            ``(logits, past)`` with logits ``(B, vocab_size)`` predicting the first action
            and ``past`` the initial per-layer cache.
        """
        prefix = self.z_proj(z).unsqueeze(1)  # (B, 1, n_embd) -- prefix at position 0
        logits, past = self._forward_trunk(prefix, position_offset=0, use_cache=use_cache)
        return logits[:, -1, :], past

    def decode_step(
        self,
        tokens: torch.Tensor,
        past_key_values: Optional[list],
        position: int,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Advance cached decoding by one token.

        Args:
            tokens: either integer token ids ``(B,)`` / ``(B, 1)`` chosen at the previous
                step, or a float distribution over the vocabulary ``(B, vocab_size)``.
                The float form multiplies the embedding table rather than indexing it,
                which keeps this step differentiable with respect to whatever produced the
                distribution -- see ``ActionGen``'s soft-token-feedback path. For a
                one-hot row the two are mathematically identical; only the backward pass
                differs.
            past_key_values: cache from the previous ``init_decode`` / ``decode_step``.
            position: absolute position of ``tokens`` (prefix is at 0, so the first real
                token is at position 1).
            use_cache: whether to keep extending the cache.

        Returns:
            ``(logits, past)`` with logits ``(B, vocab_size)`` predicting the next token.
        """
        if tokens.is_floating_point():
            # (B, vocab) @ (vocab, n_embd) -> (B, 1, n_embd)
            token_embeddings = (tokens @ self.tok_emb.weight).unsqueeze(1)
        else:
            token_embeddings = self.tok_emb(tokens.view(-1, 1))  # (B, 1, n_embd)
        logits, past = self._forward_trunk(
            token_embeddings,
            position_offset=position,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return logits[:, -1, :], past

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively sample a token sequence conditioned on ``z`` (KV-cached).

        Args:
            z: Latent conditioning vector of shape ``(B, decoder_input_size)``.
            max_new_tokens: Number of tokens to generate. Defaults to ``block_size``.
            temperature: Softmax temperature; lower is more deterministic.
            do_sample: If ``True`` sample from the distribution, else take the argmax.
            top_k: If set, restrict sampling to the ``top_k`` most likely tokens.

        Returns:
            Generated token indices of shape ``(B, max_new_tokens)``.
        """
        if max_new_tokens is None:
            max_new_tokens = self.block_size

        # Process the z-prefix once; logits predict the first token.
        logits, past = self.init_decode(z, use_cache=True)

        generated = []
        for step in range(max_new_tokens):
            step_logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(step_logits, min(top_k, step_logits.size(-1)))
                step_logits = step_logits.masked_fill(step_logits < v[:, [-1]], float("-inf"))

            probs = F.softmax(step_logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            generated.append(next_token)
            if step < max_new_tokens - 1:
                # The token just produced sits at absolute position ``step + 1``.
                logits, past = self.decode_step(next_token, past, position=step + 1, use_cache=True)

        return torch.cat(generated, dim=1)
