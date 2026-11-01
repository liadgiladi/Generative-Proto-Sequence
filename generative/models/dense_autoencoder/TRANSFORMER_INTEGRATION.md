# Transformer Decoder Integration

> **Picking up this work?** Start with [`EVIDENCE.md`](../../../EVIDENCE.md) at the
> repository root — current results, methodology, and the metrics and flags that silently
> lie or do nothing. This file is the full chronological record; `EVIDENCE.md` is the
> orientation layer.

Replacing the VAE's single-shot MLP decoder with a minimal, latent-conditioned causal
Transformer (minGPT-style) and migrating the surrounding mechanics from a flattened
one-hot representation to integer-token autoregressive modeling.

Files touched:
- **New:** [mingpt_decoder.py](mingpt_decoder.py)
- **Changed:** [dense_var_auto_encoder_vin_1.py](dense_var_auto_encoder_vin_1.py)
- **Changed:** [decoder_utils.py](decoder_utils.py)
- **Changed:** [gps_simplegrid_levels.py](../../../gps_simplegrid_levels.py) (§7, §8)

Tooling added for the §7–§8 investigations, all at the repository root:
- `bench_decoder.py` — per-call decoder microbenchmark (§7.2)
- `eval_checkpoint.py` — run the test evaluation against saved checkpoints, for runs that
  died before reaching their own (§8.1)
- `check_z_influence.py` — separate actor collapse from decoder collapse (§8.3)
- `sbatch/` — the experiment matrix (`e0`–`e8` runtime, `a1`–`a3`/`b2` accuracy), plus
  `sbatch/_common.sh` and `sbatch/eval_checkpoint.sbatch`

---

## 1. The new Transformer — `mingpt_decoder.py`

A small, self-contained causal decoder adapted from Andrej Karpathy's minGPT, trimmed to
the essentials and extended with **latent conditioning**.

### Classes

| Class | Role |
|---|---|
| `GPTConfig` | Dataclass of hyper-parameters. Defaults are deliberately tiny: `n_layer=2`, `n_head=4`, `n_embd=32`, `vocab_size=5`, `block_size=10`, `decoder_input_size=16` (shrunk from `n_layer=3`/`n_embd=64` — see §6.3). |
| `CausalSelfAttention` | Multi-head masked self-attention. Fused QKV projection, output projection, and a registered lower-triangular causal mask buffer sized `block_size + 1`. |
| `Block` | Pre-LayerNorm Transformer block: attention + 4×-expansion GELU MLP, each with a residual connection. |
| `GPT` | The decoder itself, plus an autoregressive `generate()` sampler. |

### Latent conditioning — the key idea

The decoder is driven by a **continuous latent `z`** (size `decoder_input_size`), not by a
start token. The mechanism:

1. `z_proj: Linear(decoder_input_size → n_embd)` maps `z` into the Transformer's embedding
   space.
2. That projected vector becomes the **first token** of the sequence — a *prefix
   conditioning token*.
3. Causal masking guarantees every real token attends back to this prefix, so the whole
   generated sequence is a function of `z`. This is the property a VAE decoder needs.

```
position:   0       1       2      ...   T
input:    [ z ]  [tok_0] [tok_1]  ... [tok_{T-1}]
            │       │       │            │
logits:   pred_0  pred_1  pred_2  ...  pred_T   ← pred_t predicts token_t
```

`pred_0` sees only `z` and predicts the **first** action — the prefix supplies the `+1`
autoregressive shift, so no BOS token is needed.

### Public API

- `forward(z, idx=None, targets=None) -> (logits, loss)` — `logits` shape `(B, T+1, vocab)`.
  Passing `targets` computes the cross-entropy internally (used for convenience; the VAE
  computes its own loss externally).
- `generate(z, max_new_tokens=None, temperature=1.0, do_sample=False, top_k=None) -> idx`
  — free-running autoregressive sampling (KV-cached, see §6.1), returns integer tokens `(B, max_new_tokens)`.
- `init_decode(z)` / `decode_step(token, past, position)` — cached incremental-decoding
  helpers (§6.1) used by `generate` and `ActionGen.gen_action_seq`.

Model size at defaults: **~27K parameters** (was ~152K before the §6.3 shrink).

---

## 2. Mechanics changes — `dense_var_auto_encoder_vin_1.py`

The data representation flipped from **flattened one-hot** to **integer token indices**,
and the decoder/loss became **autoregressive**. The encoder side is unchanged.

### 2.1 `DenseVAE.__init__`

- **Removed** the MLP `self.decoder` (the `Linear → InstanceNorm → LeakyReLU …` stack) and
  `self.decoder_last_layer = nn.Sigmoid()`.
- **Added** `self.decoder = GPT(GPTConfig(vocab_size=n_words, block_size=input_length,
  decoder_input_size=decoder_input_size))`.
- The GPT is built **after** `self.apply(self._init_weights)` so the Transformer keeps
  minGPT's own `normal(0, 0.02)` init instead of being overwritten by the encoder's
  Kaiming-leaky_relu scheme.
- Stored `self.input_length` and `self.n_words` for use in `forward`.

### 2.2 `encode` / `decode` / `forward`

- `encode` — **unchanged**. Still consumes a flattened one-hot view and outputs
  `mean, logvar`.
- `decode(z, idx)` — now teacher-forces the Transformer: `logits, _ = self.decoder(z, idx)`,
  returns logits `(B, T+1, n_words)`. (Previously returned `(sigmoid_probs, logits)`.)
- `forward(x, get_z=False)` — **input is now integer tokens `(B, T)`**. It one-hots them
  internally to feed the encoder, samples `z`, then teacher-forces the decoder on the true
  tokens. **Returns `(logits, mean, logvar[, z])`** — note this is **3 (or 4) values**, down
  from the previous 4 (`x_hat, mean, logvar, x_hat_logits`).

### 2.3 `loss_function`  ← signature changed

```python
loss_function(logits, targets, mean, log_var, desired_var, label_smoothing=0.1)
```

- Reconstruction term is now **`F.cross_entropy`** over the vocabulary
  (`reduction='sum'`), replacing the per-element BCE on one-hot vectors.
- Alignment: `logits[:, :T]` vs the full `targets` — the `+1` shift is supplied by the
  z-prefix at position 0.
- **KLD term is kept verbatim.**
- Old params `x` / `x_hat` / `x_seq_len` / `smoothing` are gone.

### 2.4 `MazeDataLoaderV2` — integer-token mode

- New constructor flag **`return_token_indices: bool = True`**.
- `convert_action` returns the bare integer class id when in token mode.
- `prepare_list` emits `(seq_len,)` `LongTensor`s; stacking a batch yields `(B, seq_len)`.
- Padding still uses token `4` (EOS).
- The soft-label jitter augmentation (writing floats into one-hot tensors) is **guarded off**
  in token mode — it's meaningless for integer indices.

### 2.5 `get_reconstructed_action_list_with_embedding`

Updated to feed integer tokens and argmax `logits[:, :T]`. As a side benefit it no longer
depends on the hardcoded length-10 `convert_action_list_to_one_hot_tensor`.

---

## 3. Production inference change — `decoder_utils.py`

`ActionGen.gen_action_seq` was rewritten from a single `self.model.decoder(gen_input)` call
into a **token-by-token autoregressive loop**:

```
step_logits, past = self.model.decoder.init_decode(z)   # process the z-prefix once (KV cache)
for t in range(n_action_seq_length):
    row = step_emit(step_logits)             # mode-specific representation
    if t < n_action_seq_length - 1:
        next_token = argmax(row).detach()    # feed back the chosen token
        step_logits, past = self.model.decoder.decode_step(next_token, past, position=t + 1)
```

*(Originally a full-forward loop `self.model.decoder(z, idx)` each step; rewritten to the
KV-cached form in §6.1 — numerically identical, O(T) instead of O(T²).)*

- The original branching is **fully preserved** — the selected `step_emit` is one of:
  temperature-scaled softmax (probs path), Gumbel-Softmax, straight-through, or argmax
  one-hot — chosen from `get_actions_as_one_hot`, `deterministic_mode`,
  `deterministic_inference`, `exclude_decoder_from_computation_graph`, `use_gumble`.
- **Output shape is unchanged:** `(batch, n_action_seq_length, n_words)`, so existing callers
  (e.g. `decoder.gen_action_seq(...)[1].argmax(-1)`) keep working.
- **Gradients still reach `z`** via the prefix token at every step. Fed-back tokens are
  detached integers (standard straight-through / Gumbel practice — we don't differentiate
  through the discrete sampling history). The old `decoder_last_layer` (Sigmoid) calls were
  replaced with `softmax`, which is the correct posterior for the cross-entropy-trained model.

---

## 4. ⚠️ Still needs updating / fixing / to notice

These spots still use the **old MLP / one-hot API** and will raise at call time. They were
left untouched because they involve evaluation design choices — fix before training/eval.

### 4.1 `__main__` training loop — **BROKEN**  ([dense_var_auto_encoder_vin_1.py:530-548](dense_var_auto_encoder_vin_1.py#L530-L548))

- `reconstructed_batch, mean, log_var, _ = model(flatten_batch)` unpacks **4** values;
  `forward` now returns **3**.
- `loss_function(flatten_batch, reconstructed_batch, mean, log_var, image_len_batch_l,
  desired_var=VAR)` uses the **old signature**.
- `flatten_batch = image_batch_tensor.flatten(1)` is no longer wanted — the model expects
  integer tokens `(B, T)`, i.e. `image_batch_tensor` directly.

**Fix:**
```python
token_batch = torch.stack(image_batch_l)          # (B, T) long
logits, mean, log_var = model(token_batch)
loss = loss_function(logits, token_batch, mean, log_var, desired_var=VAR)
```

### 4.2 `get_model_performance_on_set` — **BROKEN**  ([dense_var_auto_encoder_vin_1.py:375-399](dense_var_auto_encoder_vin_1.py#L375-L399))

- Unpacks 4–5 return values from `eval_model(...)` (now 3–4).
- `inverse_transform_sequence(recon_seq)` assumes one-hot output; `recon_seq` is now logits
  `(B, T+1, vocab)`.
- Calls `loss_function(...)` with the old signature.

**Fix direction:** get `logits, mean, logvar = eval_model(token_seq.unsqueeze(0))`, take
`recon_act_seq = logits[:, :T].argmax(-1)`, compare directly to the integer ground-truth,
and call the new `loss_function`. Decide whether to measure reconstruction with **teacher
forcing** (cheap, what `forward` gives) or **free-running `generate()`** (matches production
but slower).

### 4.3 `evaluate_k_seq_in_decoder` — `DenseVAE` branch **BROKEN**  ([dense_var_auto_encoder_vin_1.py:442-444](dense_var_auto_encoder_vin_1.py#L442-L444))

`decoder.decoder(rand_noise.unsqueeze(0)).reshape(-1, 10, 5)` no longer works — `decoder.decoder`
is a GPT returning a `(logits, loss)` tuple from a single prefix step, not a full sequence.

**Fix:** `rand_action_seq = decoder.decoder.generate(rand_noise.unsqueeze(0))` → already
`(B, 10)` integer tokens, no reshape/argmax needed. *(The `ActorGen` else-branch is already
correct.)*

### 4.4 `inverse_transform_sequence` / `transform_sequence`  ([dense_var_auto_encoder_vin_1.py:364-372](dense_var_auto_encoder_vin_1.py#L364-L372))

With `input_size == 1` in token mode, `reshape(-1, 1).argmax(1)` returns all zeros — these
helpers no longer fit the integer-token format. Only `get_model_performance_on_set` uses
them; retire or rewrite them as part of 4.2.

### 4.5 Checkpoints — incompatible

The decoder's `state_dict` keys changed (MLP → GPT). **Old `.pt` weights will not load** via
`get_decoder(load_pretrained_weights=True)` in [decoder_utils.py](decoder_utils.py). Retrain
and re-save.

### 4.6 Things to notice (not bugs)

- **Loss scale / β:** reconstruction is now CE with `reduction='sum'`; this changes its
  magnitude relative to the (unchanged) summed KLD. Expect to **retune the KLD weight (β)**
  to avoid posterior collapse or under-reconstruction.
- **EOS padding in the loss:** CE is computed over **all** positions including the EOS
  padding tail. The model will spend capacity learning to emit EOS for padding (usually
  fine for an EOS-terminated scheme). If it dominates, consider masking after the first EOS
  or `ignore_index`.
- **Generation cost:** ~~`gen_action_seq` / `generate` recompute the full forward each step
  (no KV cache) — O(T²)~~ **Fixed in §6.1** — a KV cache makes each step O(1) in the prefix.
- **`InstanceNorm1d` warning:** the encoder emits a benign size warning on 2-D input — this
  is **pre-existing**, unrelated to the Transformer change.
- **Unused leftover:** `convert_action_list_to_one_hot_tensor` is no longer called after the
  `get_reconstructed_action_list_with_embedding` rewrite.

---

## 5. Verification done so far

- `mingpt_decoder.py`: import + forward + `generate` smoke test (logits `(B, 11, 5)`).
- `DenseVAE`: forward → `loss_function` → `backward`; gradients reach `decoder.z_proj`.
- `MazeDataLoaderV2`: integer-token output `(B, 10)`, padded with `4`.
- `ActionGen.gen_action_seq`: all five emission modes produce `(B, 10, 5)`; probs sum to 1;
  one-hot modes are valid one-hots; Gumbel and STE paths propagate gradient to `z`.

**Not yet exercised:** a full end-to-end training run (blocked on §4.1–4.3) and loading a
real action-sequence pickle.

---

## 6. Follow-up fixes (performance + quality)

Applied after end-to-end RL runs were working, to address (a) very long runtimes and
(b) a from-scratch GPT under-performing the from-scratch MLP.

### 6.1 KV cache — faster autoregressive decoding  (`mingpt_decoder.py`)

- `CausalSelfAttention` and `Block` now accept `layer_past` / `use_cache` and return
  `(output, present)`. The causal-mask slice `[Tk-Tq:Tk, :Tk]` generalises the old
  `[:T,:T]` mask (identical for the full teacher-forced pass).
- New `_forward_trunk`, **`init_decode(z)`**, **`decode_step(token, past, position)`** keep a
  per-layer key/value cache, so each generation step only computes the *new* token's
  attention instead of re-encoding the whole growing prefix — **O(T) instead of O(T²)**.
- `generate()` and `ActionGen.gen_action_seq` rewritten to use them.
- **Verified numerically identical** to the non-cached decode (max logit diff ~2e-7);
  gradients to `z` preserved for the Gumbel/STE training paths. ~3× faster per call on a
  CPU microbenchmark. (Supersedes the old "no KV cache" note in §4.6.)

### 6.2 `no_grad` audit — no change needed  (`gps_simplegrid_levels.py`)

All *inference* `gen_action_seq` call sites already run under `torch.no_grad()` (eval
functions are `@torch.no_grad()`-decorated; the rollout/target/diversity decodes sit inside
`with torch.no_grad()` blocks). The one **E2E-training** decode correctly keeps its graph.
Nothing to change — the acting path was never building throwaway graphs.

### 6.3 Shrink the decoder  (`GPTConfig` defaults)

`n_layer 3→2`, `n_embd 64→32` (`n_head=4`, `decoder_input_size=16`, `vocab_size=5`,
`block_size=10` unchanged). Params **152,512 → 26,688** (~5.7×). Rationale: the default
minGPT size was ~60× the old MLP decoder (~2.5K) for a 16-dim→50-logit map — over-sized for
length-10/vocab-5, slower, and harder to optimise from sparse RL reward.

### 6.4 Conditioning-aware init — make `z` drive the decoder  (`GPT.__init__`)

The generic `_init_weights` only touches `Linear`/`Embedding`/`LayerNorm`, so two conditioning
parameters were mis-initialised. Two overrides applied **after** `self.apply(_init_weights)`:

- **`pos_emb`** was an `nn.Parameter(zeros)` and stayed **all-zeros** the whole run (positions
  indistinguishable). Now `normal(std=0.02)`, the standard GPT positional init.
- **`z_proj`** got the tiny trunk `std=0.02`; bumped to `std = 1/√decoder_input_size ≈ 0.25`
  (~10×). Because the residual stream is *not* normalised (only branch inputs are), the larger
  prefix makes **`z` dominate the residual reaching the head**, so different `z` produce
  different outputs from step 0. The scale is a tunable knob.
- **Measured effect:** unique greedy sequences over 64 random `z` rose **12 → 29** at init.

### 6.5 Context / open status

- From-scratch E2E (random init, RL reward only): GPT scored **~0.385** test success vs the
  from-scratch MLP's **~0.794**. §6.3–6.4 target that gap (over-parameterisation + z-agnostic
  init); §6.1 targets runtime.
- **Recalibration:** "valid actions per episode" is ~1.6 even for the strong MLP, so short
  per-decision sequences are *normal* for this task — the real gap is **success rate**, not
  sequence length. (An earlier read treated the GPT's 1.29 as a collapse; the MLP's 1.599
  shows that was overstated.)
- **Open:** a post-shrink/-init GPT run has not yet been measured — that's the test of whether
  §6.3–6.4 close the gap toward ~0.79. If it plateaus below target, the next lever is
  **reconstruction pretraining** (which also needs the §4.1–4.3 standalone-loop fixes).
- **Superseded by §7:** the ~0.385 figure above was measured while two E2E training bugs
  (§7.1) were active, so it does not measure the architecture. It needs re-measuring.

---

## 7. Runtime investigation + E2E correctness fixes (round 2)

Prompted by runtime: the MLP baseline completed in **10:24:26** (slurm 18136376, `cs-1080-02`,
`gtx_1080`) while the GPT run took **2-02:19:29** (slurm 19543017, same GPU class) — a 4.8×
regression. Investigation found the regression was only partly the Transformer; two bugs were
amplifying it and simultaneously preventing the decoder from training.

### 7.1 Two bugs in the E2E path  (`gps_simplegrid_levels.py`)

Both are specific to `--train_decoder_end_to_end`, and both were invisible with the old MLP
decoder — which is why they appeared only after the Transformer landed.

#### 7.1.1 `decoder_optimizer.zero_grad()` was never called

`critic_optimizer.zero_grad()` and `actor_optimizer.zero_grad()` were both present; the
decoder optimizer only ever received `.step()`. Because the decoder sits inside the actor's
computation graph, its gradients **accumulated across every actor update for the whole run** —
by step N the optimizer was stepping along the sum of N/4 gradients rather than the current
one. Adam normalises the magnitude, so this never produced a NaN or a crash; it silently
pointed the decoder in the wrong direction.

**Fix:** clear the decoder's gradients on the same schedule as the actor's.

**Verified:** decoder gradient norm now reads 1.0 → 16.7 over 30k steps. Accumulation would
have put it in the hundreds or thousands (7,500 summed updates at that point).

#### 7.1.2 The decoder never left `train()` mode

`decoder.model.to(device).train()` is set for E2E mode and nothing ever set it back.
`eval_model` put the *actor* and *critic* into eval mode but not the decoder. The old MLP
decoder had **no dropout layers at all**, so its train/eval mode was a no-op and nobody
noticed; minGPT has three (`embd_pdrop`, `attn_pdrop`, `resid_pdrop`, all 0.1), which were
therefore **injecting noise into every generated sequence** during action selection, the
target-Q computation, and all evaluation.

**Fix:** two parts —
- `eval_model` saves/restores the decoder's training flag and calls `.eval()`.
- New `--decoder_dropout` (**default 0.0**) matches the old MLP's behaviour, which also
  covers the rollout and target-Q paths that `eval_model` does not touch. Pass `0.1` to
  restore the previous behaviour.

#### Combined effect on runtime

A noisy/mis-trained decoder emits sequences that trim to length 1, so the agent makes far
more decisions per episode. `num_decoder_generations` saturates toward `max_episode_steps`
(75) instead of ~20, which multiplies the **evaluation** term — the largest single cost in a
production run — by roughly 10×. The bugs were both a correctness problem and the bulk of the
runtime regression.

### 7.2 Measurements

All on **GTX 1080 Ti (sm_61)**, torch 2.2.1+cu121, via `bench_decoder.py` and `sbatch/e*.sbatch`.

**Per call** (ms), `bench_decoder.py`:

| decoder | fwd b=1 | fwd b=256 | fwd+bwd b=256 |
|---|---|---|---|
| MLP (legacy, 1 pass) | 0.469 | 0.493 | 1.213 |
| GPT eager `L=2 embd=32` | 10.478 | 10.921 | 29.954 |
| GPT eager `L=1 embd=32` | 7.165 | 7.525 | 19.722 |
| GPT eager `L=1 embd=16` | 7.325 | 7.351 | 22.886 |
| **GPT cuda-graph** `L=2` | **1.228** | **5.051** | n/a (forward only) |
| GPT on CPU (+transfers) | 10.686 | 37.360 | n/a |

**The decoder is ~100% dispatch overhead, not arithmetic.** Three independent confirmations:
batch 1 and batch 256 cost the same to within 1%; `embd` 32→16 is 3.5× fewer params and no
faster; cost fits `3.73 ms + 3.91 ms × n_layer`, so even a 0-layer trunk would cost 7× the
entire old decoder — that constant is the 10-iteration Python loop itself.

**End-to-end** (30k steps, projected to 1M):

| run | config | projected |
|---|---|---|
| E1 | eval off, deterministic, eager | 19.67 h |
| E3 | eval on @5000, deterministic, eager | **30.54 h** |
| E5 | eval off, **non-deterministic**, eager | 11.52 h |
| E8 | eval on @5000, deterministic, **cuda graph** | **15.27 h** |

- **Evaluation costs 10.87 h** (E3 − E1) — 36% of a production run.
- **Determinism costs 8.15 h** (E1 − E5), a 1.71× penalty. Not the convolutions (`conv2d` is
  1.2% of the loop) but **cuBLAS**: `CUBLAS_WORKSPACE_CONFIG=:4096:8` exists only to satisfy
  `torch.use_deterministic_algorithms`, and `torch._C._nn.linear` is ~30% of the loop.
- **The CUDA graph halves the run** (E3 − E8, exactly 2.0×).

**Profile** (`cProfile`, 3k steps, `sbatch/e6_profile_trainloop.sbatch`): `_forward_trunk`
cumtime **120.96 s of 241.11 s — ~50% of the training loop is the decode loop**.
`torch._C._nn.linear` is 555,271 calls / 73.49 s, of which ~508k (91%) are the decoder
(56,450 trunk passes × 9 linears).

### 7.3 Performance changes

#### 7.3.1 Fused attention  (`mingpt_decoder.py`)

`CausalSelfAttention.forward` replaces the matmul / scale / `masked_fill` / softmax / dropout
/ matmul chain with one `F.scaled_dot_product_attention`. The causal mask buffer is now
`torch.bool` (what SDPA wants, no per-call conversion), and when `Tq == 1` — every cached
`decode_step` — the single query sits at the last position, so **every key is already in its
past and no mask is passed at all**. Mathematically identical.

Note: on sm_61 with fp32, flash is unavailable and this likely dispatches to the math
backend, so the win here is small on *this* hardware. It should matter on sm_75+.

#### 7.3.2 CUDA graph for the gradient-free decode paths  (`decoder_utils.py`)

`gen_action_seq`'s loop is extracted into `_generate(z, step_emit)`, so one code path serves
both eager and captured execution. `_generate_graphed` caches graphs keyed by
`(batch_size, emit_mode)`; enabled by **`--decoder_use_cuda_graph`** (default off).

The switch is **`torch.is_grad_enabled()`**, not a call-site flag: grad is off for action
selection, target-Q and evaluation — nearly all the calls — and on for the actor update,
which stays eager so gradients still reach `z` and the decoder parameters.

Three properties make this safe inside a *training* loop:

- **Weights stay live.** The graph holds pointers to parameter tensors and Adam mutates
  `param.data` in place, so replays see current weights. ⚠️ `handle_nan_weights`
  (`gps_simplegrid_levels.py`) does `param.data = torch.randn_like(...)`, which **rebinds** and
  would silently invalidate every captured graph. It is currently dead code — if it is ever
  wired up, the graph cache must be cleared alongside it.
- **The output is cloned.** A graph writes into a fixed buffer that the next replay
  overwrites, and the training loop re-reads `action_list_batch` after stepping the
  environment. Returning the live buffer would be silent data corruption.
- **Capture failure degrades, not crashes.** One warning, `use_cuda_graph = False`, continue
  eager — a 15-hour job should not die on an unsupported op.

**Verified (E8 vs E3, identical config, graph the only difference):** exactly three captures
and no fallback — `(256, gumbel)` target-Q, `(1, gumbel)` action selection, `(1, argmax)`
eval. Success rate 0.27 → 0.25 (within the ±0.044 standard error of a 100-episode val set);
valid actions/sequence 1.237 → 1.408; decoder generations 19.963 → 17.84. No corruption.

#### 7.3.3 Decoder shape + dropout as CLI flags

`--decoder_n_layer`, `--decoder_n_head`, `--decoder_n_embd`, `--decoder_dropout`, plumbed
`Config` → `get_decoder_api` → `get_decoder` → `DenseVAE` → `GPTConfig`. Defaults match the
previously hardcoded values, so nothing changes unless passed. Added so §7.2's sweeps needed
no file edits between runs.

### 7.4 Ruled out

| candidate | result |
|---|---|
| `torch.compile(mode="reduce-overhead")` | **Fails on sm_61**: `Triton only supports devices of CUDA Capability >= 7.0`. Untested on sm_75+, where it could also graph the *backward* pass that §7.3.2 leaves eager. |
| Decoder on CPU | No better at b=1, **3.4× worse** at b=256. |
| Shrinking the model | `embd` 32→16 gains 1.6%. The cost is not arithmetic. |
| Faster GPU | RTX 2080 Ti vs GTX 1080 Ti, same script: 10.73 h vs 11.52 h — **7%**. Dispatch overhead is CPU-side; GPU FLOPS are nearly irrelevant. |
| Replay-buffer sampling | Hypothesised that `random.sample` over a `deque` (O(n) indexing, ~12.5k hops × 256 draws per step) was significant. **Wrong** — all `deque` ops total ~0.02 s. |
| `TensorCache` on `trim_action_sequence_from_eos_tokens` | Real but minor: its key is `tuple(actions.tolist())`, forcing a GPU→CPU sync per call, ~4% of the loop — the key costs more than the ops it caches. **Not applied** (also an aliasing hazard: the function mutates `actions` in place, so a cached tensor lets one caller's edit surface in another's result). |
| Gradient-monitoring block | 103,736 `.item()` calls / 2.47 s (~1%) every 250 steps. **Not applied.** |

### 7.5 Status

Production config is `run_gps.sbatch` (deterministic, ~15 h) and `run_gps_nondet.sbatch`
(~9 h). Both pin **`--gpus=gtx_1080:1`** — the partition is heterogeneous (`gtx_1080`,
`rtx_2080`, `rtx_3090`) and a bare `--gpus=1` silently hands out other hardware, which makes
runtime incomparable to the 10.4 h MLP baseline. Both enable `--decoder_use_cuda_graph` and
disable the periodic test-set evaluations (`--eval_test_dataset_during_training_freq -1`),
which feed no model selection and no training signal — only logging. Verified: of the three
evaluation sites, **only the val evaluation ever saves a checkpoint**.

**Runtime: resolved.** 50.3 h → ~15 h deterministic, ~9 h non-deterministic, against a 10.4 h
MLP baseline on the same GPU class.

**Accuracy: open, and not addressed by any of the above.** §7.3 is mathematically neutral;
§7.1 is what should move the number. The §6.5 comparison (GPT ~0.385 vs MLP ~0.794) was
measured with both bugs active and must be re-run before it means anything.

For calibration when reading the new runs: at 30k steps (3% of a run) E3/E8 reached 0.25–0.27
val success with 1.24–1.41 valid actions per sequence. Per §6.5's own recalibration, ~1.6 is
normal for this task even for the 0.794 MLP, so short sequences are **not** a collapse signal.

**Resolved in §8:** the fixed code measures 0.432–0.488 test success — parity with the 0.509
GPT run, not an improvement. §8 has the diagnosis.

---

## 8. Accuracy investigation (round 3)

### 8.1 Three more correctness bugs

All pre-existing, all found by runs dying rather than by review.

| bug | symptom | fix |
|---|---|---|
| A non-finite gradient `return`ed out of `train()` | Run exits 0, reports "Finished", **no test evaluation at all** despite good checkpoints on disk (job 22379903, step 625476) | Set `continue_training = False` and unwind to the end of `train()`, which still runs the test evaluation. Optimizer steps are still skipped, so NaN gradients never reach the weights. |
| The actor-diversity logging block assigned to `action_list_batch` | `AttributeError: 'NoneType' object has no attribute 'argmax'` (job 22379902, step 650000 = 13 x 50000) | It calls `gen_action_seq` without `get_actions_as_one_hot`, which returns `(None, probs)`, clobbering the acting loop's variable. If an episode ended on that same step the verbose logger crashed. Now uses local names. Intermittent: needs a multiple of `evaluate_actor_diversity_every_k_epochs` *and* an episode ending on it. |
| `eval_with_pretrained_model` never reloaded the decoder | Test success 0.056 with `valid actions per sequence = 10.0` (a decoder that never emits EOS) | It loads actor and critic from the best-on-validation checkpoint but used the *live* decoder object. Invisible with a frozen pretrained decoder; under `--train_decoder_end_to_end` it paired a best-on-validation actor with a decoder from ~600k steps later. Now loads the matching `decoder` checkpoint and restores the caller's in a `finally` (the function also runs mid-training). |

`eval_checkpoint.py` + `sbatch/eval_checkpoint.sbatch` were added to recover test numbers from
runs that died before their final evaluation — minutes against checkpoints already on disk,
versus re-running for hours.

### 8.2 Measured accuracy after §7.1

| run | test success (stochastic) | deterministic inference |
|---|---|---|
| `earthy-bird-16` (deterministic) | 0.432 | 0.329 |
| `true-capybara-17` (non-deterministic) | 0.488 | 0.381 |
| **prior GPT run (`eager-lion-9`)** | **0.509** | — |
| **MLP baseline** | **~0.794** | — |

**The §7.1 fixes did not improve accuracy.** At n=1000, SE ≈ 0.016, so 0.488 is
indistinguishable from 0.509 while 0.432 is genuinely below it.

Two calibration points for every future experiment:

- **Run-to-run variance is ~0.05** (0.432 vs 0.488 on the same config bar determinism), so
  anything under that is noise.
- **Deterministic inference costs ~0.1** in both runs. `deterministic_inference` defaults to
  `True`, so the deterministic pass is the pessimistic number; both are printed.

The 0.794 MLP run's summary shows `gradient/decoder_norm: 1255`, which means **§7.1.1's
accumulating-gradient bug was active in the MLP baseline too**. The bug applied to whichever
decoder was plugged in, so it cannot explain the architecture gap.

### 8.3 Diagnosis: the actor collapsed, not the decoder

`sequences/num_unique` collapses monotonically in the GPT runs while the MLP baseline
sustains ~40.8. That metric is end-to-end (observation → actor → `z` → decoded sequence), so
it cannot say which half failed — and the two halves have opposite fixes. `check_z_influence.py`
separates them, offline, from saved checkpoints.

Measured on `earthy-bird-16`, 256 samples:

```
PART A -- decoder, fed random z
  untrained decoder (control)    unique(trimmed)=20/256   mean_len=3.06
  trained decoder                unique(trimmed)=44/256   mean_len=1.77

PART B -- actor, fed real observations
  mean pairwise cosine similarity = 0.7783   (1.0 = all identical)
  std across embedding dims       = 0.0133
  sequences from actor z          unique(trimmed)=5/256   mean_len=1.01
```

**The decoder is healthy** — more z-responsive after training than at initialisation (44 vs
20 unique). **The actor collapsed**: near-constant embeddings landing in a region of `z`-space
that decodes to single-action plans. Five distinct outputs across 256 states.

**The method has degenerated into a 1-step policy.** Generate a multi-action plan and let the
critic choose among candidates has become "emit one action at a time" through an elaborate
actor → `z` → decoder pipeline. It also explains the ~20 decoder generations per episode seen
throughout §7. A length-1 plan is a sensible local optimum: it cannot commit to a wrong
multi-step move, so the critic prefers it.

### 8.4 `sequence_length_weight` was a no-op

`compute_sequence_lengths_differentiable` ended with `lengths = torch.floor(lengths)`.
`floor` has zero gradient everywhere, so the function — whose name states its purpose —
returned a constant with respect to the parameters. Both consumers (`sequence_length_weight`
and `invalid_actions_weight`) add their penalty to `actor_loss`, so those terms were **logged
but contributed no gradient** and could not influence training. The floor is removed.

### 8.5 Experiments

All at `--total_timesteps 500000` (validation peaks ~305k and declines, so the second half
only costs time) and `--no-torch_deterministic` (1.71x faster; these are exploratory).
Baseline to beat: **0.488**. Noise floor: **~0.05**.

| | config | targets |
|---|---|---|
| `sbatch/a1_multihead.sbatch` | `--n_proto_plan_candidates 5` | `n_proto_plan_candidates` defaults to 1, which makes candidate selection, `sample_from_candidates` and the per-head softmax weighting all no-ops. The five-element `min_lengths_per_head` defaults imply 5 was intended. Control for A2/A3. |
| `sbatch/a2_diversity.sbatch` | A1 + `--diversity_weight 0.1` | the 0.7783 cosine similarity. `compute_diversity_loss_with_normalized_embeddings` acts on the **actor's head embeddings**, so it needs >1 head to have pairs at all. |
| `sbatch/a3_length.sbatch` | A1 + `--sequence_length_weight 1e-4` | `mean_len = 1.01`, head-on. Per-head bands `[1,2,4,6,8]` force plans longer and heads apart. Only works because of §8.4. |

**Result — all three were worse, and all destabilised. See §8.8.**

| run | best val | best step | outcome |
|---|---|---|---|
| baseline (1 candidate) | **0.53** | **305,000** | — |
| A1 (5 candidates) | 0.37 | 60,000 | NaN crash at 3:25 |
| A2 (5 + diversity) | 0.44 | 60,000 | NaN crash at 2:35 |
| A3 (5 + length) | 0.43 | 60,000 | NaN crash at 3:19 |

Each peaked at step 60,000 and then degraded for ~250k steps before dying. These measured
instability, not the mechanism — **multi-head remains untested, not refuted.**

### 8.6 Open: forward response vs backward gradient

Part A tested whether the decoder responds to `z` in the **forward** direction. It says
nothing about **gradient quality in the backward** direction, and those are different
properties. So the actor's collapse may still be *caused* by the decoder:

- **(a) The actor was never the problem.** The MLP maps `z` → all 50 logits in one linear
  layer, so near-identical `z` still perturbs all 10 positions and yields different
  sequences. The GPT's autoregressive chain washes small `z` differences out against
  accumulating token history. Fix would be decoder-side.
- **(b) The MLP passed a stronger gradient to the actor.** Its path is critic → one Gumbel →
  one linear → `z`. The GPT's is critic → Gumbel at 10 steps → through the trunk each time →
  `z_proj` → `z`, with fed-back tokens **detached**. A weaker signal makes "emit a constant
  safe `z`" attractive. Consistent with actor gradient norms: MLP 0.387 vs GPT 0.197–0.265.

Under (b) the fix is **soft token feedback** — `row @ tok_emb.weight` instead of
`tok_emb(argmax(row).detach())`. With `hard=True` Gumbel the forward pass is numerically
identical (`row` is one-hot), but gradients then flow through the whole chain to `z`.

**Resolved — and no new code was needed.** The training loop already logs this metric
(`mean cosine similarity all pairs`, over the target actor's embeddings on a batch of
next-states), so the answer was in the old logs all along:

| | MLP (job 18136376, 0.794) | GPT (job 22379902, 0.432) |
|---|---|---|
| actor cosine, **early → late** | **1.0 → 0.507** | **0.706 → 0.619** |
| plan length (over *unique* sequences) | 3.67 | 3.17 |
| end-to-end unique sequences | **41.0** | **17.5** |

**The trajectory is the finding, not the endpoint.** The MLP's actor started *fully
collapsed* — cosine 1.0, one embedding for every state — and learned down to 0.507. Ours
started better (0.706) and barely moved (0.619). The MLP's actor learned; ours did not.

That points at the learning signal rather than the starting point, i.e. explanation **(b)**,
and matches the actor gradient norms (MLP 0.387 vs GPT 0.197–0.265).

**Correction to §8.3's phrasing.** "Degenerated into a 1-step policy" is right in practice
but was stated too strongly. `mean_len = 1.01` (from `check_z_influence.py`) averages over
**all 256 states**; the log's `mean length action sequences: 3.17` averages over the
**unique** sequences only. Both are true: nearly every state receives a 1-action plan, while
the ~17 distinct plans that exist average 3.17 actions. The decoder *can* produce long plans
— the actor almost never asks for one.

### 8.7 B2 — soft token feedback  (`--decoder_soft_token_feedback`)

The fix implied by (b). `ActionGen._generate` fed the next step a **detached argmax index**,
so gradient reached `z` only through the prefix token; the 10-step chain was a gradient dead
end. With the flag, the emitted row is fed back directly and `GPT.decode_step` multiplies the
embedding table instead of indexing it:

```python
token_embeddings = (tokens @ self.tok_emb.weight).unsqueeze(1)   # float row
token_embeddings = self.tok_emb(tokens.view(-1, 1))              # integer index
```

For a one-hot row these are mathematically identical, so **generation is unchanged — only the
backward pass differs.** Applied only to the hard-one-hot emission modes (`gumbel`,
`straight_through`, `argmax`): the scaled-probs row is a temperature-0.01 softmax, not a
one-hot, so feeding it back would change what the model generates rather than only how
gradients flow.

Default off, so it is A/B-able. `sbatch/b2_soft_feedback.sbatch` runs it at **one** proto-plan
candidate with no other regulariser, making it a single-variable comparison against the
0.53-val / 0.432-test baseline. (It was originally written with 5 candidates; see §8.8 for
why that was wrong.)

**Risk:** a 10-step differentiable chain is less stable than a 1-step one, and both full runs
already hit NaN gradients. Watch `gradient/decoder_norm` and `gradient/actor_norm`. Per §8.1
a NaN abort now still produces a final test evaluation, so an instability yields a number
rather than nothing.

---

## 9. Maximisation bias, and a brake that was never connected (round 4)

### 9.1 Why A1–A3 all destabilised

All three died the same way, 2.5–3.5h in:

```
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
  in select_action
```

The critic had produced **NaN Q-values**, and `torch.multinomial` refuses them. The crash
site is itself diagnostic: with `n_proto_plan_candidates=1`, `batch_q_stack.shape == (1, 1)`
takes the `q_idx = tensor([0])` branch and `multinomial` is **never reached**. That branch
had been dead in every previous run; 5 candidates made it live.

**The cause is the Bellman target:**

```python
qf_next_target = qf_next_target.view(cfg.batch_size, cfg.n_proto_plan_candidates)
best_candidate_indices = torch.argmax(qf_next_target, dim=1)      # max over candidates
qf_next_target = qf_next_target[range(cfg.batch_size), best_candidate_indices]
```

With 1 candidate that `argmax` is a no-op. With 5 it takes the maximum of five noisy
estimates — textbook **maximisation bias**. The inflated value becomes the training target,
the critic learns to predict higher, and it compounds until Q diverges to NaN. Hence the
peak at step 60,000 followed by 250k steps of decay.

### 9.2 `max_qf_value` has never been applied

The standard defence already exists in the code and has never run:

```python
if cfg.min_qf_value is not None:
    qf1_a_values = torch.clip(qf1_a_values, min=cfg.min_qf_value, max=cfg.max_qf_value)
```

`min_qf_value` defaults to `None`, so the clip is skipped and **`max_qf_value = 1.0` is dead
config** — in every run in this document, including the MLP baseline. Episodic returns span
roughly −20 to +1, so `--min_qf_value -20.0 --max_qf_value 1.0` activates it.

Worth enabling even at one candidate: both 1-candidate full runs also reached NaN gradients
(steps 625,476 and 918,000). Five candidates accelerate the divergence; they do not create it.

### 9.3 `select_action` now survives non-finite Q

A NaN there killed three 3-hour runs outright, and because they crashed they never ran their
final evaluation — the numbers in §8.5 had to be recovered from log greps. `select_action`
now falls back to candidate 0 with a warning. This fixes nothing; it means an instability
still yields a result, like the §8.1 abort.

### 9.4 Revised experiments — back to one candidate

A1 and A2 are shelved: A1 *is* the multi-candidate change, and the diversity loss compares
pairs of actor heads, so at one head the term is identically zero. Both need the §9.2 clip
before they are worth retrying.

| | config | note |
|---|---|---|
| `sbatch/a3_length.sbatch` | 1 candidate, `--sequence_length_weight 5e-4`, band `4`–`10`, `--min_qf_value -20.0` | Single-element length bands because the penalty asserts `len(min_lengths_per_head) == n_proto_plan_candidates`. Weight raised from 1e-4 since the sum is now over 256 terms, not 1280. |
| `sbatch/b2_soft_feedback.sbatch` | 1 candidate, `--decoder_soft_token_feedback`, `--min_qf_value -20.0` | Single variable against the 0.53-val baseline. |

### 9.5 Result — the length penalty works

| run | test (stochastic) | test (deterministic) | best val | best step | hours |
|---|---|---|---|---|---|
| baseline, deterministic, 1M | 0.432 | 0.329 | 0.49 | 330,000 | 11.29 |
| baseline, non-deterministic, 1M | 0.403 | 0.318 | 0.45 | 355,000 | 7.62 |
| **A3 — length penalty, 500k** | **0.582** | **0.511** | **0.61** | 265,000 | 4.26 |
| B2 — soft feedback, 500k | 0.479 | 0.400 | 0.55 | 225,000 | 5.50 |
| *prior GPT run (`eager-lion-9`)* | *0.509* | — | — | — | — |
| *MLP baseline* | *0.794* | — | — | — | — |

**A3 is the first GPT configuration to beat the 0.509 prior run** — by 4.6 SE (n=1000,
SE ≈ 0.016) — and it beats the fixed baseline by 0.15, three times the ~0.05 run-to-run
noise floor. Even its pessimistic deterministic-inference number (0.511) edges past the old
stochastic 0.509. It closes roughly **40%** of the gap between the fixed GPT baseline and
the MLP.

**The win came from deleting one line.** §8.4 removed `torch.floor` from
`compute_sequence_lengths_differentiable`; that was the only reason
`--sequence_length_weight` could affect training at all. The diagnosis in §8.3 predicted
the fix — the actor almost never asks for a long plan — and the fix delivered.

**B2 (0.479) is unproven**: above the 0.432 baseline, but inside the noise band.

Caveats: A3 is one run at one seed, and it ran 500k steps against the baseline's 1M (best
val arrived at 265k, so it was not cut short). Still well below the MLP's 0.794.

### 9.6 Logging fixes

**The final test metrics never reached wandb.** `eval_with_pretrained_model` logs them with
`c_writer.add_scalar(tag, value)` and no step, so tensorboard records them at step 0 — but
by then the run's counter is at `total_timesteps`, and `sync_tensorboard` requires
non-decreasing steps, so they were dropped. The headline numbers existed only in the slurm
logs, which is why every result in §8–§9 was recovered by grepping. They are now mirrored
into `wandb.run.summary`, which is step-independent and surfaces as columns
(`test-eval-with-critic/mean_episodic_success` and the `-deterministic` variant).

**A correct actor-embedding metric.** The logged `mean cosine similarity all pairs` reduces
`dim=2` of a `(batch, n_heads, embedding_dim)` tensor — the **head** axis, not the embedding
axis. At `n_heads=1` the vectors have length 1 and cosine degenerates to sign agreement; at
`n_heads=5` it measures something else entirely, which is why A2 reported 0.952 against a
1-head baseline of 0.619. The original is kept unchanged so the §8.6 trajectory stays
readable in its own units, and **`actor/embedding_cosine`** is added alongside: pairwise
cosine between per-state head-0 embeddings, reduced over the embedding axis, self-pairs
excluded. That matches what `check_z_influence.py` reports (0.7783 on `earthy-bird-16`) and
is comparable across head counts.

### 9.7 Wave D — three experiments against A3

A3 is now the control: it is the best configuration, so improvements stack on it rather
than on a baseline already beaten. Each run changes exactly one thing.

| | config | question |
|---|---|---|
| `sbatch/d1_a3_repeat.sbatch` | A3 unchanged | Is 0.582 real? `--no-torch_deterministic` makes this an independent sample without a seed change. |
| `sbatch/d2_a3_soft_feedback.sbatch` | A3 + `--decoder_soft_token_feedback` | Does soft feedback pay off once plans are long enough to have a chain worth differentiating? B2 alone was noise, but on length-1 plans (`mean_len = 1.01`) there is barely a chain for it to act on. |
| `sbatch/d3_a3_actor_lr.sbatch` | A3 + `--actor_learning_rate 3e-4` | Cheapest test of the weak-gradient theory: the actor's gradient norm was 0.197–0.265 vs the MLP's 0.387. If the signal is simply ~2x too small, this compensates. |
| `sbatch/d4_a3_stronger_length.sbatch` | A3 + `--sequence_length_weight 2e-3` (4×) | §9.8 shows the penalty is under-powered — `mean_len` settled at 2.53 against a band asking for ≥4. Does pushing further keep paying? |

### 9.8 How A3 actually won — and what it did *not* fix

`check_z_influence.py` on A3's checkpoint (`kind-bush-24`) against the baseline
(`earthy-bird-16`), 256 samples each:

| | baseline (0.432) | **A3 (0.582)** | change |
|---|---|---|---|
| **mean_len from actor `z`** | 1.01 | **2.53** | **+150%** |
| **unique plans from actor `z`** | 5/256 | **17/256** | **3.4×** |
| actor embedding cosine | 0.7783 | 0.7892 | unchanged |
| decoder on *random* `z` (unique / mean_len) | 44 / 1.77 | 47 / 1.99 | barely moved |
| `invalid - empty action sequence` | 12 | 3 | fewer |

**The penalty did what it was designed to do, and the score tracks the mechanism** — plans
more than doubled in length and 3.4× more distinct plans reach execution.

**But the actor did not become more state-dependent.** Cosine moved 0.778 → 0.789, i.e.
nowhere. Note the asymmetry between the two halves of the measurement: Part A (random `z`)
barely moved while Part B (actor `z`) moved a lot. Training only reshapes the region the
actor actually visits. So A3's win came from **the decoder learning to spread the actor's
narrow `z`-region across longer, more varied plans** — not from repairing the actor.

**The §8.3 actor collapse is therefore still fully intact**, and every actor-targeted lever
(diversity, learning rate, soft feedback) remains genuinely unexplored in terms of moving
that number. D2 and D3 are the first tests of it.

**And the penalty is under-powered.** `mean_len` equilibrated at 2.53 against a band asking
for ≥ 4 — the Q term pushes back and the balance settled below target. D4 raises the weight
4× to find out whether the lever has more to give.

One reason that is not a formality: §6.5 records the MLP baseline at ~1.6 valid actions per
sequence, and A3 is already at 2.53. Longer is not obviously monotonically better — there
may be an optimum nearby or already passed.

### 9.9 Wave D results

| config | seeds | mean |
|---|---|---|
| baseline | 0.432, 0.403 | 0.418 |
| B2 — soft feedback **alone** | 0.479, 0.481 | 0.480 |
| **A3 — length penalty** | 0.582, 0.611, 0.665 | **0.619** |
| **D2 — A3 + soft feedback** | **0.691** | — (1 seed) |
| D3 — A3 + actor lr 3e-4 | 0.445 | — (1 seed) |
| *MLP baseline* | *0.794* | — |

**The noise floor is ~0.042, not 0.03.** Three A3 seeds (0.582 / 0.611 / 0.665) give a
sample std of 0.042, so a 2σ result needs roughly **+0.084** over the 0.619 mean. The
earlier 0.03 figure came from a deterministic/non-deterministic pair, which differ in more
than seed.

#### D3 is refuted, and it invalidates the metric we were tracking

`actor/embedding_cosine` over matched sample indices (every 5000 steps):

| sample (≈steps) | 5 (25k) | 10 (50k) | 17 (85k) | 25 (125k) | 35 (175k) | 44 (220k) |
|---|---|---|---|---|---|---|
| D1 (A3 control) | 0.687 | 0.743 | 0.701 | 0.656 | 0.623 | 0.641 |
| D2 (soft feedback) | 0.794 | 0.762 | 0.629 | 0.421 | 0.363 | **0.337** |
| D3 (actor lr) | 0.825 | 0.782 | 0.555 | 0.373 | 0.339 | **0.375** |

D1 is flat — the actor barely learns, exactly the §8.3 collapse, and consistent with A3
leaving it untouched (0.778 → 0.789 in §9.8). D2 and D3 both diverge from the control
around 50–85k steps and end at roughly **half** its value. The first interventions to move
the actor at all.

**And they went in opposite directions on success: 0.691 vs 0.445.**

So **`actor/embedding_cosine` is a mechanism signal, not a performance proxy.** You can
halve it and lose 0.17. D3 — simply taking larger actor steps — achieves the diversity and
damages the agent. Actor collapse is real and diagnosable, but *forcing* diversity is not
the fix, and any future experiment judged on that metric alone will mislead.

#### D2 composes with A3, which is the result worth chasing

- soft feedback **alone** (B2, 2 seeds): 0.480 — *worse* than A3
- soft feedback **on top of A3** (D2, 1 seed): 0.691 — best so far

That is precisely what D2 was built to test. Gradient flowing through the decode chain has
almost nothing to work with when plans are length 1 (`mean_len = 1.01` on the baseline);
A3 is what makes them long. The two mechanisms appear to compose rather than overlap.

**Confirmed by D5** (second seed): 0.691 and 0.675, mean **0.683**, spread 0.016. Welch's t
against A3's three seeds gives t ≈ 2.5 — a real gain of +0.064.

### 9.10 Final wave D results — the length weight is the dominant lever

| config | seeds | mean | vs previous |
|---|---|---|---|
| baseline | 0.432, 0.403 | 0.418 | — |
| B2 — soft feedback **alone** | 0.479, 0.481 | 0.480 | worse than A3 |
| A3 — length 5e-4 | 0.582, 0.611, 0.665 | 0.619 | +0.20 over baseline |
| D2 — A3 + soft feedback | 0.691, 0.675 | 0.683 | +0.064 |
| **D4/D6 — length 2e-3** | **0.741, 0.709** | **0.725** | **+0.106 over A3** |
| D7 — length 2e-3 + soft feedback | 0.734 | — | +0.009 — **noise** |
| D8 — length 4e-3 | 0.742 | — | +0.017 — **noise** |
| *MLP target* | *0.794* | *(1 seed)* | |

Noise floor **0.042 (1σ)**, from A3's three seeds.

**The length penalty is the dominant lever, and it has plateaued.** The weight sweep reads
5e-4 → 0.619, 2e-3 → 0.725, 4e-3 → 0.742; the last step is inside the noise. D4 at 2e-3 is
the operating point.

**Soft feedback and the length penalty overlap — they do not compose.** Its contribution
shrank as the penalty strengthened:

- on A3 (5e-4): 0.619 → 0.683, **+0.064**
- on D4 (2e-3): 0.725 → 0.734, **+0.009**

Which is what the mechanism predicts. Soft feedback lets gradient flow through the decode
chain, and that is worth something when plans are long enough to have a chain worth
differentiating — but worthless once a stronger penalty has already forced them long. Two
routes to the same improvement, not two independent ones. Only testing the combination
revealed it; assuming additivity would have predicted ~0.805.

**Status: ~0.725–0.742 against the MLP's 0.794** — a gap of ~0.06, about 1.5σ, and the MLP
is a single seed. The configurations are no longer clearly distinguishable.

**The cheap levers are exhausted.** D7 and D8 both demonstrate that more knob-turning will
not close the remainder. The one substantial untried option is **reconstruction
pretraining** (§6.5): the decoder has never been shown an action sequence, having been
asked to learn sequence structure from sparse RL reward alone — which is the hardest
possible way to train a sequence model and throws away the Transformer's actual advantage.
Blocked on the §4.1–4.3 standalone-loop fixes.

**Best-known configuration** (`sbatch/d6_d4_repeat.sbatch`):

```
--n_proto_plan_candidates 1 --min_qf_value -20.0 --max_qf_value 1.0
--sequence_length_weight 0.002 --min_lengths_per_head 4 --max_lengths_per_head 10
--total_timesteps 500000 --no-torch_deterministic --decoder_use_cuda_graph
```

~4.3 h on a `gtx_1080`, against the 50.3 h / 0.509 run this investigation started from.

Read against A3's 0.582 with a 0.05 noise floor, so a real win clears ~0.63. Watch
`actor/embedding_cosine` as well as the score — movement there is the mechanism signal,
independent of success rate.

**Parked, with reasons:**

- **Multi-head** — genuinely untested rather than refuted (§9.1). Worth one retry now that
  `--min_qf_value` exists, but after wave D.
- **`z` at every position** — would strengthen conditioning, but §8.3 Part A already shows
  the decoder responds well to `z` (44 unique sequences from 256 random `z`, better than
  untrained). It reinforces something that works.
- **`--include_actor_embedding_in_critic_input`** (direct `dQ/dz`) — needs two fixes first:
  four call sites unpack 5 values from `DenseVAE.forward`, which returns 4; and those calls
  encode sub-sequences with the VAE encoder, which receives no gradient in from-scratch E2E
  runs, so the stored `z` would be noise from a random encoder. It also carries a design
  risk: with `z` in the critic's input the actor can raise Q by moving `z` without the
  decoded sequence improving.
- **Reconstruction pretraining** — the highest-ceiling option and §6.5's own recommendation:
  it attacks the real difficulty, a from-scratch autoregressive model learning sequence
  structure from sparse reward, and would train the encoder the item above needs. Blocked on
  the §4.1–4.3 standalone-loop fixes.
