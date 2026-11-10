from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from .dense_var_auto_encoder_vin_1 import DenseVAE


class ActionGen:

    def __init__(self, pretrained_decoder: DenseVAE, n_act_seq_len: int, device: torch.device, maze_n_actions: int = 4,
                 use_gumble: bool = True, penalize_cyclic_position_revisits: bool = False,
                 deterministic_inference: bool = True, use_cuda_graph: bool = False,
                 soft_token_feedback: bool = False):
        self.n_words = maze_n_actions + 1
        self.n_action_seq_length = n_act_seq_len
        self.model = pretrained_decoder
        self.device = device
        self.reverse_action_mapping = {0: 1, 1: 0, 2: 3, 3: 2}
        self.use_gumble = use_gumble
        self.penalize_cyclic_position_revisits = penalize_cyclic_position_revisits
        self.deterministic_inference = deterministic_inference

        # CUDA graph capture for the gradient-free decode paths. The autoregressive loop
        # issues ~550 tiny kernels per call and is ~100% dispatch overhead at our sizes
        # (measured: batch 1 and batch 256 cost the same to within 1%), so replaying it as
        # a single captured graph is worth ~8x at batch 1 and ~2x at batch 256.
        # Feed the emitted (one-hot) row back into the next decode step instead of a
        # detached argmax index, so gradient flows through the whole 10-step chain rather
        # than reaching z only via the prefix token. Generation is unchanged -- for a
        # one-hot row, row @ tok_emb.weight == tok_emb(argmax) -- so this alters the
        # backward pass only. Motivated by the actor failing to learn state-dependent
        # embeddings (cosine 0.706 -> 0.619 over a full run, where the MLP decoder's actor
        # went 1.0 -> 0.507); see TRANSFORMER_INTEGRATION.md §8.6.
        self.soft_token_feedback = soft_token_feedback

        self.use_cuda_graph = use_cuda_graph
        # Keyed by (batch_size, emit mode): the training loop only uses a handful of
        # shapes, and a graph is bound to exactly one.
        self._graph_cache: dict = {}

    @staticmethod
    def temperature_scaled_softmax(logits, temperature=1.0):
        logits = logits / temperature
        return torch.softmax(logits, dim=2)

    def gen_action_seq(self, gen_input, get_actions_as_one_hot: bool = False,
                       exclude_decoder_from_computation_graph: bool = False, deterministic_mode: bool = False):
        """Autoregressively decode an action sequence from the conditioning latent ``z``.

        The decoder is now a causal Transformer (``mingpt_decoder.GPT``) rather than a
        single-shot MLP, so the sequence is produced token-by-token: at each step the
        Transformer is conditioned on ``z`` (the prefix token) plus the tokens generated
        so far, and emits a distribution over the next token. The per-step distribution is
        turned into the same representation the old code returned -- temperature-scaled
        probabilities, a Gumbel-Softmax one-hot, a straight-through one-hot, or a plain
        argmax one-hot -- and the chosen token is fed back in for the next step.

        Gradients still reach ``z`` because the latent prefix conditions *every* step;
        the discrete tokens fed back are detached (we never differentiate through the
        sampling history), mirroring standard straight-through / Gumbel practice.

        Returns:
            ``(None, scaled_probs)`` when ``get_actions_as_one_hot`` is False, else
            ``(one_hot, None)``. Both sequence tensors keep the historical shape
            ``(batch, n_action_seq_length, n_words)``.

        ``gen_input`` may carry extra leading dimensions (e.g. a proto-plan candidates
        axis, ``(batch, n_candidates, decoder_input_size)``). These are flattened into a
        single batch dimension before decoding -- matching the old MLP decoder, whose
        ``decoder(z).reshape(-1, seq_len, n_words)`` collapsed them the same way -- so the
        returned tensor is ``(prod(leading_dims), n_action_seq_length, n_words)``.
        """
        # Collapse any leading dims into one batch dim (no-op when already 2-D); the GPT
        # decoder expects z of shape (N, decoder_input_size).
        z = gen_input.reshape(-1, gen_input.size(-1))

        def emit_scaled_probs(step_logits: torch.Tensor) -> torch.Tensor:
            # Temperature-scaled softmax over the vocabulary for a single step.
            # temperature_scaled_softmax expects (batch, seq, vocab) and softmaxes dim=2.
            return self.temperature_scaled_softmax(step_logits.unsqueeze(1), temperature=0.01).squeeze(1)

        def emit_argmax_one_hot(step_logits: torch.Tensor) -> torch.Tensor:
            # Deterministic argmax one-hot (no gradient) -- used for inference / when the
            # decoder is excluded from the computation graph.
            indices = torch.argmax(step_logits, dim=-1)
            return F.one_hot(indices, num_classes=self.n_words).float()

        def emit_gumbel_one_hot(step_logits: torch.Tensor) -> torch.Tensor:
            # Differentiable hard one-hot via the Gumbel-Softmax trick.
            return F.gumbel_softmax(step_logits, tau=1.0, hard=True)

        def emit_straight_through_one_hot(step_logits: torch.Tensor) -> torch.Tensor:
            # Argmax one-hot with a straight-through estimator for backprop.
            probs = torch.softmax(step_logits, dim=-1)
            indices = torch.argmax(probs, dim=-1)
            one_hot = F.one_hot(indices, num_classes=self.n_words).float()
            return (one_hot - probs).detach() + probs

        # Select the per-step emission function, preserving the original branching logic.
        # ``emit_key`` identifies the choice so that captured graphs are not shared between
        # emission modes (an argmax graph would silently serve the gumbel call sites).
        if not get_actions_as_one_hot:
            step_emit, emit_key = emit_scaled_probs, "scaled_probs"
        elif deterministic_mode:  # in_inference_mode
            if self.deterministic_inference:
                step_emit, emit_key = emit_argmax_one_hot, "argmax"
            else:
                step_emit, emit_key = emit_gumbel_one_hot, "gumbel"
        elif exclude_decoder_from_computation_graph:
            step_emit, emit_key = emit_argmax_one_hot, "argmax"
        elif self.use_gumble:
            step_emit, emit_key = emit_gumbel_one_hot, "gumbel"
        else:
            step_emit, emit_key = emit_straight_through_one_hot, "straight_through"

        # Replay a captured graph when there is no autograd graph to build. Grad mode is
        # the right switch: it is off for action selection, the target-Q computation and
        # evaluation (all of which dominate the call count), and on for the actor update,
        # which must stay eager so gradients still reach z and the decoder parameters.
        # Soft feedback only applies to the hard-one-hot emission modes. The scaled-probs
        # row is a temperature-0.01 softmax, not a one-hot, so feeding it back would change
        # what the model generates rather than only how gradients flow.
        soft_feedback = self.soft_token_feedback and emit_key in (
            "gumbel", "straight_through", "argmax"
        )

        if self.use_cuda_graph and z.is_cuda and not torch.is_grad_enabled():
            sequence = self._generate_graphed(z, step_emit, emit_key, soft_feedback)
        else:
            sequence = self._generate(z, step_emit, soft_feedback)

        if not get_actions_as_one_hot:
            return None, sequence
        return sequence, None

    def _generate(self, z: torch.Tensor, step_emit, soft_feedback: bool = False) -> torch.Tensor:
        """Autoregressive generation loop (KV-cached).

        ``init_decode`` processes the z-prefix once and returns both the logits predicting
        the first action and the key/value cache; each ``decode_step`` then feeds back only
        the newly chosen token, so attention over earlier positions is not recomputed
        (O(T) instead of O(T^2)). Gradients still reach z and the decoder params because
        the latent prefix (and the trunk) participate at every step.

        Returns ``(batch, n_action_seq_length, n_words)``.
        """
        generated_rows = []  # per-step emitted rows, each (batch, n_words)
        step_logits, past = self.model.decoder.init_decode(z, use_cache=True)
        for t in range(self.n_action_seq_length):
            row = step_emit(step_logits)
            generated_rows.append(row)

            if t < self.n_action_seq_length - 1:
                if soft_feedback:
                    # Feed the emitted row itself. For the hard-one-hot emission modes the
                    # forward pass is unchanged (row @ tok_emb.weight == tok_emb(argmax)),
                    # but gradient now flows back through every step of the chain instead
                    # of stopping at the detach below -- so the critic's signal reaches z
                    # through all 10 positions rather than only through the prefix token.
                    next_input = row
                else:
                    # Condition the next step on the token we just produced (greedy w.r.t.
                    # the emitted row, detached so no gradient flows through the discrete
                    # choice).
                    next_input = torch.argmax(row, dim=-1).detach()  # (batch,)
                step_logits, past = self.model.decoder.decode_step(
                    next_input, past, position=t + 1, use_cache=True
                )

        return torch.stack(generated_rows, dim=1)

    def _generate_graphed(self, z: torch.Tensor, step_emit, emit_key: str,
                          soft_feedback: bool = False) -> torch.Tensor:
        """Run ``_generate`` by replaying a captured CUDA graph.

        Capture records the concrete kernel sequence, so the KV cache's growing
        ``torch.cat`` shapes are fine -- they are constants at capture time. Two properties
        make this safe for a *training* loop:

        * The graph holds pointers to the decoder's parameter tensors, and the optimizers
          update those in place (``Adam`` mutates ``param.data``), so replays automatically
          see current weights. Nothing in this codebase rebinds a parameter to a new
          tensor, which would invalidate a graph silently.
        * RNG inside the graph (the gumbel path) is handled by PyTorch's graph-safe
          generator state, so each replay draws fresh noise rather than repeating capture.

        A capture failure is downgraded to a one-time warning and permanently falls back to
        eager, so an unsupported configuration slows a long run down instead of killing it.
        """
        key = (int(z.shape[0]), emit_key)
        entry = self._graph_cache.get(key)

        if entry is None:
            try:
                entry = self._capture_graph(z, step_emit, soft_feedback)
            except Exception as exc:  # pragma: no cover - depends on driver/GPU support
                print(f"[GPS] CUDA graph capture failed ({exc}); falling back to eager "
                      f"decoding for the rest of this run.", flush=True)
                self.use_cuda_graph = False
                return self._generate(z, step_emit, soft_feedback)
            self._graph_cache[key] = entry
            print(f"[GPS] captured decoder CUDA graph for batch={key[0]} emit={key[1]}",
                  flush=True)

        static_z, graph, static_out = entry
        static_z.copy_(z)
        graph.replay()
        # The graph writes into a fixed output buffer that the next replay overwrites, but
        # callers hold the returned sequence across subsequent decoder calls (the training
        # loop re-reads action_list_batch after stepping the environment), so hand back a
        # copy rather than the live buffer.
        return static_out.clone()

    def _capture_graph(self, z: torch.Tensor, step_emit, soft_feedback: bool = False):
        """Warm up and capture one (batch size, emit mode) variant of the decode loop."""
        static_z = z.detach().clone()

        # Warm up on a side stream first: lazy initialisation (cuBLAS handles, allocator
        # blocks) must not land inside the captured graph.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._generate(static_z, step_emit, soft_feedback)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self._generate(static_z, step_emit, soft_feedback)

        return static_z, graph, static_out

    def run_forward_actions(self, current_env, act_list):
        state = None
        done = False
        for act in act_list:
            observation, reward, truncated, _ = current_env.step(act)
            done = truncated
            state = torch.tensor(observation, dtype=torch.float32, device=self.device)
        return current_env, state, done


    def get_reward_per_sequence(self, current_env, act_list, old_state,
                                visited_positions_stats: Optional[dict] = None):
        def get_global_visited_positions(visited_positions_stats):
            """Extract global visited positions."""
            if visited_positions_stats is None:
                return None

            assert "prev_visited_positions" in visited_positions_stats
            assert "global_visited_positions" in visited_positions_stats
            return visited_positions_stats["global_visited_positions"] - visited_positions_stats[
                "prev_visited_positions"]

        def simulate_step(env, act):
            """Simulate the agent's next position based on the action."""
            row, col = env.agent_xy
            dx, dy = env.MOVES[act]
            target_row, target_col = row + dx, col + dy

            # Check if the move is within bounds and free
            is_valid = env.is_in_bounds(target_row, target_col) and env.is_free(target_row, target_col)
            return (target_row, target_col), is_valid

        def is_cycle(target_row, target_col, visited_positions, visited_positions_stats):
            """Check if the target position creates a cycle."""
            if (target_row, target_col) in visited_positions:
                return True  # Revisiting a position in the current sequence

            return False

        global_visited_positions = get_global_visited_positions(visited_positions_stats)
        total_reward = 0
        observation = current_env.envs[0].encode().transpose(2, 0, 1)[np.newaxis, ...] if visited_positions_stats is not None else None  # (1, 3, maze_size, maze_size), prevents None in case of cycle on the first step
        terminations = np.array([False])
        truncations = np.array([False])

        total_infos = {'agent_xy': np.empty((1, ), dtype=object), '_agent_xy': np.empty((0, ), dtype=bool)}
        n_steps = 0
        forward_actions = []
        infos = {'agent_xy': current_env.envs[0].agent_xy} if visited_positions_stats is not None else {}  # Handle the case of cycle on the first step

        observation_l = []
        reward_l = []
        terminations_l = []

        visited_positions = set([old_state])
        current_sequence_revisits_count = 0
        prev_sequence_revisits_count = 0
        global_sequence_revisits_count = 0

        # mapping = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        for idx, act in enumerate(act_list.tolist()):
            if visited_positions_stats is not None:
                _env = current_env.envs[0].unwrapped  # Unwrapped environment

                # Simulate the step to get the next position
                (target_row, target_col), is_valid = simulate_step(_env, act)

                # If the move is invalid, don't check for cycles
                if is_valid:
                    # Check for cycles if the move is valid
                    if is_cycle(target_row, target_col, visited_positions, visited_positions_stats):
                        break

            observation, reward, terminations, truncations, infos = current_env.step([act])
            observation_l.append(observation)
            reward_l.append(reward)
            terminations_l.append(terminations)
            n_steps += 1
            total_reward += reward

            state_xy = infos['agent_xy'][0]
            if not state_xy == old_state:
                forward_actions.append(act)
                old_state = state_xy

                # Check if this position has been visited before
                if state_xy in visited_positions:
                    current_sequence_revisits_count += 1

                if visited_positions_stats is not None:
                    if state_xy in visited_positions_stats["prev_visited_positions"]:
                        prev_sequence_revisits_count += 1

                    if state_xy in global_visited_positions:
                        global_sequence_revisits_count += 1

            # Add current position to visited positions
            visited_positions.add(state_xy)

            for key, val in infos.items():
                total_infos[key] = np.concatenate((total_infos.get(key, np.empty((0, ), dtype=bool)), val))

            if terminations or truncations:
                if self.penalize_cyclic_position_revisits:
                    total_reward -= -5 * current_sequence_revisits_count

                if visited_positions_stats is not None:
                    visited_positions_stats.update(
                        {
                            'global_sequence_revisits_count': global_sequence_revisits_count,
                            'prev_sequence_revisits_count': prev_sequence_revisits_count,
                            'current_sequence_revisits_count': current_sequence_revisits_count,
                            'visited_positions': visited_positions,
                        }
                    )


                return observation, total_reward, terminations, truncations, total_infos, n_steps, forward_actions, True, infos, observation_l, reward_l, terminations_l

        if self.penalize_cyclic_position_revisits:
            total_reward -= -5 * current_sequence_revisits_count


        if visited_positions_stats is not None:
            visited_positions_stats.update(
                {
                    'global_sequence_revisits_count': global_sequence_revisits_count,
                    'prev_sequence_revisits_count': prev_sequence_revisits_count,
                    'current_sequence_revisits_count': current_sequence_revisits_count,
                    'visited_positions': visited_positions,
                }
            )

        return observation, total_reward, terminations, truncations, total_infos, n_steps, forward_actions, True, \
               infos, observation_l, reward_l, terminations_l


def get_decoder_api(decoder_model_path: str, decoder_seq_len: int, device: torch.device, maze_n_actions: int = 4,
                    var_for_sample: int = 1, use_gumble_in_decoder: bool = True, penalize_cyclic_position_revisits: bool = False,
                    deterministic_inference: bool = False, load_pretrained_weights: bool = True,
                    decoder_n_layer: int = 2, decoder_n_head: int = 4, decoder_n_embd: int = 32,
                    decoder_dropout: Optional[float] = None,
                    decoder_use_cuda_graph: bool = False,
                    decoder_soft_token_feedback: bool = False) -> ActionGen:
    decoder = get_decoder(decoder_f_name=decoder_model_path, decoder_seq_len=decoder_seq_len, device=device,
                          maze_n_actions=maze_n_actions, var_for_sample=var_for_sample,
                          load_pretrained_weights=load_pretrained_weights,
                          decoder_n_layer=decoder_n_layer, decoder_n_head=decoder_n_head,
                          decoder_n_embd=decoder_n_embd, decoder_dropout=decoder_dropout)
    return ActionGen(pretrained_decoder=decoder, n_act_seq_len=decoder_seq_len, device=device,
                     maze_n_actions=maze_n_actions, use_gumble=use_gumble_in_decoder,
                     penalize_cyclic_position_revisits=penalize_cyclic_position_revisits,
                     deterministic_inference=deterministic_inference,
                     use_cuda_graph=decoder_use_cuda_graph,
                     soft_token_feedback=decoder_soft_token_feedback)


def get_decoder(decoder_f_name: str, decoder_seq_len: int, device: torch.device, maze_n_actions: int,
                var_for_sample: int = 1, load_pretrained_weights: bool = True,
                decoder_n_layer: int = 2, decoder_n_head: int = 4, decoder_n_embd: int = 32,
                decoder_dropout: Optional[float] = None):
    decoder = DenseVAE(input_length=decoder_seq_len, n_words=maze_n_actions + 1, device=device,
                       variance_for_sample=var_for_sample,
                       decoder_n_layer=decoder_n_layer, decoder_n_head=decoder_n_head,
                       decoder_n_embd=decoder_n_embd, decoder_dropout=decoder_dropout).to(device)
    if load_pretrained_weights:
        # Load state dict directly to the correct device
        decoder.load_state_dict(torch.load(decoder_f_name, map_location=device))
        decoder.to(device)
        decoder.eval()
    else:
        # Random initialization - ensure on correct device and keep in training mode
        decoder.to(device)
        decoder.train()
    return decoder
