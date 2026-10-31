# Evidence log — Transformer decoder for GPS

**Read this before proposing experiments.** It is the current state and the rules that
earned themselves through failed runs. Full technical detail is in
[`generative/models/dense_autoencoder/TRANSFORMER_INTEGRATION.md`](generative/models/dense_autoencoder/TRANSFORMER_INTEGRATION.md)
§7–§9; this file is the orientation layer.

Branch: `transformer-integration` (PR #2 against `liadgiladi:main`, draft).
Task: `16x16_corridors_v1`, from-scratch end-to-end RL (`--train_decoder_end_to_end
--no-initialize_decoder_from_pretrained`).

---

## 1. Where things stand

| config | test success | seeds | note |
|---|---|---|---|
| **MLP decoder (the target)** | **0.794** | 1 | 10.4h, 1M steps |
| prior GPT run, pre-fixes | 0.509 | 1 | 50.3h |
| GPT baseline, post-fixes | 0.418 | 0.432, 0.403 | 1M steps |
| GPT + soft feedback alone (B2) | 0.480 | 0.479, 0.481 | worse than A3 |
| **GPT + length penalty (A3)** | **0.619** | 0.582, 0.611, 0.665 | 500k, ~4.3h |
| **A3 + soft feedback (D2)** ← best | **0.691** | 1 seed | needs confirming (D5) |
| A3 + actor lr 3e-4 (D3) | 0.445 | 1 seed | **refuted** — see §3 |

Runs are 500k steps unless noted; validation peaks ~265k–355k so 500k loses nothing.

**Runtime is solved**: 50.3h → 4–11h on the same GPU class. Do not spend more effort there
without a reason; see §7.2/§7.4 of the integration doc for what was already measured and
ruled out.

**Accuracy is the open problem.** A3 closes ~40% of the baseline→MLP gap. It is **one seed**.

Best-known config (`sbatch/a3_length.sbatch`): 1 proto-plan candidate,
`--sequence_length_weight 5e-4` with bands `4`–`10`, `--min_qf_value -20.0`, 500k steps,
`--no-torch_deterministic`, `--decoder_use_cuda_graph`.

---

## 2. Rules that cost us runs to learn

1. **Noise floor is ~0.042 (1σ).** Three A3 seeds gave 0.582 / 0.611 / 0.665. At n=1000
   test episodes SE ≈ 0.016, but run-to-run variance dominates it, so **a 2σ result needs
   ~+0.084**. A3's +0.20 over baseline clears that easily; D2's +0.072 over A3 is ~1.7σ and
   is *not* yet established.
2. **Two seeds per config.** Almost every number in the integration doc is single-seed.
   `--no-torch_deterministic` makes a re-run an independent sample without changing `--seed`.
3. **One variable per run**, and **build on the current best**, not on a baseline already
   beaten. Wave D changes exactly one flag each against A3.
4. **Judge mechanism, not just score.** A change can be right and still land inside the
   noise. Track `actor/embedding_cosine` and plan length alongside success — a config that
   moves the mechanism but not the score has still taught you something.
5. **500k steps is enough.** Validation peaks around 265k–355k and declines. Check
   `best val mean global step`: near 500k means the run was cut short and deserves 1M; near
   300k means the extra steps buy nothing.
6. **Pin the GPU type.** `--gpus=gtx_1080:1`, never a bare `--gpus=1` — the partition has
   `gtx_1080`, `rtx_2080` and `rtx_3090`, and one experiment silently landed on different
   silicon. (GPU model barely affects *speed* here — the workload is CPU-side dispatch
   bound, 7% between a 1080 Ti and a 2080 Ti — but it affects comparability.)

---

## 3. Traps — things that lie or silently do nothing

Each of these cost real time. Check them before trusting a measurement.

| trap | what happens |
|---|---|
| **`mean cosine similarity all pairs`** (logged) | Reduces the **head** axis of a `(batch, n_heads, embed)` tensor, not the embedding axis. At 1 head it degenerates to sign agreement; at 5 heads it measures something else. **Only comparable between runs with the same head count.** Use **`actor/embedding_cosine`** (added later) or `check_z_influence.py`. |
| **`actor/embedding_cosine`** — even the correct one | A **mechanism signal, not a performance proxy.** D2 and D3 both roughly halved it (0.337 / 0.375 vs a 0.641 control) and scored 0.691 and **0.445** — opposite directions. Forcing actor diversity can cost you 0.17. Never judge a config on this alone. |
| **`max_qf_value`** | Dead config unless `min_qf_value` is also set — the clip is gated on it being non-`None`. It was never active in *any* run in this repo, including the 0.794 MLP baseline. |
| **`--sequence_length_weight` / `--invalid_actions_weight`** | Were no-ops: `compute_sequence_lengths_differentiable` ended in `torch.floor`, which has zero gradient, so the penalties were logged and could not influence training. **Fixed** — and removing that one line is what produced A3's +0.15. |
| **wandb TEST columns showing `-`** | The final test metrics log under `test-eval-with-critic/*`; the columns usually displayed are the **periodic** `test-eval-with-critic-<step>/*`, which our configs disable via `--eval_test_dataset_during_training_freq -1`. Also, before the fix, final metrics were written with no step and dropped entirely. Runs before that fix have empty columns — their numbers are in the slurm logs. |
| **`mean length action sequences`** (logged) | Averages over **unique** sequences, not over all states. The log said 3.17 while the per-state mean was 1.01. Both true, very different meanings. |
| **`--diversity_weight`** | Compares *pairs of actor heads*. At `n_proto_plan_candidates=1` there are no pairs and the term is identically zero. Needs >1 head to do anything. |
| **`--sequence_length_weight` + head count** | Asserts `len(min_lengths_per_head) == n_proto_plan_candidates`. The 5-element defaults crash at 1 candidate — pass single-element bands. |
| **`n_proto_plan_candidates > 1`** | The Bellman target takes `argmax` over candidates — maximisation bias, a no-op at 1 candidate. At 5 it diverged: peak at step 60k, then 250k steps of decay, then NaN. Needs `--min_qf_value` before retrying. |
| **`handle_nan_weights`** | Dead code that rebinds `param.data`. If ever wired up it silently invalidates every captured CUDA graph. |
| **`torch_deterministic`** | Costs 1.71× runtime — via cuBLAS (`CUBLAS_WORKSPACE_CONFIG`), not the convolutions (`conv2d` is 1.2% of the loop). |
| **wandb SDK** | 0.16.4 cannot authenticate against the current backend (`CommError: 'NoneType' object has no attribute 'get'`), and its interactive `wandb login` rejects modern 86-char `wandb_v1_` keys with a 40-char check. Upgrade the SDK; write `~/.netrc` directly if the prompt refuses the key. Never commit the key — one was auto-revoked for exactly that. |

---

## 4. Diagnosis: what is actually wrong

Measured by `check_z_influence.py` on a trained checkpoint (256 samples):

```
PART A -- decoder, fed random z
  untrained decoder (control)    unique(trimmed)=20/256   mean_len=3.06
  trained decoder                unique(trimmed)=44/256   mean_len=1.77

PART B -- actor, fed real observations
  mean pairwise cosine similarity = 0.7783   (1.0 = all identical)
  sequences from actor z          unique(trimmed)=5/256   mean_len=1.01
```

**The decoder is healthy** — *more* responsive to `z` after training than at init.
**The actor collapsed**: near-constant embeddings, and it asks for 1-action plans almost
everywhere. The proto-plan method had degenerated into a 1-step policy.

Two candidate causes, still not fully separated:

- **(a) Safe local optimum.** A length-1 plan cannot commit to a wrong multi-step move, so
  the critic prefers it. *Supported by A3*: forcing plans longer gave +0.15.
- **(b) Weak gradient to the actor.** Actor gradient norm 0.197–0.265 vs the MLP's 0.387;
  across a full run the MLP's actor embedding cosine moved 1.0 → 0.507 while ours moved
  0.706 → 0.619 (old metric units). *Weakly supported*: B2, the purpose-built test, landed
  inside the noise at 0.479.

**Current reading: (a) is the stronger effect, (b) is unresolved.** Wave D's D2 tests
whether (b) only pays off once (a) is fixed — soft feedback has little chain to act on when
plans are length 1.

### What A3 changed, measured

Same diagnostic on A3's checkpoint (`kind-bush-24`, 0.582) vs the baseline above:

| | baseline (0.432) | A3 (0.582) |
|---|---|---|
| mean_len from actor `z` | 1.01 | **2.53** |
| unique plans from actor `z` | 5/256 | **17/256** |
| actor embedding cosine | 0.7783 | 0.7892 (unchanged) |
| decoder on *random* `z` | 44 / 1.77 | 47 / 1.99 (barely moved) |

**A3 did not fix the actor.** It made the decoder spread the actor's still-narrow
`z`-region across longer, more varied plans. Note Part A barely moved while Part B moved a
lot — training only reshapes the region the actor visits. So the actor collapse in §4 above
is still intact, and every actor-targeted lever remains untested.

`mean_len` also settled at 2.53 against a band asking for ≥ 4, so the penalty is
under-powered — D4 tests a 4× weight.

---

## 5. Tooling

| script | answers |
|---|---|
| `check_z_influence.py` | Actor collapse vs decoder collapse, offline from checkpoints. The diagnostic that produced §4 above. |
| `eval_checkpoint.py` / `sbatch/eval_checkpoint.sbatch` | Test-set evaluation against saved checkpoints — recovers numbers from runs that died before their own final eval. Minutes, not hours. |
| `bench_decoder.py` / `sbatch/e0_bench_decoder.sbatch` | Per-call decoder cost: MLP vs GPT vs CUDA-graph vs CPU. |
| `sbatch/e1`–`e8` | The runtime matrix (eval cost, determinism cost, CUDA-graph win). Settled; see §7.2. |
| `sbatch/a1`–`a3`, `b2`, `d1`–`d3` | Accuracy experiments. `a1`/`a2` are stale 5-candidate versions without `--min_qf_value`. |

Checkpoints live in `runs/<wandb-run-name>/{actor,critic,decoder}`, written whenever
validation success improves. **Only the validation evaluation ever saves**; the train and
test evaluations are logging only.

---

## 6. Parked, with reasons — do not re-propose without reading these

- **Multi-head (`n_proto_plan_candidates > 1`)** — *untested, not refuted*. The three runs
  that tried it were diverging from step 60k, so they measured instability. Worth one retry
  now that `--min_qf_value` exists.
- **`z` at every position** (instead of prefix-only conditioning) — plausible, but §4 Part A
  shows the decoder already responds well to `z`. It strengthens something that works.
- **`--include_actor_embedding_in_critic_input`** (direct `dQ/dz`) — needs two fixes first:
  four call sites unpack 5 values from `DenseVAE.forward`, which returns 4; and those calls
  encode sub-sequences with the VAE **encoder**, which gets no gradient in from-scratch E2E
  runs, so the stored `z` would be noise. Also a design risk: with `z` in the critic's
  input, the actor can raise Q by moving `z` without the decoded sequence improving.
- **Reconstruction pretraining** — highest ceiling, and §6.5's own recommendation. Attacks
  the real difficulty (a from-scratch autoregressive model learning sequence structure from
  sparse reward) and would train the encoder the item above needs. Blocked on the §4.1–4.3
  standalone-loop fixes.
- **Runtime work** — done. `torch.compile` fails on sm_61 (Triton needs ≥7.0); CPU decoding
  is worse; shrinking the model gains ~1.6% because the cost is dispatch, not arithmetic.

---

## 7. Running and reading

```bash
sbatch sbatch/d1_a3_repeat.sbatch          # ~4.5h
sbatch sbatch/eval_checkpoint.sbatch runs/<run-dir>   # recover a dead run's test numbers
```

```bash
# results from a finished run
grep -E "^test dataset with critic.* mean success" <log>
grep "best val mean success" <log> | tail -1
grep "best val mean global step" <log> | tail -1   # ~500k means it was cut short
grep "actor embedding cosine" <log> | head -2      # mechanism signal: early vs late
grep "actor embedding cosine" <log> | tail -2
```

In wandb, add the column **`test-eval-with-critic/mean_episodic_success`** once; it is
populated automatically for runs after the §9.6 fix.

A NaN no longer destroys a run's result: the gradient-monitoring abort and `select_action`
both degrade gracefully so the final test evaluation still runs against the best saved
checkpoint.
