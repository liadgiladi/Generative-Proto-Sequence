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
| soft feedback alone (B2) | 0.480 | 0.479, 0.481 | worse than A3 |
| length 5e-4 (A3) | 0.619 | 0.582, 0.611, 0.665 | |
| A3 + soft feedback (D2) | 0.683 | 0.691, 0.675 | real gain, +0.064 |
| A3 + actor lr 3e-4 (D3) | 0.445 | 1 | **refuted** — see §3 |
| **length 2e-3 (D4/D6)** ← **best** | **0.725** | 0.741, 0.709 | 500k, ~4.3h |
| length 2e-3 + soft feedback (D7) | 0.734 | 1 | +0.009 — noise |
| length 4e-3 (D8) | 0.742 | 1 | +0.017 — noise |

Runs are 500k steps unless noted; validation peaks ~265k–355k so 500k loses nothing.

**Best-known config** (`sbatch/d6_d4_repeat.sbatch`): 1 candidate,
`--sequence_length_weight 0.002`, bands 4–10, `--min_qf_value -20.0`, 500k steps,
`--no-torch_deterministic`, `--decoder_use_cuda_graph`.

**The length penalty is the dominant lever and it has plateaued** (5e-4 → 0.619,
2e-3 → 0.725, 4e-3 → 0.742; the last step is inside the noise). **Soft feedback overlaps
with it rather than composing**: worth +0.064 on top of A3, only +0.009 on top of D4 —
both are routes to exploiting longer plans, so once the penalty forces them long, the
gradient fix adds nothing. Assuming additivity would have predicted 0.805.

**P1 — the 1M-step run at the baseline's own step count** (`sbatch/p1_full_1m.sbatch`,
deterministic, matching the MLP's settings): **0.706 test, 11.98h, best validation at step
200,000.** Same as the 500k runs, so 500k was never a handicap and every comparison above
stands.

**Headline: 0.726 ± 0.018** — the strong-length-penalty family, five runs:

| run | config | test |
|---|---|---|
| D4 | length 2e-3 | 0.741 |
| D6 | length 2e-3 | 0.709 |
| P1 | length 2e-3, 1M steps, deterministic | 0.706 |
| D7 | length 2e-3 + soft feedback | 0.734 |
| D8 | length 4e-3 | 0.742 |

Pooling is justified because **no variant is distinguishable from another** (D7 +0.009 and
D8 +0.017 against the 2e-3 runs, both inside the noise), so these are five samples of one
effect rather than five different results. Restricting to the three repeated 2e-3 seeds
gives 0.719 ± 0.019 — the same picture, less data.

Note **0.74 was reached twice independently** (D4 at 0.741, D8 at 0.742), so the top of the
range is not a single lucky draw. Still, quote the **mean, not the maximum**: picking 0.742
because it is the largest cell is selecting on noise, especially against a baseline that is
itself a single seed.

**Gap to the MLP: 0.068** (0.794 − 0.726), ≈3.8σ at sd 0.018 — it does not look like noise.
An earlier note in this file called it "~1.5σ, not clearly distinguishable"; that used A3's
looser spread (0.042). The one caveat is that the MLP is a **single seed** with no variance
estimate of its own, so 0.794 could itself be a high draw.

**Runtime: parity, not a win.** 50.3h → 11.98h deterministic at 1M steps, against the MLP's
10:24:26 — and ours skipped the periodic test evaluations the baseline paid for (~1.3h), so
at matched logging it is ~28% slower. The 4× improvement over where this started is real;
"faster than the MLP" is not.

**The open problem is the stall — see §4.**

---

## 2. Rules that cost us runs to learn

1. **Noise floor is 0.019–0.042 (1σ), config-dependent.** Three A3 seeds gave
   0.582 / 0.611 / 0.665 (sd 0.042); three of the best config gave 0.741 / 0.709 / 0.706
   (sd 0.019). At n=1000 test episodes SE ≈ 0.016, so run-to-run variance dominates. **Use
   the looser figure when judging a new config** (a 2σ result needs ~+0.084) and the tighter
   one only when you have seeds of that exact config.
2. **Two seeds per config.** Almost every number in the integration doc is single-seed.
   `--no-torch_deterministic` makes a re-run an independent sample without changing `--seed`.
3. **One variable per run**, and **build on the current best**, not on a baseline already
   beaten. Wave D changes exactly one flag each against A3.
4. **Judge mechanism, not just score.** A change can be right and still land inside the
   noise. Track `actor/embedding_cosine` and plan length alongside success — a config that
   moves the mechanism but not the score has still taught you something.
5. **500k steps is enough — verified.** Validation peaks at 200k–355k and then declines;
   P1 ran the full 1M and scored 0.706, the same as the 500k runs. Check
   `best val mean global step` anyway: near the end of the run means it was cut short.
   (The MLP baseline is the exception — its best was at step 999,999, still climbing.)
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
| **`--actor_grad_clip`** | Applies to **both** the actor and the decoder. Their gradient norms differ by an order of magnitude — actor 0.2–0.4, decoder 2–16 — so a value that merely guards the actor throttles the decoder 2–16× every step. F2 died at step 61.5k where the unclipped F1 reached 159k. Clip the actor only, or add a separate decoder value. |
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

### The stall — the open problem

From `best val mean global step`:

| | best validation at step | still improving at 1M? |
|---|---|---|
| **MLP (0.794)** | **999,999** | **yes — 0.794 is a lower bound** |
| **Transformer (0.706 at 1M)** | **200,000** | no — plateaus, then declines |

And at **matched steps** the MLP is also ahead: **0.77 val at ~215k vs 0.72 at 200k.**

So this is **not** a sample-efficiency trade. The MLP dominates at every point *and* keeps
climbing; the Transformer simply stops after 200k. Any claim that the Transformer learns
faster is wrong — an earlier version of this file said so before the matched-step numbers
came in.

"Learns fast, then stalls" is the signature of a **capacity ceiling**, not a bad optimiser.
And there is a specific suspect: §6.3 shrank the decoder from `n_layer=3, n_embd=64`
(152,512 params) to `n_layer=2, n_embd=32` (26,688) **to save runtime** — and §7.2 later
measured the decoder as ~100% dispatch overhead, where `embd 32→16` is 3.5× fewer params
and 1.6% faster. **The shrink bought almost nothing and was decided before that was known.**

#### Capacity: two runs, two different failures — **unresolved, not refuted**

| run | config | outcome |
|---|---|---|
| **F1** | `n_layer 3, n_embd 64` | NaN at step **159,000** in the *actor* (`cnn.0.weight`); test 0.657 |
| **F2** | F1 + `--actor_grad_clip 1.0` | NaN at step **61,500**, same layer; test 0.391 |

Neither number is a capacity result — both are best-checkpoint scores from runs that died
before their peak (the 2e-3 family peaks around 200k).

**F1 was on the same trajectory when it died**, which is the important part:

| | best val | at step |
|---|---|---|
| P1 (`n_layer 2, n_embd 32`) | 0.64 | 140,000 |
| F1 (`n_layer 3, n_embd 64`) | **0.65** | 150,000 |

P1 went on to peak at 0.72 around 200k; F1 never got there. So "bigger is worse" is **not**
what was observed — the larger decoder was marginally ahead and then destabilised.

**F2's clip was misconfigured, and that is the lesson.** A single value was applied to both
networks, but their gradient scales differ by an order of magnitude: the actor runs 0.2–0.4
(so 1.0 barely bites) while the decoder runs 2–16. Clipping both at 1.0 cut the decoder
2–16× on every step — an effective learning-rate collapse, not a safety net. The actor then
absorbed more of the adaptation and failed *sooner*. Any retry must clip the actor only, or
use separate values.

**Verdict: dropped, not because capacity was refuted but because the expected value is low.**
F1 was ahead by 0.01 at matched steps, so even a clean win looks like +0.02–0.05, inside the
noise band. Two runs spent; a third is poor value at ~0.74.

Read the **stall step**, not only the score:

- best-val step **moves past 200k** → capacity was binding; sweep further (`n_layer 4`,
  `n_embd 128`) before anything else
- still **~200k at ~0.72** → capacity is not it; look at the length-penalty schedule
  (it contributes ~71% of the Q term at 2e-3 and keeps pushing after plans are long enough),
  critic saturation, or overfitting to the 100 training pairs
- **worse** → the larger decoder is harder to optimise from sparse reward, which is what
  §6.3 originally argued, and the question closes

Two free checks on `p1`'s log that narrow it further, not yet done:

```bash
L=$(ls sbatch/logs/p1_full_1m-*.out)
grep "losses/critic_values" "$L" | awk 'NR%20==0' | tail -10          # pinned near 1.0 = critic stops discriminating
grep "val dataset with critic mean success" "$L" | awk 'NR%20==0' | tail -10   # plateau vs decline
```

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
| `sbatch/e0`–`e8` | The runtime matrix (eval cost, determinism cost, CUDA-graph win). Settled; see §7.2. |
| `sbatch/a1`–`a3`, `b2`, `d1`–`d8` | Accuracy experiments. `a1`/`a2` are stale 5-candidate versions without `--min_qf_value`. `d6` is the best-known config. |
| `sbatch/p1_full_1m.sbatch` | The headline run: best config at the baseline's 1M steps, deterministic, matched settings. |
| `sbatch/f1_capacity_restore.sbatch`, `f2_capacity_clipped.sbatch` | Decoder capacity. Both NaN'd in the actor before reaching their peak — see §4. Dropped. |
| `sbatch/g1`–`g2_train_data_*.sbatch` | 300 / 500 training pairs, against the 0.17 train-val gap at the peak. |
| `sbatch/g3_d8_repeat.sbatch` | Second seed of length 4e-3 — decides whether the reportable figure is 0.726 or ~0.74. |
| `sbatch/g4_multihead_clipped.sbatch` | Multi-head retry with Q clipping and actor clipping, the guards A1–A3 lacked. |

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
- **Reconstruction pretraining** — **the only substantial lever left.** The decoder has
  never been shown an action sequence; it has been asked to learn sequence structure from
  sparse RL reward alone, which is the hardest possible way to train a sequence model and
  discards the Transformer's actual advantage. Would also train the encoder the item above
  needs. Blocked on the §4.1–4.3 standalone-loop fixes (~a day's work).
- **More knob-turning on length / soft feedback** — finished, and D7/D8 are the proof.
  Stacking soft feedback on the best config gained +0.009; doubling the length weight
  gained +0.017. Both inside the 0.042 noise floor. Do not re-sweep these.
- **Runtime work** — done. `torch.compile` fails on sm_61 (Triton needs ≥7.0); CPU decoding
  is worse; shrinking the model gains ~1.6% because the cost is dispatch, not arithmetic.
  *(That last measurement is why `f1_capacity_restore` is worth running: the §6.3 shrink
  traded capacity for a speedup that does not exist.)*

### Next, in order

1. **`f1_capacity_restore` — in flight.** Record its result in §4 under "The stall". Judge
   the **best-val step** first, the score second.
2. If capacity moves the stall → sweep further (`n_layer 4`, `n_embd 128`). Nearly free at
   runtime for the same reason.
3. If it does not → run the two free `p1` log checks in §4 to separate critic saturation
   from a length-penalty that keeps pushing after it has done its job. A decaying
   `sequence_length_weight` would test the latter.
4. Only then **reconstruction pretraining**, which is a day of work on the §4.1–4.3
   standalone loop before a single run.

---

## 7. Running and reading

```bash
sbatch sbatch/d6_d4_repeat.sbatch                     # best-known config, ~4.5h
sbatch sbatch/p1_full_1m.sbatch                       # 1M steps, deterministic, ~12h
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
