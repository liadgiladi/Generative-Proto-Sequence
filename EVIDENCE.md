# Evidence log — Transformer decoder for GPS

**Read this before proposing experiments.** It is the current state and the rules that
earned themselves through failed runs. Full technical detail is in
[`generative/models/dense_autoencoder/TRANSFORMER_INTEGRATION.md`](generative/models/dense_autoencoder/TRANSFORMER_INTEGRATION.md)
§7–§9; this file is the orientation layer.

Branch: `transformer-integration` (PR #2 against `liadgiladi:main`, draft).
Task: `16x16_corridors_v1`, from-scratch end-to-end RL (`--train_decoder_end_to_end
--no-initialize_decoder_from_pretrained`). Multi-environment sweep in flight — §1.3.

---

## 1. Where things stand

### 1.1 The headline — the gap was the dataset, not the architecture

Everything in §1.4 below was measured at **100 training pairs**, the paper's protocol. At
that size both decoders are data-starved and the comparison measures starvation. Raising
the training set closes the gap completely:

| training pairs | GPT (tuned) | MLP | gap |
|---|---|---|---|
| 100 | 0.725 ± 0.018 (n=5) | 0.794 | −0.069 |
| 300 | 0.897 (g1) | *h2 in flight* | ? |
| **500** | **0.990** (g2) | **0.982** (h1) | **+0.008 — tied** |

So **the deficit was never architectural.** Every symptom diagnosed in §4 — the actor
collapse, the length-1 plans, the 200k stall, the 0.17 train/val gap — is a symptom of
overfitting 100 pairs, and all of them dissolve at 500. Do not spend another run on
capacity, heads, or optimiser tuning aimed at that 0.069.

### 1.2 The real result — sequence-level efficiency at tied success

g2 vs h1, both 500 pairs, both at the success ceiling. This is what GPS actually claims,
and it is invisible in accuracy alone:

| metric | GPT | MLP | |
|---|---|---|---|
| success | 0.990 | 0.982 | tied |
| **decoder generations/ep** | **3.36** | 13.34 | **4.0× fewer decisions** |
| **agent step ratio** | **0.936** | 0.573 | **near-optimal paths** |
| actions/ep | 10.1 | 25.1 | 2.5× shorter episodes |
| valid actions/seq | 3.01 | 2.06 | longer usable plans |
| revisits (global) | 0.12 | 1.94 | **16× less wandering** |
| wall-clock / optimal step | **0.0041s** | 0.0102s | 2.5× cheaper per unit progress |
| inference / decision | 0.0029s | 0.0025s | per-call cost ≈ equal |

Note the last two rows together: a single decode is *slightly more expensive* for the
Transformer, and it still wins 2.5× on wall-clock per optimal step, because it needs 4×
fewer of them. **That is the sequence-level argument, measured.**

### 1.3 Multi-environment sweep — in flight

Launched at 500 pairs across the six remaining maps (`j1_gpt_map` / `j2_mlp_map`); corridors
is done and is the table above. Results and the standing question go in **§8**.

### 1.4 The 100-pair study (superseded, kept for the mechanism it established)

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

> **Superseded by §1.1.** The gap is real at 100 pairs and gone at 500. It measured which
> decoder degrades more gracefully under data starvation, not which decoder is better.

**Runtime: parity, not a win.** 50.3h → 11.98h deterministic at 1M steps, against the MLP's
10:24:26 — and ours skipped the periodic test evaluations the baseline paid for (~1.3h), so
at matched logging it is ~28% slower. The 4× improvement over where this started is real;
"faster than the MLP" is not.

**The stall (§4) was also a 100-pair artifact** — it is overfitting a tiny training set, not
a capacity ceiling. §4 is kept because its *mechanism* findings (decoder healthy, actor
collapsed, what the length penalty actually changes) still hold and still inform
`check_z_influence.py` readings.

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
6. **Check the data regime before blaming the model.** `--train_dataset_size 100` is a
   starvation regime, and starvation looks exactly like a capacity ceiling: fast early
   learning, a hard plateau, a wide train/val gap. We spent wave D, F and most of §4 inside
   it. **Run one 500-pair control before any architectural hypothesis** — it is the same
   cost as the experiment and it decides whether the experiment is worth running at all.
7. **Success rate saturates; report the sequence-level metrics too.** At 500 pairs both
   decoders sit at ~0.99 and accuracy can no longer separate them — the separation is 4×
   in decisions per episode and 1.6× in path optimality (§1.2). Run `compare_runs.py`, not
   `grep mean success`.
8. **Pin the GPU type.** `--gpus=gtx_1080:1`, never a bare `--gpus=1` — the partition has
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
| **The MLP baseline tree is unpatched** | `~/gps-mlp-baseline` sits at 6f627a2, the paper's code drop, and **predates every §7–§9 fix**: accumulating decoder gradient, inference-time dropout, `torch.floor`. Deliberate — it is the published method as published — but **state the asymmetry when reporting**. The dropout one does not apply (the MLP has no dropout layers); the gradient one does. |
| **`--train_dataset_size` default (100)** | Not a neutral setting. See rule 6 — it is a starvation regime that mimics a capacity ceiling. Both `j1`/`j2` now default to 500; the paper's protocol is the third/second positional argument. |
| **`--train_dataset_size` is not comparable across maps** | It is an absolute count, drawn from pools that differ by 35×. At levels 1–14, 500 pairs is **24.8%** of 8x8_empty's train pool, **2.0%** of 16x16_empty's, **0.7%** of 24x24_empty's. Comparing success across maps at a fixed count conflates map difficulty with data coverage. Quote coverage, or set the size per map. |
| **`max_level` (default 14) caps the horizon on every map** | The dataset only ever contains start/goal pairs with optimal path ≤ 14 — so **no map tests long horizons**, whatever its size. 8x8's diameter is 14 (levels 1–14 = 100% of the map), 16x16's is 30 (76% of pairs usable), 24x24's is 46 (46%). **24x24 is short paths on a large canvas, not a longer horizon.** Any horizon-scaling claim needs `max_level` raised with the map, which also changes the pool and the data regime. |
| **1.0 success on 8x8** | Not leakage — the splits are disjoint (`simplegrid_dataset.py` partitions each level into `[:train]`, `[train:train+eval]`, `[train+eval:]`). It is saturation: a trivial empty grid with a quarter of its pairs in the training set. **8x8 success carries no information**; read the sequence-level metrics or drop the row. |
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
| `compare_runs.py` | **The reporting tool.** Parses both trees' logs for the full GPS metric set — decoder generations, agent step ratio, valid/unused actions, revisits, timing — and prints them side by side. `--csv` to write. Flags runs that died before their final eval rather than presenting partial numbers as comparable. |
| `sbatch/g1`–`g2_train_data_*.sbatch` | 300 / 500 training pairs. **The experiment that resolved the whole investigation** — see §1.1. |
| `sbatch/g3_d8_repeat.sbatch` | Second seed of length 4e-3 — decides whether the reportable figure is 0.726 or ~0.74. |
| `sbatch/g4_multihead_clipped.sbatch` | Multi-head retry with Q clipping and actor clipping, the guards A1–A3 lacked. |
| `sbatch/h1`/`h2_mlp_*pairs.sbatch` | The MLP at 500 / 300 pairs — the matched-data baseline for g2 / g1. |
| `sbatch/i1`–`i3_empty_*.sbatch` | `16x16_empty` at 100 pairs. Superseded by the `j`-series at 500; run them only for a paper-protocol comparison. |
| **`sbatch/j1_gpt_map.sbatch`**, **`j2_mlp_map.sbatch`** | **Parameterised by map** — the multi-environment sweep (§8). `j1 <map> [length_weight] [pairs]`, `j2 <map> [pairs]`; both default to 500 pairs. `j1 <map> 0` is the no-length-penalty arm. |

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

1. **The multi-environment sweep — in flight.** Fill in §8 as logs land. This is now the
   main line of work; everything below waits on it.
2. **Second seed on two or three maps**, once the sweep shows where the effect is largest.
   The sequence-level metrics have no measured noise floor yet — §1.2 is n=1 per arm, and
   a 4× ratio is almost certainly real while a 1.2× one is not.
3. **Reconstruction pretraining** — still the only substantial untried lever, and now a
   better bet than when it was parked: at 500 pairs the decoder is no longer starved, so
   a pretrained one has something to improve *on* rather than a collapsed actor to fight.
   A day of work on the §4.1–4.3 standalone loop before a single run.

**Do not** re-open capacity, multi-head, or length/soft-feedback tuning. All three were
aimed at a 0.069 gap that no longer exists.

---

## 7. Running and reading

```bash
sbatch sbatch/j1_gpt_map.sbatch 16x16_empty           # Transformer on any map, 500 pairs
sbatch sbatch/j2_mlp_map.sbatch 16x16_empty           # the MLP baseline for it
sbatch sbatch/d6_d4_repeat.sbatch                     # best-known config, corridors, ~4.5h
sbatch sbatch/eval_checkpoint.sbatch runs/<run-dir>   # recover a dead run's test numbers
```

**Reading results — use `compare_runs.py`, not grep.** Success rate alone is saturated at
500 pairs and will tell you the two decoders are identical (§1.2):

```bash
poetry run python compare_runs.py --csv all_maps.csv sbatch/logs/j1_*.out sbatch/logs/j2_*.out
grep "=== J1:" sbatch/logs/j1_*.out        # which map each log ran -- not in the filename
```

```bash
# single-run spot checks
grep -E "^test dataset with critic.* mean success" <log>
grep "best val mean global step" <log> | tail -1   # ~500k means it was cut short
grep "actor embedding cosine" <log> | head -2      # mechanism signal: early vs late
grep "actor embedding cosine" <log> | tail -2
```

In wandb, add the column **`test-eval-with-critic/mean_episodic_success`** once; it is
populated automatically for runs after the §9.6 fix.

A NaN no longer destroys a run's result: the gradient-monitoring abort and `select_action`
both degrade gracefully so the final test evaluation still runs against the best saved
checkpoint.

---

## 8. Multi-environment sweep — 500 pairs, GPT vs MLP

The paper evaluates 9 environments; **this repo ships 7** (both 24x24 obstacle maps are
absent). Corridors is done; the other six are in flight as of 2026-11-04.

```bash
MAPS="8x8_empty 16x16_empty 16x16_room_v1 16x16_obstacles_v1_15p 16x16_obstacles_v1_25p 24x24_empty"
for m in $MAPS; do
  sbatch sbatch/j1_gpt_map.sbatch "$m"
  sbatch sbatch/j2_mlp_map.sbatch "$m"
done
```

Fill in from `compare_runs.py`. Success is expected to be ~tied everywhere — **the
discriminating columns are decisions/ep and step ratio**:

| map | GPT succ | MLP succ | GPT gen/ep | MLP gen/ep | GPT ratio | MLP ratio |
|---|---|---|---|---|---|---|
| 16x16_corridors_v1 | 0.990 | 0.982 | **3.36** | 13.34 | **0.936** | 0.573 |
| 8x8_empty | | | | | | |
| 16x16_empty | | | | | | |
| 16x16_room_v1 | | | | | | |
| 16x16_obstacles_v1_15p | | | | | | |
| 16x16_obstacles_v1_25p | | | | | | |
| 24x24_empty | | | | | | |

### The three questions this answers

- **~~Does the advantage scale with horizon?~~** Not answerable here — `max_level 14` fixes
  the horizon on every map. See the caveat below. What the map size *does* vary is the
  observation and the data coverage, so a trend across sizes is a coverage trend.
- **Does path optimality survive clutter?** 15p → 25p obstacles. The 0.936 step ratio was
  measured on corridors, which reward committing to long straight runs. If it holds under
  clutter the result generalises; if the MLP catches up, it was corridor-specific.
- **Is anything clipped by the plan-length ceiling?** See the caveat below.

### Caveat — the sweep cannot answer the horizon question

`max_level` defaults to 14, so **every map is capped at 14-step optimal paths**. Map size
changes the observation and how much of the space 500 pairs covers; it does **not** change
the horizon. An earlier version of this section claimed 24x24 was the long-horizon case and
told readers to watch for the `n_actions_in_seq=10` ceiling there — wrong. That ceiling
(two generations minimum for a 14-step path) binds equally on every map.

Coverage at levels 1–14, which is what actually varies:

| map | usable pairs | 500 train pairs | diameter |
|---|---|---|---|
| 8x8_empty | 4,032 (100% of the map) | **24.8%** of the pool | 14 |
| 16x16_empty | 49,840 (76%) | 2.0% | 30 |
| 24x24_empty | 151,760 (46%) | 0.7% | 46 |

So **8x8 is saturated by construction** — a quarter of every pair that exists, on a trivial
grid. Expect 1.0 from both decoders and read only the sequence-level columns there. At the
other end, 24x24 at 0.7% is closer to the starvation regime of §1.1 than to corridors' 500.

A genuine horizon-scaling experiment needs `max_level` raised with the map (e.g. 14 / 30 /
46), which changes the pool size too and so needs the data regime re-fixed per map. That is
a different sweep, not a re-read of this one.

### Known asymmetries, to state when reporting

1. **500 pairs, not the paper's 100.** Deliberate (§1.1) — at 100 the comparison measures
   starvation. Any direct comparison to the paper's published numbers must use 100.
2. **The MLP tree is unpatched** — accumulating decoder gradient, see §3.
3. **1M steps for the MLP, 500k for the GPT.** Not an oversight: the MLP's best validation
   landed at step 999,999 on corridors (still climbing) while the Transformer peaks around
   200k. Equal step counts would handicap the MLP; this gives each its own best case.
4. **n=1 per cell.** No noise floor on the sequence-level metrics yet. A 4× ratio is safe
   to quote; anything under ~1.5× is not.
