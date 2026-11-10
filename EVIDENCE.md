# Evidence log — Transformer decoder for GPS

**Read this before proposing experiments.** It is the current state and the rules that
earned themselves through failed runs. Full technical detail is in
[`generative/models/dense_autoencoder/TRANSFORMER_INTEGRATION.md`](generative/models/dense_autoencoder/TRANSFORMER_INTEGRATION.md)
§7–§10; this file is the orientation layer.

**Writing a summary or a report from this file? Start at §9** — it names the metric the work
is about (replanning frequency), where to find it, and how it has to be qualified. Leading
with success rate will produce a misleading document.

Branch: `transformer-integration` (PR #2 against `liadgiladi:main`, draft).
Task: from-scratch end-to-end RL (`--train_decoder_end_to_end
--no-initialize_decoder_from_pretrained`) across all 7 maps the repo ships — §8.
Thesis direction after the baseline closes: **adaptive replanning with the Transformer**,
which §8's fixed-3-action finding is the motivation for.

---

## 1. Where things stand

### 1.1 The headline — the gap was the dataset, not the architecture

Everything in §1.4 below was measured at **100 training pairs**, the paper's protocol. At
that size both decoders are data-starved and the comparison measures starvation. Raising
the training set closes the gap completely:

| training pairs | coverage | GPT (tuned) | MLP | gap |
|---|---|---|---|---|
| 100 | 5.6% | 0.725 ± 0.018 (n=5) | 0.794 | −0.069 |
| 300 | 16.9% | 0.897 (g1) | *h2 in flight* | ? |
| **500** | **28.1%** | **0.990** (g2) | **0.982** (h1) | **+0.008 — tied** |

So **the deficit was never architectural.** Every symptom diagnosed in §4 — the actor
collapse, the length-1 plans, the 200k stall, the 0.17 train/val gap — is a symptom of
overfitting a tiny fraction of the space, and all of them dissolve at 500. Do not spend
another run on capacity, heads, or optimiser tuning aimed at that 0.069.

> **This table is corridors only — see §8 for the other six maps.** At matched 1M budgets the
> Transformer lands within 0.04 of the MLP on five of them and fails completely on
> `24x24_empty`. Corridors is the only map where it is ahead.
>
> Coverage (28.1% here vs 3–7% elsewhere) explains why 100 pairs looked starved on this map.
> It does **not** explain the cross-map results — the MLP reaches 0.973 on 24x24_empty at
> 0.8% coverage. An earlier version of this file claimed coverage was the ordering variable;
> it is not.

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

> **Largely confirmed by §8, with corridors the extreme case.** 64 free cells in 1-wide
> passages means almost no branch points, so the 4× decision advantage and the 0.936/0.573
> step-ratio gap are both bigger here than anywhere else. Across the other six maps at
> matched budgets the **decision-count advantage holds everywhere** (2–3×) and the
> **step-ratio advantage holds on five of seven** — it trails only on `obstacles_15p` and
> `24x24_empty`. Quote §8's table rather than this one: the margins here are corridors-sized.

### 1.3 Multi-environment sweep — complete (2026-11-09)

All seven maps, both decoders, 500 pairs, matched 1M budgets. **Near-parity on six maps
(within 0.04), one complete failure on `24x24_empty`** (0.110 vs 0.973, two seeds). The
Transformer wins only on corridors. Full table and the three sub-findings in **§8**:

- the 500k budget, not the architecture, produced the apparent rout — obstacles_25p went
  0.505 → 0.915 at 1M
- the length penalty is **load-bearing**, not the cause of the fixed-plan behaviour: at
  `weight 0` the agent collapses to 0.20–0.24 on every cluttered map
- `24x24_empty` is diagnosed: the decoder is healthy there, but the actor's `z` is 2.7×
  smaller **in magnitude, not in variety** (`std/norm` is identical to a working map), so the
  prefix is drowned out in an unnormalised residual stream
- at **100 pairs** the Transformer is flat at ~0.50 across four maps (range 0.033, below the
  noise floor) while the MLP ranges 0.40–0.93 — it is insensitive to map difficulty, because
  a ~20-plan repertoire cannot span what an open map needs

> **Writing this up? Read §9 first.** `decoder generations per episode` is the headline
> metric for the replanning thesis, not success rate — §9 says what it means, where to find
> it, and how it must be qualified.

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
5. **500k steps is enough on corridors only — it does not transfer.** Validation peaks at
   200k–355k there, and P1's full 1M scored the same. **On every other map 500k truncates.**
   Three sweep cells peaked at 470k–499,999 and recovered +0.107 to +0.410 when re-run at
   1M; reading them at 500k produced a conclusion that was wrong for two days. **Always
   check `best val mean global step` against the budget** — within ~10% of it means the run
   was cut short and its score is a floor, not a measurement. `j1` takes the budget as a
   4th argument.
6. **Check the data regime before blaming the model.** `--train_dataset_size 100` is a
   starvation regime, and starvation looks exactly like a capacity ceiling: fast early
   learning, a hard plateau, a wide train/val gap. We spent wave D, F and most of §4 inside
   it. **Run one 500-pair control before any architectural hypothesis** — it is the same
   cost as the experiment and it decides whether the experiment is worth running at all.
7. **Never judge a run on success rate alone.** On the easy maps it saturates at ~1.0 and
   separates nothing; on the hard ones it hides *why*. The informative axes are decisions
   per episode, path optimality and plan length — and they can point opposite ways from
   success (§8: the Transformer halves the decisions and still loses). Run
   `compare_runs.py`, not `grep mean success`, and see **§9** for which columns to report.
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
| **Cosine barely moves when the actor *does* differ** | A 0.961 checkpoint and a 0.099 checkpoint of the same config differ by **0.011** in actor cosine (0.7639 vs 0.7746) — indistinguishable — while `std across embedding dims` differs **2.8×** (0.0271 vs 0.0097), which is the quantity that tracks the failure (§8). Every actor diagnosis in §4 and wave D was read off cosine. **Prefer `std / norm`**, which `check_z_influence.py` now prints: cosine is scale-free but insensitive, raw `std` is sensitive but confounded by `z` magnitude, and the ratio is both. |
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
| **`--train_dataset_size` is not comparable across maps** | It is an absolute count drawn from pools that differ by orders of magnitude, because the maps differ in free space. 500 pairs is **28.1%** of `corridors_v1`'s train pool (64 free cells!), 5.8% of `room_v1`'s, 3.1% of `16x16_empty`'s, 0.7% of `24x24_empty`'s. **Comparing maps at a fixed count conflates difficulty with coverage** — the Transformer's 0.990 on corridors and 0.838 on rooms is mostly this. Quote coverage, or set the size per map (§8 has the table). |
| **`max_level` (default 14) caps the horizon on every map** | The dataset only ever contains start/goal pairs with optimal path ≤ 14 — so **no map tests long horizons**, whatever its size. 8x8's diameter is 14 (levels 1–14 = 100% of the map), 16x16's is 30 (76% of pairs usable), 24x24's is 46 (46%). **24x24 is short paths on a large canvas, not a longer horizon.** Any horizon-scaling claim needs `max_level` raised with the map, which also changes the pool and the data regime. |
| **`grep -m1` on an MLP log gives an early-training number** | `j2` does not pass `--eval_test_dataset_during_training_freq -1`, so MLP logs contain **many** `test dataset with critic mean success:` lines — the periodic evaluations — and the final one is last, not first. A `grep -m1` over the j2 sweep logs returned 0.12–0.33 where the true finals were 0.95–1.00, which inverted the entire sweep conclusion. **Use `compare_runs.py`** (it takes the last match) or `grep ... \| tail -1`. |
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
| **`sbatch/j1_gpt_map.sbatch`**, **`j2_mlp_map.sbatch`** | **Parameterised by map** — the multi-environment sweep (§8). `j1 <map> [length_weight] [pairs] [total_steps]`, `j2 <map> [pairs]`; both default to 500 pairs, j1 to 500k steps. `j1 <map> 0` is the no-length-penalty arm; `j1 <map> 0.002 500 1000000` matches the MLP's budget. |
| `sbatch/j3_dqn_map.sbatch` | Plain DQN, the non-GPS arm. **Not needed for the decoder comparison** — j1 and j2 are both GPS — and the paper already publishes DQN/TempoRL/DAR baselines. Kept in case a GPS-vs-RL claim is ever wanted in-house. |

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

**The baseline comparison is closed** (§8): near-parity on six maps at matched budgets, one
failure, and the length penalty confirmed load-bearing rather than harmful.

1. **Re-generate the sequence-level metrics from the 1M logs.** §8 and §9's `gen/ep`, step
   ratio and plan-length figures are all from the 500k runs. One `compare_runs.py` call.
2. **`24x24_empty` — diagnosed, one step from a fix.** The decoder is healthy there; the
   actor's `z` is 2.8× narrower than on a working map, and the actor reaches only 23% of the
   decoder's plan range against room's 43% (§8). Two things left, both cheap:
   - **re-run `check_z_influence.py` on both checkpoints** and read the new `std / norm`
     line. Low `std` *with* a low norm → scale `z_proj` (§6.4) or LayerNorm `z`, a
     decoder-side one-liner. Low `std` at a normal norm → genuine collapse, actor-side.
   - **add a Part-B-only mode** so the same probe runs against an MLP-tree checkpoint. That
     answers whether the actor narrows because the map is large or because the Transformer
     passes it a weaker gradient (§4(b)) — i.e. whether this is a map problem or a decoder
     problem.
3. **Then the thesis: make plan length state-dependent.** A policy that commits to 3 actions
   whether it is in a corridor or a cluttered room is doing fixed-stride, not sequence-level,
   decision making. Candidates: a learned stop token, per-state length conditioning, or
   letting the critic choose among candidate plans of differing length (needs
   `n_proto_plan_candidates > 1`, see §3). **It must match the length penalty, which is
   load-bearing** — at `weight 0` the agent collapses to 0.20–0.24 (§8).
4. **Reconstruction pretraining** — unchanged in priority; addresses sequence *quality*, not
   the length rigidity.

**Do not** re-open capacity, multi-head-for-accuracy, or soft-feedback tuning: all were
aimed at the corridors-only 0.069 gap, which §1.1 shows was data starvation.

---

## 7. Running and reading

```bash
sbatch sbatch/j1_gpt_map.sbatch 16x16_empty           # Transformer on any map, 500 pairs
sbatch sbatch/j2_mlp_map.sbatch 16x16_empty           # the MLP baseline for it
sbatch sbatch/d6_d4_repeat.sbatch                     # best-known config, corridors, ~4.5h
sbatch sbatch/eval_checkpoint.sbatch runs/<run-dir>   # recover a dead run's test numbers
```

**Reading results — use `compare_runs.py`, not grep.** Success rate alone saturates on the
easy maps and hides the mechanism on the hard ones; §9 lists the columns to report. A
`grep -m1` on an MLP log returns an *early-training* number (§3) — this has inverted a
conclusion once already.

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
absent).

```bash
MAPS="8x8_empty 16x16_empty 16x16_room_v1 16x16_obstacles_v1_15p 16x16_obstacles_v1_25p 24x24_empty"
for m in $MAPS; do
  sbatch sbatch/j1_gpt_map.sbatch "$m"
  sbatch sbatch/j2_mlp_map.sbatch "$m"
done
```

**Complete, 2026-11-09, at matched 1M budgets. Near-parity on six maps, one hard failure.**

| map | cov. | GPT succ | MLP succ | gap | GPT gen/ep | MLP gen/ep | GPT ratio | MLP ratio | GPT budget |
|---|---|---|---|---|---|---|---|---|---|
| 8x8_empty | 24.8% | 1.000 | 1.000 | 0.000 | **2.19** | 3.92 | **0.977** | 0.934 | 500k |
| 16x16_empty | 3.1% | 0.994 | 1.000 | −0.006 | **3.42** | 5.73 | **0.946** | 0.889 | 1M |
| 16x16_corridors_v1 | 28.1% | 0.990 | 0.982 | **+0.008** | **3.36** | 13.34 | **0.936** | 0.573 | 500k |
| 16x16_obstacles_v1_15p | 4.4% | 0.949 | 0.987 | −0.038 | **4.98** | 10.66 | 0.755 | **0.825** | 1M |
| 16x16_room_v1 | 5.8% | 0.945 *(.961/.929)* | 0.980 | −0.035 | **4.56** | 11.56 | **0.805** | 0.774 | 1M, n=2 |
| 16x16_obstacles_v1_25p | 6.7% | 0.915 *(.924/.905)* | 0.953 | −0.038 | **4.83** | 13.66 | **0.785** | 0.701 | 1M, n=2 |
| **24x24_empty** | 0.8% | **0.110** *(.120/.099)* | **0.973** | **−0.863** | 7.16 | 12.40 | 0.640 | **0.717** | 500k, n=2 |

**Six of seven maps are within 0.04 on success; one fails completely.** Against a 0.019–0.042
noise floor and a single-seed MLP, the −0.035/−0.038 cells are barely distinguishable from a
tie.

**And at matched budget the Transformer wins on both efficiency axes:** 2–3× fewer decisions
per episode on *every* map, and a better agent step ratio on **five of seven** (8x8, empty,
corridors, room, obstacles_25p). Only `obstacles_15p` and `24x24` favour the MLP on path
quality. An earlier version of this section, written from the 500k runs, said the step-ratio
advantage "inverts on every cluttered map" — that was truncation, not a property of the
decoder.

#### The 500k budget was most of the apparent gap

`--total_timesteps 500000` was validated on corridors, where validation peaks ~200k. It does
not transfer. The first sweep ran GPT at 500k against MLP at 1M, and three cells hit the
budget wall (best validation at 470,000–499,999, still climbing):

| map | GPT @500k | **GPT @1M** | recovered |
|---|---|---|---|
| 16x16_room_v1 | 0.838 | **0.945** | +0.107 |
| 16x16_obstacles_v1_15p | 0.824 | **0.949** | +0.125 |
| 16x16_obstacles_v1_25p | 0.505 | **0.915** | **+0.410** |

An earlier version of this section read those 500k numbers as "the Transformer loses on every
non-trivial map." That was a truncation artifact. **Always check `best val mean global step`
against the budget before interpreting a score** (rule 5).

#### 24x24_empty — the one genuine failure

0.120 and 0.099 across two seeds against the MLP's 0.973, and **not truncation**: the first
seed peaked at 295k and then declined, so a longer budget will not help. Ruled out —

- **not coverage** — the MLP reaches 0.973 at the same 0.8%
- **not horizon** — `max_level 14` caps every map at 14-step paths
- **not the length penalty** — the `weight 0` arm scores 0.141, equally bad

Something specific to the Transformer breaks on the 484-free-cell map while the MLP is
unaffected — it is the only cell where the two decoders are genuinely far apart.

**Diagnosed** by `check_z_influence.py` against a working checkpoint of the same config
(`unique-snow-60`, room_v1, 0.961) vs the failing one (`sunny-galaxy-68`, 0.099), both at
500 pairs:

| | room_v1 (0.961) | 24x24 (0.099) |
|---|---|---|
| Part A trained, unique trimmed (random `z`) | 60/256 | **64/256** |
| actor cosine | 0.7639 | 0.7746 |
| **actor `z`: std across dims** | **0.0271** | **0.0097** |
| plans from actor `z`, unique trimmed | 26/256 | 15/256 |

**The decoder is healthy on the failing map — marginally healthier than on the working one.**
64 distinct trimmed plans from random `z` vs room's 60. The decoder is not the cause.

**The actor's `z` is 2.8× narrower**, and the consequence is how much of the decoder's range
the actor actually reaches: room **26/60 = 43%**, 24x24 **15/64 = 23%**. Fifteen distinct
plans cannot cover 484 free cells; on corridors' 64 cells they can. That is the failure.

**Two readings, not yet separated** — `std` is not scale-invariant:

- **smaller-magnitude `z`** → decoder-side and cheap. `z_proj`'s init std (§6.4,
  `1/√16 ≈ 0.25`) is explicitly a tunable knob; a LayerNorm on `z` before projection would
  also make a small `z` still move the residual stream.
- **genuinely less diversity** → actor-side, and a poor road: §9.9 showed forcing actor
  diversity can cost 0.17.

`check_z_influence.py` now prints `mean L2 norm` and `std / norm` to tell them apart — a low
`std` with a proportionally low norm is magnitude, a low `std` at a normal norm is collapse.
**Re-run both checkpoints to read those two lines; it decides which fix applies.**

**Still open: is the narrowing caused by the map or by the Transformer?** The MLP's actor on
24x24 may be perfectly wide — its decoder maps `z` → all 50 logits in one linear layer and
passes a stronger gradient back (§4(b): actor grad norm 0.387 vs 0.197–0.265). Testing it
needs Part B against an MLP-tree checkpoint, which will not load here because the decoder
`state_dict` keys differ; it needs a Part-B-only mode. That decides whether the problem is
"large maps" or "the Transformer starves its own actor."

#### The length penalty is load-bearing

`j1 <map> 0` against the tuned config, both at 500k so the comparison is matched:

| map | plain (weight 0) | tuned (0.002) | penalty worth |
|---|---|---|---|
| 16x16_empty | 0.907 | 0.990 | +0.083 |
| 16x16_room_v1 | 0.226 | 0.838 | **+0.612** |
| 16x16_obstacles_v1_15p | 0.200 | 0.824 | **+0.624** |
| 16x16_obstacles_v1_25p | 0.244 | 0.505 | +0.261 |
| 24x24_empty | 0.141 | 0.120 | −0.021 (both failing) |

**Without it the agent collapses to 0.20–0.24 on every cluttered map.** So the penalty is not
the cause of the fixed-3-action behaviour to be removed — it is what makes the method work at
all. Any replacement (a learned stop token, state-dependent length) has to match it from the
start, not merely avoid its rigidity.

#### Plan length is rigid either way — the penalty moves a constant

`valid actions/seq` with and without the penalty:

| | empty | room | obs15 | obs25 | 24x24 |
|---|---|---|---|---|---|
| tuned (0.002) | 3.002 | 3.001 | 3.000 | 3.000 | 2.986 |
| **plain (0)** | **1.026** | **1.001** | **1.007** | **1.001** | **1.006** |
| MLP | 1.798 | 1.127 | 1.086 | 1.115 | 1.255 |

**The decoder has exactly two modes: 1.000 or 3.000, neither state-dependent.** The penalty
does not make plan length adaptive, it relocates the constant. That is the precise statement
of the problem an adaptive-replanning mechanism exists to solve.

**But note the MLP column.** It also runs short plans — 1.09 to 1.80 — and scores 0.95–1.00,
while the GPT at 1.00 scores 0.20–0.24. **Short plans are therefore not the problem in
themselves; the GPT's individual action choices are worse**, and the 3-action penalty
compensates for that rather than exploiting commitment. Do not design on the premise that
commitment is what makes GPS work — on this evidence it is a workaround for weaker per-step
action quality.

**Coverage is not the explanatory variable** for any of this — the MLP scores 0.973 on
24x24_empty at **0.8%**. Coverage explains only why corridors at 100 pairs looked starved
(§1.1).

#### The 100-pair axis — the Transformer is blind to map difficulty

The same four maps at the paper's 100-pair protocol, both arms at 1M steps:

| map | coverage | GPT | MLP | gap |
|---|---|---|---|---|
| 16x16_empty | 0.6% | 0.520 | **0.930** | −0.410 |
| 16x16_obstacles_v1_15p | 0.9% | 0.497 | **0.636** | −0.139 |
| 16x16_room_v1 | 1.2% | 0.515 | 0.507 | +0.008 |
| 16x16_obstacles_v1_25p | 1.3% | **0.487** | 0.399 | **+0.088** |

**2–2, and the GPT wins on the hardest map while losing badly on the easiest.** But the
scores are not the finding — the *variances* are:

| | range across the four maps |
|---|---|
| **GPT** | 0.487–0.520 = **0.033** (below the 0.042 noise floor) |
| MLP | 0.399–0.930 = **0.531** |

**The Transformer scores statistically the same on an empty grid and on 25% clutter.** It is
insensitive to map difficulty. The MLP tracks difficulty exactly as expected (0.930 empty →
0.636 light clutter → 0.507 rooms → 0.399 heavy clutter), which is what a decoder actually
solving the navigation problem looks like.

**Why the empty maps are the GPT's worst case — and it is not coverage.** Empty maps have
the most free cells, so a fixed pair count buys the *least* coverage there; yet the MLP does
**better** with less data when the map is open (0.930 at 0.6%) than with more data when it is
cluttered (0.636 at 0.9%). On an open grid the optimal plan is a smooth function of the
goal's relative offset, so 100 examples teach a rule that generalises. Clutter shatters that
rule into special cases.

The Transformer cannot exploit that regularity, because an open map needs a near-continuum of
plans and the actor supplies only **15–26 distinct ones across 256 states** (§8 diagnosis
above). Walls do the constraining for it; open space does not. **Its bottleneck — plan
diversity — bites hardest exactly where diversity matters most.** This is also why corridors
is its best map: 1-wide passages require the fewest distinct plans of any environment here.

`24x24_empty` is the compounded extreme: the largest open space (most diversity required)
*and* the smallest `z` magnitude (2.7× below a working map, so the least diversity
delivered). Hence 0.110 rather than the ~0.50 the other starved maps give.

**Falsifiable prediction:** if `--decoder_normalize_z` recovers 24x24 substantially,
`z`-resolution was the dominant term; if it recovers only partly, the plan-repertoire limit
is a separate and real constraint — and that is the one an adaptive-replanning mechanism
addresses.

**Caveat for reporting:** deterministic inference costs far more here than the historical
~0.10 — 0.399/0.067 and 0.507/0.113 on two of these runs, a 0.33–0.39 drop. The policy is
leaning on sampling noise, which is a signature of under-training rather than breakage, and
the paper's protocol number is the deterministic one when `deterministic_inference` is true.

#### The mechanism: the Transformer emits a rigid 3-action plan everywhere

| `valid actions/seq` | 8x8 | empty | room | obs15 | obs25 | 24x24 |
|---|---|---|---|---|---|---|
| **GPT** | 3.003 | 3.002 | **3.000** | **3.000** | **3.000** | **3.000** |
| MLP | 1.638 | 1.798 | 1.127 | 1.086 | 1.115 | 1.255 |

Exactly 3.000 across four structurally different maps is a degenerate policy, not adaptation
— and the length bands ask for 4–10. On corridors a blind multi-step commitment is nearly
free (1-wide passages, no branch points); on cluttered maps it is actively wrong. The cost
shows up in the derived metrics: the GPT's step ratio is **worse than the MLP's on every
hard map** (0.735/0.705/0.636/0.601 vs 0.774/0.825/0.701/0.717) and its global revisits are
higher (1.16/0.96/1.39/2.12 vs 0.75/0.46/1.29/1.20).

The §1.2 decision-count advantage does survive — about half the generations per episode on
every map — but here it buys worse paths and lower success, where on corridors it bought
both. **A fixed 3-action commitment is the whole story, and corridors is the one map where
it is free.**

Open: whether the rigidity is the length penalty's doing. The `j1 <map> 0` arms on
`16x16_empty`, `room_v1` and `obstacles_15p` (jobs 23971719–21) test exactly that.

### Coverage, and what matched coverage would cost

`--train_dataset_size 500` on every map is **not** a controlled comparison. Train-pool sizes
at levels 1–14, by BFS over each map's free cells:

| map | free cells | train pool | 500 = | @6% | @15% | @28% |
|---|---|---|---|---|---|---|
| 16x16_corridors_v1 | **64** | 1,778 | **28.1%** | 107 | 267 | 498 |
| 16x16_obstacles_v1_25p | 147 | 7,452 | 6.7% | 447 | 1,118 | 2,087 |
| 16x16_room_v1 | 160 | 8,626 | 5.8% | 518 | 1,294 | 2,415 |
| 16x16_obstacles_v1_15p | 167 | 11,454 | 4.4% | 687 | 1,718 | 3,207 |
| 16x16_empty | 196 | 16,380 | 3.1% | 983 | 2,457 | 4,586 |

Matching coverage at 28% means ~2,000–4,600 pairs on the open maps against corridors' 500.
A matched-coverage sweep is a *different, more expensive* experiment; this one is a
fixed-budget sweep and should be reported as such.

### The questions it answered

- **Does path optimality survive clutter?** **No.** The 0.936 step ratio was corridors-only;
  under clutter the MLP overtakes on every map (§8 table). This was the sweep's main job.
- **~~Does the advantage scale with horizon?~~** Not answerable here — `max_level 14` fixes
  the horizon on every map (caveat below). Map size varies the observation and the coverage,
  not the horizon.

### Caveat — the sweep cannot answer the horizon question

`max_level` defaults to 14, so **every map is capped at 14-step optimal paths**. Map size
changes the observation and how much of the space 500 pairs covers; it does **not** change
the horizon. An earlier version of this section claimed 24x24 was the long-horizon case and
told readers to watch for the `n_actions_in_seq=10` ceiling there — wrong. That ceiling
(two generations minimum for a 14-step path) binds equally on every map.

Coverage at levels 1–14, which is what actually varies:

By BFS over each map's actual free cells (the three "empty" maps are not all equally empty —
16x16 and 24x24 carry a wall border, 8x8 does not):

| map | free cells | reachable pairs | at level 1–14 | train pool | 500 pairs = |
|---|---|---|---|---|---|
| 8x8_empty | 64 | 4,032 | 4,032 (**100%**) | 2,016 | **24.8%** |
| 16x16_empty | 196 | 38,220 | 32,760 (86%) | 16,380 | 3.1% |
| 24x24_empty | 484 | 233,772 | 121,240 (52%) | 60,620 | 0.8% |

So **8x8 is saturated by construction** — its diameter *is* 14, so levels 1–14 is the whole
map and 500 pairs is a quarter of every pair that exists. Expect 1.0 from both decoders and
read only the sequence-level columns there.

(Coverage does **not** order the results — the MLP reaches 0.973 on 24x24_empty at 0.8%.
It explains only why 100 pairs looked starved on corridors, §1.1.)

A genuine horizon-scaling experiment needs `max_level` raised with the map (e.g. 14 / 30 /
46), which changes the pool size too and so needs the data regime re-fixed per map. That is
a different sweep, not a re-read of this one.

### Known asymmetries, to state when reporting

1. **500 pairs, not the paper's 100.** Deliberate (§1.1) — at 100 the comparison measures
   starvation. Any direct comparison to the paper's published numbers must use 100.
2. **The MLP tree is unpatched** — accumulating decoder gradient, see §3.
3. **1M steps for the MLP, 500k for the GPT** in the first sweep. Not an oversight: the
   MLP's best validation landed at step 999,999 on corridors (still climbing) while the
   Transformer peaks around 200k, so equal budgets would have handicapped the MLP. But it
   did not transfer — three GPT cells hit their budget wall (⚠ above) and are being re-run
   at 1M. Quote the 1M numbers once they land.
4. **n=1 per cell** in the first sweep. A 4× ratio is safe to quote; anything under ~1.5×
   is not, and the −0.010 on 16x16_empty is not a result. Second seeds are in flight on the
   four maps the conclusion rests on.
5. **The Transformer has a working length penalty; the MLP's is dead.** `torch.floor`
   (§3) silently disabled `--sequence_length_weight` in the paper's tree, so `j2` runs
   without it while `j1` runs at 0.002. The `j1 <map> 0` arms are the control.

---

## 9. Reporting — the metric this work is actually about

**For anyone writing this up: lead with replanning frequency, not success rate.**

Success rate answers "did it reach the goal", which both decoders do on the easy maps and
which the MLP simply wins on the hard ones. It is the *control* variable. The thesis
direction is **adaptive replanning with a Transformer**, and the quantity that measures
replanning is:

### `decoder generations per episode`

| where | exact name |
|---|---|
| slurm log | `test dataset with critic mean num decoder generations: <value>` |
| `compare_runs.py` | row `decoder generations/ep` |
| wandb | `test-eval-with-critic/mean_num_decoder_generations` (and the `-deterministic` variant) |
| source | `gps_simplegrid_levels.py:595` — incremented once per `get_reward_per_sequence` call |

**What it is.** The number of times the decoder was invoked to produce a fresh plan during
one episode — i.e. **how many times the agent replanned**. It is counted over successful
episodes only and averaged across the 1,000 test episodes.

**Why it is the right headline.** GPS's whole claim is that acting over sequences means
deciding less often. This is that claim as a number. A policy that replans at every step is
doing single-step control with extra machinery; one that replans 3 times per episode is
genuinely operating at the sequence level. Current baseline, 500 pairs:

| | corridors | 8x8 | empty | room | obs15 | obs25 | 24x24 |
|---|---|---|---|---|---|---|---|
| **GPT** (best budget) | **3.36** | **2.19** | **3.42** | **4.56** | **4.98** | **4.83** | **7.16** |
| MLP | 13.34 | 3.92 | 5.73 | 11.56 | 10.66 | 13.66 | 12.40 |

The Transformer replans **2–3× less often on every map**, and 4× less on corridors. This is
the one result that holds across the entire sweep without qualification.

**How it must be qualified — never quote it alone.** The metric is trivially gamed: emit one
enormous plan, replan once, fail the episode. Three companions make it honest, and all three
come out of the same `compare_runs.py` table:

| companion | why |
|---|---|
| `success` | did it actually reach the goal? Fewer decisions at lower success is not a win. |
| `agent step ratio` | optimal steps ÷ actual steps. Fewer decisions that produce *longer wandering paths* is not a win either. At matched 1M budgets the Transformer is **ahead on five of seven maps** here; it trails on `obstacles_15p` (0.755 vs 0.825) and `24x24_empty` (0.640 vs 0.717). |
| `valid actions/seq` | the emitted plan length. Pinned at **3.000** with the length penalty and **1.000** without it, on every map — a constant either way, never state-dependent. **This is the defect the thesis addresses.** |

**The honest summary of the baseline**, as of 2026-11-09 (matched 1M budgets):

> At matched training budgets the Transformer decoder reaches the goal about as often as the
> MLP on six of seven environments — within 0.04, and ahead on `16x16_corridors_v1` — while
> replanning **2–3× less often on every map** and following more direct paths on five of the
> seven. It fails completely on the largest map, `24x24_empty` (0.11 vs 0.97 across two
> seeds), for reasons not yet explained.

The motivation for adaptive replanning is not in that summary but in the mechanism behind it:
**plan length is a constant, not a decision.** The decoder emits 3.000 actions with the
length penalty and 1.000 without it, on every map (§8) — it never adapts to the state. The
commitment is cheap and works; *choosing when to commit* does not exist yet.

Two caveats a write-up must carry: the length penalty producing the 3-action plan is
**load-bearing** (at `weight 0` success collapses to 0.20–0.24), and the MLP achieves 0.95–1.00
with plans of 1.09–1.80, so **short plans are not the deficiency** — the Transformer's
per-step action quality is.

### Also worth a table in any write-up

| metric | log name | what it adds |
|---|---|---|
| `agent step ratio` | `mean agent step ratio in sequences per episode` | path optimality, 0–1 |
| `valid actions/seq` | `mean num valid actions in sequences per episode` | plan length (despite the name — it is `len(actions)`, `gps_simplegrid_levels.py:609`) |
| `unused actions/seq` | `mean num unused actions in sequences per episode` | plan waste: emitted minus executed |
| `revisits (global)` | `mean global visited position count in sequences per episode` | wandering |
| `wall-clock / optimal step` | `average wall-clock time per optimal step` | compute per unit of real progress |
| `inference / decision` | `average inference time per decision` | per-call cost; near-identical between decoders, which is what makes the decision *count* the interesting axis |

Generate all of them with:

```bash
python compare_runs.py --csv all_maps.csv sbatch/logs/j1_*.out sbatch/logs/j2_*.out
```
