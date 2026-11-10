# Results — Transformer decoder vs MLP decoder in GPS

Final baseline comparison. Numbers only; reasoning and diagnosis are in
[`EVIDENCE.md`](EVIDENCE.md) §8, full history in
[`TRANSFORMER_INTEGRATION.md`](generative/models/dense_autoencoder/TRANSFORMER_INTEGRATION.md) §10.

Task: from-scratch end-to-end RL, `--train_decoder_end_to_end
--no-initialize_decoder_from_pretrained`. Test set 1,000 start/goal pairs per map.
GPT = minGPT causal decoder (this work). MLP = the paper's decoder, run unmodified from the
`6f627a2` code drop.

---

## Table 1 — main comparison, 500 training pairs

GPT at its best budget (1M where it was still improving at 500k); MLP at 1M throughout.

| map | success |  | decoder generations / episode |  |  | agent step ratio |  | plan length |  |
|---|---|---|---|---|---|---|---|---|---|
| | **GPT** | MLP | **GPT** | MLP | **reduction** | **GPT** | MLP | **GPT** | MLP |
| 8x8_empty | 1.000 | 1.000 | **2.19** | 3.92 | **1.8×** | **0.977** | 0.934 | 3.00 | 1.64 |
| 16x16_empty | 0.994 | 1.000 | **3.42** | 5.73 | **1.7×** | **0.946** | 0.889 | 3.00 | 1.80 |
| 16x16_corridors_v1 | **0.990** | 0.982 | **3.36** | 13.34 | **4.0×** | **0.936** | 0.573 | 3.01 | 2.06 |
| 16x16_obstacles_v1_15p | 0.949 | 0.987 | **4.98** | 10.66 | **2.1×** | 0.755 | **0.825** | 3.00 | 1.09 |
| 16x16_room_v1 | 0.945 | 0.980 | **4.56** | 11.56 | **2.5×** | **0.805** | 0.774 | 3.00 | 1.13 |
| 16x16_obstacles_v1_25p | 0.915 | 0.953 | **4.83** | 13.66 | **2.8×** | **0.785** | 0.701 | 3.00 | 1.12 |
| **24x24_empty** | **0.110** | **0.973** | 7.16 | 12.40 | 1.7× | 0.640 | **0.717** | 2.99 | 1.26 |

**Read:** success within 0.04 on six of seven maps; **2–3× fewer decisions per episode on
every map**; better path optimality on five of seven. One complete failure, `24x24_empty`.

`n=2` for `16x16_room_v1` (0.961, 0.929), `16x16_obstacles_v1_25p` (0.924, 0.905) and
`24x24_empty` (0.120, 0.099); `n=1` elsewhere. MLP is `n=1` throughout.

---

## Table 2 — data regime, 100 training pairs

Both arms 1M steps. The paper's protocol pair count.

| map | coverage | GPT | MLP | gap |
|---|---|---|---|---|
| 16x16_empty | 0.6% | 0.520 | **0.930** | −0.410 |
| 16x16_obstacles_v1_15p | 0.9% | 0.497 | **0.636** | −0.139 |
| 16x16_room_v1 | 1.2% | 0.515 | 0.507 | +0.008 |
| 16x16_obstacles_v1_25p | 1.3% | **0.487** | 0.399 | **+0.088** |
| **range** | | **0.033** | **0.531** | |

**Read:** 2–2 on wins, but the *variance* is the finding. The GPT's spread across four maps
(0.033) is below the 0.042 noise floor — it is insensitive to map difficulty. The MLP tracks
difficulty as a navigating policy should.

"Coverage" is the share of each map's level-1–14 train pool that the pair count buys; it
differs per map because the maps differ in free space. See Table 5.

---

## Table 3 — sample efficiency, `16x16_corridors_v1`

| training pairs | coverage | GPT | MLP |
|---|---|---|---|
| 100 | 5.6% | 0.725 (n=5) | 0.794 |
| 300 | 16.9% | 0.897 | **0.977** |
| 500 | 28.1% | **0.990** | 0.982 |

**Read:** the MLP saturates by 300 pairs; the GPT needs 500 to reach the same level. The
0.069 gap at 100 pairs closes entirely by 500.

---

## Table 4 — length-penalty ablation, 500 pairs, 500k steps

`--sequence_length_weight 0.002` vs `0`, all else identical.

| map | plain (0) | tuned (0.002) | penalty worth | plan length, plain → tuned |
|---|---|---|---|---|
| 16x16_empty | 0.907 | 0.990 | +0.083 | 1.03 → 3.00 |
| 16x16_room_v1 | 0.226 | 0.838 | **+0.612** | 1.00 → 3.00 |
| 16x16_obstacles_v1_15p | 0.200 | 0.824 | **+0.624** | 1.01 → 3.00 |
| 16x16_obstacles_v1_25p | 0.244 | 0.505 | +0.261 | 1.00 → 3.00 |
| 24x24_empty | 0.141 | 0.120 | −0.021 | 1.01 → 3.00 |

**Read:** the penalty is load-bearing, not a source of rigidity. Plan length is a **constant
either way** — 1.000 without it, 3.000 with it, on every map — never state-dependent.

---

## Table 5 — dataset coverage by map

Train-pool size at levels 1–14, by BFS over each map's free cells. `--train_dataset_size` is
an absolute count, so the same flag buys very different coverage per map.

| map | free cells | train pool | 100 pairs | 500 pairs |
|---|---|---|---|---|
| 16x16_corridors_v1 | 64 | 1,778 | 5.6% | 28.1% |
| 8x8_empty | 64 | 2,016 | 5.0% | 24.8% |
| 16x16_obstacles_v1_25p | 147 | 7,452 | 1.3% | 6.7% |
| 16x16_room_v1 | 160 | 8,626 | 1.2% | 5.8% |
| 16x16_obstacles_v1_15p | 167 | 11,454 | 0.9% | 4.4% |
| 16x16_empty | 196 | 16,380 | 0.6% | 3.1% |
| 24x24_empty | 484 | 60,620 | 0.16% | 0.8% |

---

## Table 6 — `24x24_empty` diagnosis

`check_z_influence.py`, 500 pairs, against a working checkpoint of the same configuration.

| | `16x16_room_v1` (0.961) | `24x24_empty` (0.099) |
|---|---|---|
| decoder, unique plans from random `z` | 60/256 | **64/256** |
| actor `z`, mean pairwise cosine | 0.7639 | 0.7746 |
| actor `z`, std across dims | 0.0271 | **0.0097** (2.8× smaller) |
| actor `z`, mean L2 norm | 0.2079 | **0.0782** (2.7× smaller) |
| actor `z`, **std / norm** | **0.1306** | **0.1246** (same) |
| plans reached from actor `z` | 26/256 (43% of decoder range) | 15/256 (23%) |

**Read:** the decoder is healthy on the failing map — marginally healthier than on the
working one. The actor has **not** collapsed: its `z` carries the same relative variety
(`std/norm` matches within 5%), it is simply 2.7× smaller in magnitude. `z` enters the
Transformer as a prefix token competing against fixed-scale token embeddings in an
**unnormalised** residual stream, so a small `z` is drowned out. The MLP decoder has no
residual stream and absorbs a uniform rescale in its weights.

---

## Caveats to carry into any write-up

1. **500 pairs is not the paper's protocol** (100). Table 2 gives the 100-pair numbers; no
   direct comparison to published figures without them.
2. **`--train_dataset_size` is not comparable across maps** — see Table 5. Within a map both
   arms get identical data, so per-map comparisons are clean; cross-map ones are not.
3. **`max_level` is 14 on every map**, so no environment here tests long horizons.
   `24x24_empty` is short paths on a large canvas.
4. **The MLP tree is unpatched** — it predates fixes to `decoder_optimizer.zero_grad()`,
   inference-time dropout, and the `torch.floor` bug that disabled
   `--sequence_length_weight`. Deliberate: it is the published method as published. The
   consequence is that the GPT has a working length penalty and the MLP does not.
5. **Single seed** on most cells and on the MLP throughout. Noise floor is 0.019–0.042 (1σ),
   so the −0.035/−0.038 gaps in Table 1 are not distinguishable from ties.
6. **Deterministic inference** costs ~0.10 normally but 0.33–0.39 at 100 pairs, where the
   policy leans on sampling noise. Report which pass you are quoting.
7. **Training budgets differ by cell in Table 1.** The MLP gets 1M steps everywhere. The GPT
   gets 1M on `16x16_empty`, `room_v1` and both obstacle maps — the cells where a 500k run
   was still improving at its budget wall (best validation at 470k–499,999) — and 500k on
   `8x8_empty`, `corridors` and `24x24_empty`, where it had already converged (best
   validation at 425k, ~200k and 230k–295k respectively, followed by decline). Giving those
   three 1M would not change them. The 500k figures for the other four were materially
   lower (e.g. obstacles_25p 0.505 → 0.915) and should not be quoted.

---

## In flight — not a result yet

A candidate fix for `24x24_empty` is running on branch `normalize-test`
(`--decoder_normalize_z`, see `EVIDENCE.md` §10). It pins `‖z‖` to a learnable constant
before the decoder projects it, targeting the magnitude deficit in Table 6. Nothing in the
tables above depends on it, and it must not be reported as a result until its runs land and
the three regression maps are confirmed unchanged.
