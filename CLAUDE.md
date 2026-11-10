# Generative Proto-Sequence (GPS)

RL agent that proposes multi-action "proto-plans": an actor emits a latent `z`, a decoder
turns `z` into an action sequence, a critic scores it. Active work on branch
`transformer-integration` replaces the VAE's single-shot MLP decoder with a small causal
Transformer (minGPT-style).

## Before proposing or running experiments

**Read [`EVIDENCE.md`](EVIDENCE.md) first.** It holds current results, the methodology that
earned itself through failed runs, and a table of metrics and flags that silently lie or do
nothing. Detailed history is in
[`generative/models/dense_autoencoder/TRANSFORMER_INTEGRATION.md`](generative/models/dense_autoencoder/TRANSFORMER_INTEGRATION.md)
§7–§9.

A previous agent proposed a well-argued experiment plan that was wrong because it worked
from a stale snapshot and did not know one of the metrics was computed over the wrong axis.
Check `EVIDENCE.md` §1 (current results) and §3 (traps) before trusting any number you read
in a log or in wandb.

## Non-negotiables

- **Noise floor is ~0.05 test success.** Two runs of the same config gave 0.432 and 0.488.
  A difference smaller than that is not a result. Prefer two seeds per config;
  `--no-torch_deterministic` makes a re-run an independent sample.
- **One variable per run**, built on the current best config, not on a superseded baseline.
- **Pin the GPU type** in every sbatch: `--gpus=gtx_1080:1`. A bare `--gpus=1` silently
  hands out `rtx_2080`/`rtx_3090` and breaks comparability.
- **Never commit a wandb API key.** One was auto-revoked for being committed here.
  Credentials belong in `~/.netrc`.
- Keep new behaviour behind a default-off flag so it is A/B-able.

## Experiments live in `sbatch/`

`e*` runtime (settled), `a*`/`b*`/`d*` accuracy (active). Each file documents its own
hypothesis, the evidence motivating it, and how to read its result — follow that style.
Every run writes `WALL_CLOCK_HOURS` and its config to wandb.
