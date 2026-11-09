"""Diagnose where sequence diversity is being lost: the decoder or the actor.

``sequences/num_unique`` collapses monotonically in the Transformer runs while the MLP
baseline sustains ~40. That metric is measured end-to-end -- actor observations -> z ->
decoded sequence -- so it cannot say which half collapsed, and the two have opposite fixes:

* a **decoder** that ignores z needs stronger conditioning (z at every position, gradients
  through the token chain);
* an **actor** emitting near-identical embeddings needs the diversity regulariser, which
  acts on the actor's per-head embeddings, not on the decoder at all.

This measures both, offline, from saved checkpoints -- no training.

    poetry run python check_z_influence.py \\
        --run_dir runs/SimpleGrid-v0__gps_simplegrid_levels__earthy-bird-16 \\
        --cuda --obstacle_map 16x16_corridors_v1 \\
        --train_dataset_size 100 --val_dataset_size 100 --test_dataset_size 1000 \\
        --train_decoder_end_to_end --no-initialize_decoder_from_pretrained

Flags that affect network shape or the dataset split must match the training run.

Reference points: §6.4 of TRANSFORMER_INTEGRATION.md measured 29 unique greedy sequences
over 64 random z at initialisation (12 before the z_proj init fix). The MLP baseline run
sustained sequences/num_unique ~40.8 at step 1M.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from gps_simplegrid_levels import (
    Config,
    dataset_setup,
    env_setup,
    get_device,
    set_seed,
)
from generative.models.actor_cnn import Actor
from generative.models.dense_autoencoder.decoder_utils import get_decoder_api
from generative.models.dense_autoencoder.dense_var_auto_encoder_vin_1 import DenseVAE


@dataclass
class DiagConfig(Config):
    run_dir: str = ""
    """directory holding the saved 'actor' and 'decoder' state dicts"""
    n_samples: int = 256
    """how many random z / observations to probe with"""


def unique_sequences(tokens: torch.Tensor) -> tuple[int, int]:
    """Count distinct sequences, raw and EOS-trimmed.

    Trimmed is the meaningful one: it is what the agent actually executes, so two
    sequences differing only after their EOS are the same proto-plan in practice.
    """
    raw = torch.unique(tokens.cpu(), dim=0).shape[0]
    trimmed = {
        tuple(DenseVAE.trim_action_sequence_from_eos_tokens(seq.clone())[0].tolist())
        for seq in tokens.cpu()
    }
    return raw, len(trimmed)


@torch.no_grad()
def greedy_decode(decoder, z: torch.Tensor) -> torch.Tensor:
    """Greedy (argmax) token sequences for each row of z -- §6.4's measure."""
    return decoder.model.decoder.generate(z, do_sample=False)


def describe(label: str, tokens: torch.Tensor, n: int) -> None:
    raw, trimmed = unique_sequences(tokens)
    lengths = [
        len(DenseVAE.trim_action_sequence_from_eos_tokens(seq.clone())[0])
        for seq in tokens.cpu()
    ]
    print(f"{label:<42} unique(raw)={raw:>4}/{n}  unique(trimmed)={trimmed:>4}/{n}  "
          f"mean_len={np.mean(lengths):.2f}")


def main() -> None:
    cfg = tyro.cli(DiagConfig)

    run_dir = Path(cfg.run_dir)
    if not cfg.run_dir or not run_dir.is_dir():
        raise SystemExit(f"--run_dir must name an existing directory (got {cfg.run_dir!r})")
    actor_path, decoder_path = run_dir / "actor", run_dir / "decoder"
    for required in (actor_path, decoder_path):
        if not required.is_file():
            raise SystemExit(f"missing checkpoint {required}")

    device = get_device(cfg)
    set_seed(seed=cfg.seed, deterministic_torch=cfg.torch_deterministic)
    n = cfg.n_samples

    run_name = run_dir.name
    envs = env_setup(cfg, run_name)
    maze_size = envs.single_observation_space.sample()[None].shape[2]
    n_input_channels = envs.single_observation_space.shape[0]
    observation_shape = (maze_size, maze_size)

    def build_decoder():
        return get_decoder_api(
            decoder_model_path=cfg.decoder_model_path,
            decoder_seq_len=cfg.n_actions_in_seq, device=device, maze_n_actions=4,
            use_gumble_in_decoder=cfg.use_gumble_in_decoder,
            penalize_cyclic_position_revisits=cfg.penalize_cyclic_position_revisits,
            deterministic_inference=cfg.deterministic_inference,
            load_pretrained_weights=False,
            decoder_n_layer=cfg.decoder_n_layer, decoder_n_head=cfg.decoder_n_head,
            decoder_n_embd=cfg.decoder_n_embd, decoder_dropout=cfg.decoder_dropout,
        )

    print("=" * 78)
    print("PART A -- does the DECODER respond to z?  (random z, greedy decode)")
    print("=" * 78)

    z_random = torch.randn(n, cfg.actor_n_output_channels, device=device)

    # Freshly initialised decoder: the control. Per §6.4 this should be healthy.
    # .to(device) matters: z_random is built on `device` above, and the trained decoder gets
    # moved on load below, but this one never was -- so under --cuda the control sat on the
    # CPU and greedy_decode raised a device mismatch. Only CPU runs worked.
    untrained = build_decoder()
    untrained.model.to(device).eval()
    describe("untrained decoder (control)", greedy_decode(untrained, z_random), n)

    trained = build_decoder()
    trained.model.load_state_dict(torch.load(decoder_path, map_location=device))
    trained.model.to(device).eval()
    describe("trained decoder", greedy_decode(trained, z_random), n)

    print()
    print("=" * 78)
    print("PART B -- does the ACTOR produce varied z?  (real observations)")
    print("=" * 78)

    _, val_dataset, _, _ = dataset_setup(
        cfg, run_name, max_level=cfg.max_level, start_level=cfg.start_level
    )
    envs.envs[0].unwrapped.start_goal_dataset = val_dataset

    observations = []
    obs, _ = envs.reset(seed=cfg.seed)
    observations.append(obs)
    for _ in range(n - 1):
        obs, _ = envs.reset()
        observations.append(obs)
    obs_tensor = torch.tensor(np.concatenate(observations), dtype=torch.float32, device=device)

    actor = Actor(
        n_input_channels=n_input_channels,
        n_output_channels=cfg.actor_n_output_channels,
        observation_shape=observation_shape,
        noise=None,
        linear_layers=cfg.actor_linear_layers,
        linear_layers_activation=cfg.actor_linear_layers_activation_function,
        use_batchnorm_linear_layers=cfg.actor_use_batchnorm_linear_layers,
        num_heads=cfg.n_proto_plan_candidates,
        pe_embedding_dim=cfg.pe_embedding_dim,
    ).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    with torch.no_grad():
        z_actor = actor(obs_tensor).reshape(-1, cfg.actor_n_output_channels)

    # Mean pairwise cosine similarity, excluding self-pairs: 1.0 means the actor emits the
    # same embedding for every state, which would make the decoder's behaviour irrelevant.
    normalized = F.normalize(z_actor, p=2, dim=1)
    sims = normalized @ normalized.T
    off_diagonal = sims[~torch.eye(sims.shape[0], dtype=torch.bool, device=device)]

    # std discriminates where cosine does not: a working 16x16_room_v1 checkpoint (0.961) and
    # a failing 24x24_empty one (0.099) differ by 0.011 in cosine and by 2.8x in std. But std
    # is NOT scale-invariant, so a small value is ambiguous between "less diversity" and
    # "smaller z overall" -- and the two have different fixes (actor-side vs scaling z_proj).
    # The norm and the ratio below separate them: a low std with a proportionally low norm is
    # a magnitude effect; a low std at a normal norm is genuine collapse.
    std = z_actor.std(dim=0).mean().item()
    norm = z_actor.norm(p=2, dim=1).mean().item()

    print(f"actor z: mean pairwise cosine similarity = {off_diagonal.mean().item():.4f} "
          f"(1.0 = all identical)")
    print(f"actor z: std across embedding dims       = {std:.4f}")
    print(f"actor z: mean L2 norm                    = {norm:.4f}")
    print(f"actor z: std / norm (scale-free spread)  = {std / norm if norm else float('nan'):.5f}")
    describe("sequences from actor z (trained decoder)",
             greedy_decode(trained, z_actor), z_actor.shape[0])

    envs.close()

    print()
    print("=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("Part A trained << untrained    -> the DECODER collapsed; it ignores z.")
    print("                                  Fix: condition on z at every position, and/or")
    print("                                  soft token feedback so gradients reach z.")
    print("Part A fine, Part B similarity -> the ACTOR collapsed; it emits one embedding.")
    print("  near 1.0 / few sequences        Fix: --diversity_weight (it acts on the")
    print("                                  actor's head embeddings), more exploration.")
    print("  BUT judge this on std/norm,     cosine barely moved between a 0.961 and a 0.099")
    print("  not on cosine:                  checkpoint (0.764 vs 0.775) while std differed")
    print("                                  2.8x (0.0271 vs 0.0097).")
    print("    low std + low norm         -> magnitude, not diversity. Scale z_proj (sec 6.4)")
    print("                                  or LayerNorm z; actor-side fixes miss the point.")
    print("    low std + normal norm      -> genuine collapse; actor-side.")
    print("Both fine                      -> diversity is lost elsewhere; look at critic")
    print("                                  selection among candidates.")


if __name__ == "__main__":
    main()
