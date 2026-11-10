"""Run the test-set evaluation against checkpoints saved by a finished training run.

Training writes its best-on-validation actor, critic and (under --train_decoder_end_to_end)
decoder to ``runs/<run_name>/`` whenever the validation success rate improves. The final
test evaluation, however, only happens if ``train()`` reaches its end -- so a run that
crashed or aborted partway through leaves a perfectly good model on disk with no test
numbers attached to it.

This recovers that: same networks, same dataset split, same ``eval_test_dataset`` the
training loop would have called.

Pass the training run's flags so the architectures and the test split match, plus the run
directory:

    poetry run python eval_checkpoint.py \\
        --run_dir runs/SimpleGrid-v0__gps_simplegrid_levels__earthy-bird-16 \\
        --cuda \\
        --obstacle_map 16x16_corridors_v1 \\
        --train_dataset_size 100 \\
        --val_dataset_size 100 \\
        --test_dataset_size 1000 \\
        --train_decoder_end_to_end \\
        --no-initialize_decoder_from_pretrained \\
        --decoder_use_cuda_graph

Anything that changes network shape or the dataset split -- obstacle_map, the dataset
sizes, seed, max_level, start_level, the decoder_* options, the actor/critic layer sizes --
must match the training run, or the state dicts will not load and the "test set" evaluated
here will not be the one the run was held out against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

import gps_simplegrid_levels as gps
from gps_simplegrid_levels import (
    Config,
    dataset_setup,
    env_setup,
    eval_test_dataset,
    get_device,
    set_seed,
)
from generative.models.dense_autoencoder.decoder_utils import get_decoder_api


@dataclass
class EvalConfig(Config):
    """Training config plus the directory holding the checkpoints to evaluate."""

    run_dir: str = ""
    """directory holding the saved 'actor', 'critic' and (E2E) 'decoder' state dicts --
       i.e. runs/<run_name> from the training job"""


def main() -> None:
    cfg = tyro.cli(EvalConfig)

    run_dir = Path(cfg.run_dir)
    if not cfg.run_dir or not run_dir.is_dir():
        raise SystemExit(f"--run_dir must name an existing directory (got {cfg.run_dir!r})")

    actor_path, critic_path = run_dir / "actor", run_dir / "critic"
    decoder_path = run_dir / "decoder"

    for required in (actor_path, critic_path):
        if not required.is_file():
            raise SystemExit(
                f"missing {required}. Checkpoints are only written once the validation "
                f"success rate improves, so a run that died before its first validation "
                f"evaluation has nothing to recover."
            )

    device = get_device(cfg)
    print(f"using device: {device}")

    # eval_test_dataset reads ``run_name`` from the module globals -- it is set in
    # gps_simplegrid_levels' __main__ block, which does not run on import. Only used to
    # name tensorboard tags and the video directory.
    run_name = run_dir.name
    gps.run_name = run_name

    # Match the training run's seeding so dataset_setup reproduces the same split;
    # otherwise the "test set" measured here is not the one the model was held out against.
    set_seed(seed=cfg.seed, deterministic_torch=cfg.torch_deterministic)

    envs = env_setup(cfg, run_name)
    maze_size = envs.single_observation_space.sample()[None].shape[2]
    n_input_channels = envs.single_observation_space.shape[0]
    observation_shape = (maze_size, maze_size)
    envs.close()

    _, _, test_dataset, _ = dataset_setup(
        cfg, run_name, max_level=cfg.max_level, start_level=cfg.start_level
    )

    # Build the decoder architecture empty. ``load_pretrained_weights`` refers to the
    # original pretrained VAE checkpoint, which is the wrong architecture in E2E mode --
    # the trained weights come from this run's own decoder checkpoint below.
    decoder = get_decoder_api(
        decoder_model_path=cfg.decoder_model_path,
        decoder_seq_len=cfg.n_actions_in_seq,
        device=device,
        maze_n_actions=4,
        use_gumble_in_decoder=cfg.use_gumble_in_decoder,
        penalize_cyclic_position_revisits=cfg.penalize_cyclic_position_revisits,
        deterministic_inference=cfg.deterministic_inference,
        load_pretrained_weights=False,
        decoder_n_layer=cfg.decoder_n_layer,
        decoder_n_head=cfg.decoder_n_head,
        decoder_n_embd=cfg.decoder_n_embd,
        decoder_dropout=cfg.decoder_dropout,
        decoder_use_cuda_graph=cfg.decoder_use_cuda_graph,
    )

    if decoder_path.is_file():
        decoder.model.load_state_dict(torch.load(decoder_path, map_location=device))
        print(f"loaded decoder checkpoint from {decoder_path}")
    elif cfg.train_decoder_end_to_end:
        raise SystemExit(
            f"missing {decoder_path} but --train_decoder_end_to_end was passed. Evaluating "
            f"a randomly initialised decoder would be meaningless -- check the run directory."
        )
    else:
        decoder.model.load_state_dict(torch.load(cfg.decoder_model_path, map_location=device))
        print(f"loaded frozen pretrained decoder from {cfg.decoder_model_path}")

    decoder.model.to(device).eval()

    writer = SummaryWriter(f"runs/{run_name}-recovered-eval")

    # The same call the end of train() makes: two passes over the test set (deterministic
    # inference and stochastic), printing mean reward, success and sequence statistics.
    eval_test_dataset(
        test_dataset,
        cfg,
        writer,
        n_input_channels,
        n_input_channels,
        observation_shape,
        decoder,
        device,
        actor_model_path=str(actor_path),
        critic_model_path=str(critic_path),
    )

    writer.close()


if __name__ == "__main__":
    main()
