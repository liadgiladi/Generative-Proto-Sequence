#!/bin/bash
# I3 -- MLP decoder (paper code, unmodified) on the obstacle-free 16x16 map.
#
# The baseline for I1/I2. Runs against the git worktree at 6f627a2, which is literally the
# paper's code drop ("GPS - Generative Proto-Sequence: Sequence-Level Decision Making for
# Long-Horizon Reinforcement Learning") and is identical to origin/main. Nothing in it is
# patched, including the torch.floor bug -- that is the point: this is the published method
# as published, which is what I2 is meant to be compared against.
#
# Create the worktree once, from the repo root:
#   git worktree add ~/gps-mlp-baseline 47d6da4^
#
# Safe to run concurrently with h1_mlp_500pairs.sbatch: both only read the source, and all
# output paths are keyed by the unique wandb run name (runs/<run-name>/, wandb/run-<id>/).
#
# Flags match the paper's defaults for this task, with only the map changed. Note the
# absence of --decoder_use_cuda_graph, --min_qf_value and the decoder_* options: none exist
# in that tree. Determinism is on (the Config default), as in the published runs.
#
# 1M steps, not 500k. On corridors the MLP's best validation landed at step 999,999 -- it
# was still improving when its run ended, unlike the Transformer which peaks around 200k.
# Giving it less than the full budget would handicap it. Expect ~10-11h.
#
#SBATCH --partition=main
#SBATCH --time=1-00:00:00
#SBATCH --job-name=i3_empty_mlp
#SBATCH --output=sbatch/logs/i3_empty_mlp-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

export CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKTREE="${HOME}/gps-mlp-baseline"

if [ ! -d "${WORKTREE}" ]; then
    echo "ERROR: ${WORKTREE} does not exist. Create it first, from the repo root:"
    echo "  git worktree add ${WORKTREE} 47d6da4^"
    exit 1
fi

# The worktree must hold the MLP decoder. If this finds mingpt it is at the wrong commit and
# the run would silently repeat the Transformer experiment under an MLP job name.
if grep -q "mingpt_decoder" "${WORKTREE}/generative/models/dense_autoencoder/dense_var_auto_encoder_vin_1.py"; then
    echo "ERROR: ${WORKTREE} contains the Transformer decoder, not the MLP."
    echo "       Check it out at 47d6da4^ (= 6f627a2, the paper's code drop)."
    exit 1
fi

cd "${WORKTREE}" || exit 1
echo "running from $(pwd) at commit $(git rev-parse --short HEAD)"

START=$(date +%s)

poetry run python gps_simplegrid_levels.py \
    --cuda \
    --track \
    --obstacle_map 16x16_empty \
    --train_dataset_size 100 \
    --train_eval_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --train_decoder_end_to_end \
    --no-initialize_decoder_from_pretrained

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
