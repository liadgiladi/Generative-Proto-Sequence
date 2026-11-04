#!/bin/bash
# J2 -- MLP decoder (paper code, unmodified) on any map.  The baseline for J1.
#
#   sbatch sbatch/j2_mlp_map.sbatch <obstacle_map>
#   sbatch sbatch/j2_mlp_map.sbatch 16x16_obstacles_v1_25p
#
# Runs against the git worktree at 6f627a2 -- the paper's code drop, identical to
# origin/main. Nothing patched, including the torch.floor bug that disables
# --sequence_length_weight. That is deliberate: this is the published method as published,
# which is what J1-with-weight-0 is meant to be compared against.
#
# Create the worktree once, from the repo root:
#   git worktree add ~/gps-mlp-baseline 47d6da4^
#
# Safe to run several of these concurrently, and alongside the Transformer runs: they only
# read the source, and every output path is keyed by the unique wandb run name.
#
# 1M steps, not 500k. On corridors the MLP's best validation landed at step 999,999 -- still
# improving when its run ended, unlike the Transformer which peaks around 200k. Giving it
# less than the full budget would handicap it. Expect ~10-11h per map.
#
# ASYMMETRY TO STATE WHEN REPORTING: this tree also predates the sec. 7-9 fixes, so it runs
# with an accumulating decoder gradient (decoder_optimizer.zero_grad() missing) and
# inference-time dropout. Both were measured to buy ~0 on their own, and the MLP has no
# dropout layers at all so the second does not apply to it -- but the first does.
#
#SBATCH --partition=main
#SBATCH --time=1-00:00:00
#SBATCH --job-name=j2_mlp_map
#SBATCH --output=sbatch/logs/j2_mlp_map-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

export CUBLAS_WORKSPACE_CONFIG=:4096:8

MAP="$1"
TRAIN_PAIRS="${2:-500}"
WORKTREE="${HOME}/gps-mlp-baseline"

if [ -z "${MAP}" ]; then
    echo "ERROR: pass the obstacle map, e.g."
    echo "  sbatch $0 16x16_obstacles_v1_25p"
    exit 1
fi

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
echo "=== J2: map=${MAP}  train_pairs=${TRAIN_PAIRS}  from $(pwd) at $(git rev-parse --short HEAD) ==="

START=$(date +%s)

poetry run python gps_simplegrid_levels.py \
    --cuda \
    --track \
    --obstacle_map "${MAP}" \
    --train_dataset_size "${TRAIN_PAIRS}" \
    --train_eval_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --train_decoder_end_to_end \
    --no-initialize_decoder_from_pretrained

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
