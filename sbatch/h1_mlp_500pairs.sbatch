#!/bin/bash
# H1 -- the MLP decoder with 500 training pairs. The decisive comparison.
#
# WHY
#
# Training-set size turned out to be the binding constraint for the Transformer:
#
#   100 pairs   0.725 +/- 0.016  (6 runs)
#   300 pairs   0.897            (g1)
#   500 pairs   0.99             (g2, best val 1.0)
#
# But the MLP baseline's 0.794 was measured on 100 pairs (slurm 18136376). So 0.99 vs 0.794
# is not a comparison -- the MLP was never given that data and would presumably also improve.
#
# Everything diagnosed over the past two weeks -- actor collapse, plan length, the 200k
# stall, the 0.17 train/val gap -- may simply have been data starvation wearing different
# masks. This run is what distinguishes the two readings:
#
#   MLP also ~0.99  -> data was the whole story; the architectures are equivalent, and the
#                      0.794-vs-0.725 gap was an artifact of training on 100 pairs
#   MLP stays lower -> the Transformer genuinely wins once it has enough data
#
# SETUP
#
# The MLP decoder was replaced by the Transformer in commit 47d6da4, so this runs against a
# git worktree at 47d6da4^ -- the last commit that still has it. That tree also predates
# every fix in sec. 7-9 (the missing decoder_optimizer.zero_grad(), the decoder never leaving
# train() mode, and so on), which is deliberate: 18136376 was run with exactly those bugs
# present, so this is comparable to its 0.794. Do not "fix" the worktree.
#
# Create it once:
#   git worktree add ~/gps-mlp-baseline 47d6da4^
#
# Flags match 18136376 exactly except --train_dataset_size. Note the absence of
# --decoder_use_cuda_graph, --min_qf_value and the decoder_* options: none of them exist in
# that tree. Determinism is on (the Config default), matching the baseline.
#
# 1M steps, not 500k: the MLP's best validation at 100 pairs landed at step 999,999 -- it was
# still improving when its run ended, unlike the Transformer which peaks around 200k. With
# more data it may need the full budget again. Expect ~10-11h.
#
#SBATCH --partition=main
#SBATCH --time=1-00:00:00
#SBATCH --job-name=h1_mlp_500
#SBATCH --output=sbatch/logs/h1_mlp_500pairs-%J.out
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

# Sanity check: the worktree must have the MLP decoder, not the Transformer. If this grep
# finds mingpt the worktree is at the wrong commit and the run would silently repeat the
# Transformer experiment.
if grep -q "mingpt_decoder" "${WORKTREE}/generative/models/dense_autoencoder/dense_var_auto_encoder_vin_1.py"; then
    echo "ERROR: ${WORKTREE} contains the Transformer decoder, not the MLP."
    echo "       Check it out at 47d6da4^ (the commit before 'transfor integration')."
    exit 1
fi

cd "${WORKTREE}" || exit 1
echo "running from $(pwd) at commit $(git rev-parse --short HEAD)"

START=$(date +%s)

poetry run python gps_simplegrid_levels.py \
    --cuda \
    --track \
    --obstacle_map 16x16_corridors_v1 \
    --train_dataset_size 500 \
    --train_eval_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --train_decoder_end_to_end \
    --no-initialize_decoder_from_pretrained

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
