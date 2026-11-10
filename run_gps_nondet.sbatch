#!/bin/bash
# Production baseline, non-deterministic variant of run_gps.sbatch.
#
# Identical in every other respect, so this pair measures the cost of determinism at full
# scale. Measured at 30k steps on gtx_1080: 19.67h -> 11.52h with evaluation off (E1 vs
# E5), a 1.71x penalty. The cause is not the convolutions -- profiling put conv2d at 1.2%
# of the loop -- but cuBLAS: CUBLAS_WORKSPACE_CONFIG=:4096:8 exists purely to satisfy
# torch.use_deterministic_algorithms, and torch._C._nn.linear is ~30% of the training
# loop. Note the deliberate absence of that export below.
#
# Expected ~9h here versus ~15h for the deterministic run.
#
# The trade is exact run-to-run reproducibility. Results should still be statistically
# equivalent -- if the two runs' success rates differ noticeably, that is seed noise
# between two different RNG streams, not evidence that determinism affects learning.
#
#SBATCH --partition=main
#SBATCH --time=0-48:00:00
#SBATCH --job-name=gps_mingpt_maze_nondet
#SBATCH --output=gps_maze_nondet_job-%J.out
# Pin the GPU type. The partition is heterogeneous (gtx_1080, rtx_2080, rtx_3090) and a
# bare --gpus=1 will happily hand out a 2080 Ti, which makes the runtime incomparable to
# the 10.4h MLP baseline (job 18136376, cs-1080-02, gtx_1080).
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# Load conda and activate your environment
module load anaconda
source activate gps_env

# wandb credentials come from ~/.netrc (`wandb login`, run once on the login node).
# Do NOT hardcode WANDB_API_KEY here -- see the note in run_gps.sbatch.
# No CUBLAS_WORKSPACE_CONFIG here -- it is only needed for deterministic cuBLAS.

START=$(date +%s)

# Run your training command
poetry run python gps_simplegrid_levels.py \
    --cuda \
    --track \
    --no-torch_deterministic \
    --obstacle_map 16x16_corridors_v1 \
    --train_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --train_decoder_end_to_end \
    --no-initialize_decoder_from_pretrained \
    --decoder_use_cuda_graph \
    --eval_test_dataset_during_training_freq -1 \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
