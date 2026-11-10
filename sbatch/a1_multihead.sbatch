#!/bin/bash
# A1 -- Turn the multi-head proto-plan mechanism back on.
#
# n_proto_plan_candidates defaults to 1, which makes the method's central idea a no-op:
# with a single candidate there is nothing for the critic to select between, so
# eval_candidates_with_critic, select_action's sample_from_candidates branch and the
# per-head softmax weighting in the actor loss all collapse to identity. The Config still
# carries five-head settings (min_lengths_per_head = [1,2,4,6,8], max = [2,4,6,8,10]),
# which suggests 5 was the intended configuration.
#
# Nearly free at runtime: E0 measured the decoder as ~100% dispatch overhead, where batch
# 1 and batch 256 cost the same, so five candidates cost roughly what one did.
#
# Compare against: 0.488 test success (true-capybara-17, non-deterministic). Two runs of
# the same config spread 0.432-0.488, so treat anything under ~0.05 as noise.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=a1_multihead
#SBATCH --output=sbatch/logs/a1_multihead-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

# Non-deterministic: 1.71x faster (E1 vs E5) and these are exploratory sweeps where
# turnaround matters more than exact reproducibility. The comparison baseline is the
# non-deterministic run for the same reason. No CUBLAS_WORKSPACE_CONFIG needed.

START=$(date +%s)

# total_timesteps 500000: validation peaked at step 305000 and declined after, so the
# second half of a 1M-step run only costs time -- best-checkpoint saving already keeps
# the peak. ~4.5h here instead of ~9h.
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
    --total_timesteps 500000 \
    --total_steps_e 15000 \
    --n_proto_plan_candidates 5 \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
