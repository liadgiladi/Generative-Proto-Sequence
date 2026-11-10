#!/bin/bash
# D1 -- A3 repeated. The control for wave D, and a second sample of the best result.
#
# A3 (job 23446342) scored 0.582 test success, against 0.432 for the fixed baseline and
# 0.509 for the prior GPT run -- the first GPT configuration to beat either. But that is
# one run, and run-to-run variance is ~0.05 (0.432 vs 0.488 on the same config earlier).
# +0.15 is 3x the noise floor so it is probably real; this confirms it.
#
# Identical config to A3 on purpose. --no-torch_deterministic means kernel reduction order
# differs between runs, so the trajectory diverges and this is an independent sample
# despite the same seed -- no --seed change needed.
#
# Also the first run to carry the wandb summary fix, so the test numbers land as columns
# in the runs table instead of only in this log.
#
# D2 and D3 each change exactly one thing against this.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=d1_a3_repeat
#SBATCH --output=sbatch/logs/d1_a3_repeat-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

START=$(date +%s)

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
    --n_proto_plan_candidates 1 \
    --min_qf_value -20.0 \
    --max_qf_value 1.0 \
    --sequence_length_weight 0.0005 \
    --min_lengths_per_head 4 \
    --max_lengths_per_head 10 \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
