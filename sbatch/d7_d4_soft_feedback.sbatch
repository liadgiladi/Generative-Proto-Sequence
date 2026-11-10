#!/bin/bash
# D7 -- D4 (length 2e-3) + soft token feedback.  Stacking the two established gains.
#
# This is the shot at the MLP's 0.794. Both components are now supported by evidence, and
# they target different mechanisms:
#
#   length penalty   -> longer plans. D4 (2e-3) scored 0.741 vs A3 (5e-4) at 0.619 mean;
#                       decoder generations per episode fell from ~20 to 4.5.
#   soft feedback    -> gradient reaches z through all 10 decode steps instead of stopping
#                       at a detached argmax. D2 = 0.691, 0.675 (mean 0.683) vs A3's 0.619.
#
# They have already been shown to compose once: soft feedback ALONE was 0.480 -- worse than
# A3 -- but on top of A3 it was 0.683. The reason is mechanical: gradient flowing through
# the decode chain has little to act on when plans are one action long (check_z_influence
# measured mean_len = 1.01 on the baseline), and the length penalty is what makes them long.
# D4 makes them longer still, so the chain soft feedback acts on is longer again.
#
# If the two gains are additive from A3's 0.619 (+0.122 and +0.064), this lands near 0.805.
# They may of course overlap, in which case expect ~0.74-0.76.
#
# NOT a single-variable run against D4 -- it is the combination. D6 is D4's control.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=d7_d4_soft
#SBATCH --output=sbatch/logs/d7_d4_soft_feedback-%J.out
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
    --sequence_length_weight 0.002 \
    --min_lengths_per_head 4 \
    --max_lengths_per_head 10 \
    --decoder_soft_token_feedback \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
