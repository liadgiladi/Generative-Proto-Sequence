#!/bin/bash
# G1 -- more training data: 300 start/goal pairs instead of 100.
#
# THE OBSERVATION
#
# At the validation peak (step ~200k) the best config sits at train 0.89 / val 0.72 -- a
# generalisation gap of 0.17. That gap is the cap: if the model generalised from what it has
# already learned, val would be 0.89, comfortably past the MLP's 0.794. The problem is not
# capacity and not learning. It is generalisation from 100 start/goal pairs driving 1M steps.
#
# Supporting evidence, from p1's 1M run:
#   train 0.89 -> 0.78  and  val 0.72 -> 0.48  over the back half  (gap widens 0.17 -> 0.30)
#   the no-penalty baseline also decays (train 0.73 -> 0.68), so degradation is intrinsic to
#     the setup rather than caused by the length penalty
#   losses/critic_values 0.40-0.53 -- the critic is healthy, not saturated
#   F1's larger decoder was on the same trajectory before it NaN'd, so capacity is not the
#     binding constraint either
#
# Everything points at the 0.17 gap, and --train_dataset_size is the only reason it is 100.
#
# READING IT
#
#   val peak up, gap narrower  -> generalisation was the cap; push further (500, or None for
#                                 the full split) and re-examine whether the MLP gap survives
#   val peak flat, gap same    -> 100 pairs were not the limit; the gap is architectural, and
#                                 reconstruction pretraining is the remaining lever
#   val peak down              -> 300 pairs need more than 500k steps to learn; check
#                                 "best val mean global step" before concluding anything
#
# Watch "best val mean global step": more pairs plausibly peak later, and a peak near 500k
# means the run was cut short rather than that the change failed.
#
# Single variable against the best config (d6): dataset size only. Note train_eval_dataset_size
# is left at its default of 100 so the train-side evaluation stays comparable across runs.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=g1_data_300
#SBATCH --output=sbatch/logs/g1_train_data_300-%J.out
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
    --train_dataset_size 300 \
    --train_eval_dataset_size 100 \
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
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
