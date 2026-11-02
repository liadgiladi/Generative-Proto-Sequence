#!/bin/bash
# F1 -- restore the decoder capacity that §6.3 removed for a reason that turned out false.
#
# THE OBSERVATION
#
# The Transformer stalls. Best validation arrives at step 200,000 and never improves again,
# while the MLP baseline's best validation lands at step 999,999 -- still climbing when its
# run ended. At matched steps the MLP is also ahead (0.77 val at ~215k vs 0.72 at 200k), so
# this is not a sample-efficiency trade: the MLP dominates at every point and the
# Transformer simply stops. Learning fast and then stalling is the signature of running out
# of capacity, not of a bad optimiser.
#
# THE SUSPECT
#
# §6.3 shrank the decoder from n_layer=3 / n_embd=64 (152,512 params) to n_layer=2 /
# n_embd=32 (26,688) -- a 5.7x cut -- explicitly to reduce runtime.
#
# §7.2 then measured the decoder and found it is ~100% dispatch overhead, not arithmetic:
# embd 32->16 is 3.5x fewer parameters and 1.6% faster; the per-call cost fits
# 3.73ms + 3.91ms * n_layer, where the constant is the Python loop itself. The shrink was
# decided before that was known, and bought almost nothing.
#
# So the ceiling may be one we imposed on ourselves. This restores the original size. Cost
# should be roughly one extra n_layer's worth of dispatch, ~10-15% wall clock -- nothing
# like the 5.7x parameter change suggests.
#
# READING IT
#
#   best val step moves past 200k      -> capacity was the binding constraint; sweep size
#                                         further (n_layer 4, n_embd 128) before anything else
#   best val still ~200k, score ~0.72  -> capacity is not it; the stall is elsewhere
#                                         (length penalty schedule, critic saturation,
#                                         overfitting to 100 training pairs)
#   worse                              -> the larger decoder is harder to optimise from
#                                         sparse reward, which is what §6.3 originally argued
#
# Watch "best val mean global step" as much as the final score -- the question is whether the
# stall moves, not only whether the number improves.
#
# Single variable against the best-known config (d6/p1): decoder size only. 500k steps and
# non-deterministic for turnaround, since p1 showed 1M adds nothing (best val at 200k) and
# determinism costs 1.71x without affecting accuracy (0.706 deterministic vs 0.741/0.709 not).
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=f1_capacity
#SBATCH --output=sbatch/logs/f1_capacity_restore-%J.out
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
    --decoder_n_layer 3 \
    --decoder_n_embd 64 \
    --decoder_n_head 4 \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
