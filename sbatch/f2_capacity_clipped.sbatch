#!/bin/bash
# F2 -- F1's larger decoder, with actor gradient clipping to keep it alive.
#
# WHY F1 DID NOT ANSWER THE QUESTION
#
# F1 (n_layer 3 / n_embd 64, restoring the pre-§6.3 size) scored 0.657 -- but it NaN'd at
# step 159,000, 32% through the run, with the NaN appearing in the ACTOR (cnn.0.weight),
# not the decoder. At matched steps it was not behind:
#
#   P1 (n_layer 2 / n_embd 32)   best val 0.64 @ 140k   -> went on to peak 0.72 @ 200k
#   F1 (n_layer 3 / n_embd 64)   best val 0.65 @ 150k   -> died at 159k
#
# So the larger decoder was on the same trajectory, marginally ahead, and never reached its
# peak. Capacity is OPEN, not refuted. The blocker is stability.
#
# THE PATTERN
#
# Two separate changes have now destabilised the actor specifically:
#   - a larger decoder (F1)                 -> NaN in cnn.0.weight at 159k
#   - actor_learning_rate 3e-4 (D3)         -> 0.445, far below the 0.726 family mean
# Both amplify what reaches the actor. The actor looks fragile to larger updates regardless
# of the cause -- and the training loop has NO gradient clipping anywhere (the only
# clipping in the file is clip_weights, which is dead code).
#
# --actor_grad_clip 1.0 clips the global gradient norm of the actor and, in end-to-end mode,
# the decoder. Typical actor gradient norms are 0.2-0.4, so 1.0 only bites on spikes and
# should not change normal training.
#
# Honest caveat: clipping is PREVENTIVE, not curative. It cannot repair a gradient that is
# already NaN -- clip_grad_norm_ on NaN input produces NaN. It works only if the NaN arises
# from a run of growing updates rather than appearing from nowhere. If F2 NaNs at a similar
# step, that distinction is the answer and clipping is not the fix.
#
# READING IT
#
#   survives to 500k, > 0.726  -> capacity WAS the binding constraint and clipping unlocked
#                                 it; sweep size further (n_layer 4, n_embd 128)
#   survives to 500k, ~0.726   -> clipping fixed the stability, extra capacity buys nothing;
#                                 the 200k stall is not a capacity ceiling. Question closed.
#   NaNs again around 159k     -> clipping does not prevent this failure; the instability is
#                                 not gradient magnitude. Look at the critic (saturation) or
#                                 the length penalty before trying capacity again.
#
# Single variable against F1: the clip. Two variables against the best config (d6): decoder
# size and the clip -- deliberate, since F1 already established size alone is not viable.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=f2_capacity_clip
#SBATCH --output=sbatch/logs/f2_capacity_clipped-%J.out
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
    --actor_grad_clip 1.0 \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
