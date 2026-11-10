#!/bin/bash
# A3 -- Penalise proto-plans that are too short.  (single-candidate version)
#
# check_z_influence.py showed the actor has collapsed onto embeddings that decode to
# length-1 plans:
#
#   actor z: mean pairwise cosine similarity = 0.7783  (1.0 = all identical)
#   sequences from actor z: unique(trimmed) = 5/256, mean_len = 1.01
#
# while the same trained decoder fed *random* z produces 44 unique sequences at mean_len
# 1.77 -- the decoder is healthy, the actor almost never asks for a long plan. A length-1
# plan is a safe local optimum: it cannot commit to a wrong multi-step move, so the critic
# prefers it and the proto-plan idea degenerates into a 1-step policy.
#
# This pushes plans into a 4-10 action band, attacking mean_len=1.01 head-on.
#
# SINGLE CANDIDATE. The earlier 5-head version of this experiment peaked at step 60000 and
# then degraded for 250k steps before dying on NaN Q-values: the Bellman target takes
# argmax over candidates, which is maximisation bias with 5 and a no-op with 1. The length
# bands are therefore single-element here rather than the 5-element Config defaults --
# sequence_length_weight asserts len(min_lengths_per_head) == n_proto_plan_candidates.
#
# --min_qf_value is set because the Q-clipping is gated on it and has never been active in
# any run (max_qf_value=1.0 was dead config). Episodic returns span roughly -20 to +1. Both
# 1-candidate full runs still hit NaN gradients (steps 625476 and 918000), so bounding Q
# matters even without extra candidates.
#
# Only works at all because of the torch.floor removal in
# compute_sequence_lengths_differentiable -- floor has zero gradient, so before that fix
# this penalty was logged but could not influence training.
#
# WEIGHT SCALING: the penalty is a *sum* over batch x heads = 256 x 1 = 256 terms, each up
# to ~3 when plans are length 1 (min band 4). Raw penalty is in the hundreds while
# actor_loss is ~0.4, hence 5e-4. Watch losses/sequence_length_loss x weight against
# losses/actor_loss: if it dwarfs the Q term the agent optimises length instead of success.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=a3_length
#SBATCH --output=sbatch/logs/a3_length-%J.out
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
