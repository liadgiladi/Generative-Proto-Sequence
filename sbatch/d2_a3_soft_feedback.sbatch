#!/bin/bash
# D2 -- A3 + soft token feedback.  (one variable against D1)
#
# B2 tested soft feedback on its own and scored 0.479 -- above the 0.432 baseline but
# inside the ~0.05 noise band, so unproven. A3 (length penalty) scored 0.582.
#
# The reason to combine them rather than treat B2 as refuted: soft feedback makes the
# 10-step decode chain differentiable so gradient reaches z through every position. On
# length-1 plans there is barely a chain for it to act on -- check_z_influence.py measured
# mean_len = 1.01 on the baseline actor. A3 is precisely the change that makes plans long.
# So the mechanism B2 provides may only pay once A3 has given it something to differentiate.
#
# If D2 > D1, that is the weak-gradient theory (§8.6 explanation (b)) finally showing up,
# and it explains why B2 alone looked like noise.
#
# Watch actor/embedding_cosine as well as the final score: the MLP's actor went 1.0 ->
# 0.507 while ours went 0.706 -> 0.619 in the old (head-axis) metric. Movement in the new
# actor/embedding_cosine is the mechanism signal independent of success rate.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=d2_a3_soft
#SBATCH --output=sbatch/logs/d2_a3_soft_feedback-%J.out
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
    --decoder_soft_token_feedback \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
