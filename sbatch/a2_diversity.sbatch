#!/bin/bash
# A2 -- Enable the diversity regulariser against the observed mode collapse.
#
# sequences/num_unique falls monotonically across training in every run so far
# (earthy-bird-16: a clean "########------" decay from max to min). The decoder converges
# on fewer and fewer distinct action sequences, which caps how much the critic can
# discriminate between proto-plans. diversity_weight defaults to -1, i.e. disabled, and
# compute_diversity_loss_with_normalized_embeddings exists for exactly this.
#
# Note the regulariser acts on the *actor's* head embeddings (cosine similarity between
# heads), so it only has pairs to work on when there is more than one head -- hence
# n_proto_plan_candidates 5 here too. That makes this A1 + diversity rather than a clean
# single-variable change; run it alongside A1 so the pair isolates the regulariser's
# contribution.
#
# Compare against: A1 (same config, diversity off), not against the 1-head baseline.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=a2_diversity
#SBATCH --output=sbatch/logs/a2_diversity-%J.out
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
    --n_proto_plan_candidates 5 \
    --diversity_weight 0.1 \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
