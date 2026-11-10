#!/bin/bash
# I1 -- Transformer decoder, best config, on the obstacle-free 16x16 map.
#
# Everything so far has been measured on 16x16_corridors_v1. This is the first test of
# whether any of it generalises. 16x16_empty is the same size with no obstacles, so it
# isolates the obstacle factor rather than confounding it with grid size.
#
# Three runs make up this comparison:
#   I1  GPT, length penalty ON   (this file)  -> our best method
#   I2  GPT, length penalty OFF              -> pure architecture test vs I3
#   I3  MLP, paper code unmodified           -> faithful published baseline
#
# I2 vs I3 is the clean decoder-vs-decoder comparison at the paper's own settings, because
# the paper's code contains the torch.floor bug that silently disabled
# --sequence_length_weight -- so their published numbers were produced without it, and our
# best config depends on it for most of its gain. I1 vs I3 is "our method vs theirs".
# Reporting only I1 vs I3 would credit the Transformer for a dead feature we switched on.
#
# 100 training pairs, matching the paper's protocol. Note that on corridors this is
# severely data-starved -- 100 pairs gave 0.725, 300 gave 0.897, 500 gave 0.99 -- so expect
# all three of these to understate what the setup can do. 500-pair versions afterwards will
# say whether that finding is specific to corridors.
#
# Config otherwise identical to d6_d4_repeat.sbatch (the best-known config).
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=i1_empty_gpt
#SBATCH --output=sbatch/logs/i1_empty_gpt_tuned-%J.out
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
    --obstacle_map 16x16_empty \
    --train_dataset_size 100 \
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
