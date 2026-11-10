#!/bin/bash
# I2 -- Transformer decoder WITHOUT the length penalty, on the obstacle-free 16x16 map.
#
# The pure architecture test. The paper's code contains the torch.floor bug that silently
# disabled --sequence_length_weight, so the MLP baseline (I3) was never able to use it.
# Running the Transformer with it off puts both decoders on the same footing: same reward,
# same critic, same actor, same everything except how z becomes an action sequence.
#
#   I2 vs I3  -> decoder vs decoder, at the paper's own settings
#   I1 vs I3  -> our tuned method vs their published one
#
# Expect this to be well below I1. On corridors the penalty was worth roughly +0.3 of the
# total gain (0.418 -> 0.725), and without it the actor collapses to single-action plans
# (check_z_influence measured mean_len = 1.01). That is the point: it establishes how much
# of the Transformer's performance is the architecture and how much is the regulariser.
#
# The other sec. 7-9 fixes stay ON -- decoder_optimizer.zero_grad(), the decoder leaving
# train() mode, Q clipping. Those are bug fixes to shared GPS machinery, not Transformer
# features, and leaving them broken would handicap this run for no reason. The MLP baseline
# ran with the first two bugs present, which is a real asymmetry worth stating when
# reporting: it had an accumulating decoder gradient and inference-time dropout. Both were
# measured to matter little (the fixes bought ~0 on their own), but it is not nothing.
#
# Single variable against I1: --sequence_length_weight removed.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=i2_empty_plain
#SBATCH --output=sbatch/logs/i2_empty_gpt_plain-%J.out
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
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
