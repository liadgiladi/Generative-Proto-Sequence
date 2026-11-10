#!/bin/bash
# D8 -- length weight 4e-3, one more point on the curve.
#
# The length penalty is the strongest lever found so far, and we have two points:
#
#   5e-4  (A3)  0.582, 0.611, 0.665   mean 0.619   mean_len 2.53
#   2e-3  (D4)  0.741                              ~4.5 decoder generations/episode
#
# +0.122 for a 4x weight increase. This doubles it again to find the peak. The question is
# not "is longer better" but "where does it stop paying" -- and the answer bounds the lever.
#
#   > 0.741  -> still climbing; sweep further (8e-3) before stacking anything else
#   ~0.741   -> plateau between 2e-3 and 4e-3; 2e-3 is the operating point
#   < 0.70   -> past the peak; the optimum sits at or below 2e-3
#
# Scale check: at 2e-3 the penalty contributed 0.462 against a Q term of 0.646 (71%). At
# 4e-3 it will be roughly 1.4x the Q term, so this run genuinely risks optimising plan
# length over task success -- which is itself the thing worth measuring, and the reason not
# to jump straight to 8e-3. Watch losses/sequence_length_loss x weight against
# losses/actor_loss; note the logged actor_loss already includes the penalty, so the Q term
# is actor_loss minus that product.
#
# Also watch "mean num decoder generations" in the evaluations: D4 reached 4.5 (val). If
# this run drives it toward 1-2, plans are being stretched to fill the band regardless of
# the task, and the success number will show it.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=d8_length_4e3
#SBATCH --output=sbatch/logs/d8_length_4e3-%J.out
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
    --sequence_length_weight 0.004 \
    --min_lengths_per_head 4 \
    --max_lengths_per_head 10 \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
