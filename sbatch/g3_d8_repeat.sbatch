#!/bin/bash
# G3 -- second seed of D8 (length weight 4e-3).
#
# The cheapest upgrade available to the number you would report.
#
# The strong-penalty family currently reads:
#
#   D4  length 2e-3                      0.741
#   D6  length 2e-3                      0.709
#   P1  length 2e-3, 1M, deterministic   0.706
#   D7  length 2e-3 + soft feedback      0.734
#   D8  length 4e-3                      0.742   <- highest, ONE seed
#                                        ---------------------------
#                                        pooled mean 0.726 +/- 0.018 (n=5)
#
# No variant is distinguishable from another (D7 +0.009, D8 +0.017 against the 2e-3 runs,
# both inside the noise), so the honest headline today is the pooled 0.726, not 0.742 --
# quoting the maximum of five runs is selecting on noise, especially against an MLP
# baseline that is itself a single seed.
#
# This run is what would change that. If 4e-3 lands near 0.74 again, it becomes a config
# with two seeds in its own right and the reportable figure moves from 0.726 to ~0.74 --
# which is the stated target. If it lands near 0.70, 0.742 was the top of the family's
# distribution and 0.726 stands.
#
# Config identical to d8_length_4e3.sbatch. --no-torch_deterministic makes this an
# independent sample without changing --seed.
#
# One thing to also record, since it has never been measured on a 4e-3 run: at 2e-3 the
# penalty contributed ~71% of the Q term (sequence_length_loss 231 x 2e-3 = 0.462 against a
# Q term of 0.646; note the logged actor_loss already includes the penalty, so the Q term is
# actor_loss minus that product). At 4e-3 it is roughly 1.4x the Q term. Watch
# "mean num decoder generations" in the evaluations -- D4 reached 4.5, and if this run
# pushes it toward 1-2 the plans are being stretched to fill the band regardless of the task.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=g3_d8_repeat
#SBATCH --output=sbatch/logs/g3_d8_repeat-%J.out
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
