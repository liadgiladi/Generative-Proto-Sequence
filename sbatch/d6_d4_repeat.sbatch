#!/bin/bash
# D6 -- second seed of D4 (length weight 2e-3), the current best config.
#
# Standing results (test success, stochastic pass):
#
#   baseline                          0.432, 0.403        mean 0.418
#   B2  soft feedback alone           0.479, 0.481        mean 0.480
#   A3  length 5e-4                   0.582, 0.611, 0.665 mean 0.619, sd 0.042
#   D2  A3 + soft feedback            0.691, 0.675        mean 0.683
#   D3  A3 + actor lr 3e-4            0.445               REFUTED
#   D4  A3 with length 2e-3           0.741               <- best, ONE seed
#   MLP target                        0.794
#
# D4 answered the headroom question: 2.53 actions per plan was not the optimum. Raising the
# weight 4x drove mean decoder generations per episode from ~20 to 4.5 -- i.e. much longer
# plans -- and success to 0.741, which is +0.122 over A3's mean (~2.9 sigma) and only 0.053
# short of the MLP.
#
# It is one seed. Everything downstream depends on it, so confirm before building on it.
# --no-torch_deterministic makes this an independent sample without changing --seed.
#
#   ~0.74 again -> D4 becomes the baseline; stack soft feedback on it (D7) and chase 0.794
#   ~0.65       -> D4 drew from the top of the distribution; A3 at 5e-4 and 2e-3 are
#                  equivalent and the length lever is flat between them
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=d6_d4_repeat
#SBATCH --output=sbatch/logs/d6_d4_repeat-%J.out
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
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
