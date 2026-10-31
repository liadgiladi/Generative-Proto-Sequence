#!/bin/bash
# D3 -- A3 + a 3x larger actor learning rate.  (one variable against D1)
#
# The cheapest test of the weak-gradient theory: one flag, no code change.
#
# §8.6 measured the actor's gradient norm at 0.197-0.265 against the MLP decoder's 0.387,
# and the actor barely learned (embedding cosine 0.706 -> 0.619, where the MLP's went
# 1.0 -> 0.507). If the signal is simply ~2x too small, raising actor_learning_rate from
# 1e-4 to 3e-4 compensates directly.
#
# The competing explanation is that the actor is not under-trained but sitting in a safe
# local optimum -- a length-1 plan cannot commit to a wrong multi-step move, so the critic
# prefers it. A3 scoring +0.15 while B2 (a gradient fix) scored noise is evidence for that
# reading. Under it, a larger learning rate just reaches the same optimum faster and D3
# lands on D1.
#
# Either outcome is informative, which is what makes a one-flag run worth a slot.
#
# Risk: a larger actor step with a critic that saturates is a plausible way to destabilise.
# --min_qf_value is active here (as in all of wave D), and a NaN now ends the run with a
# final test evaluation rather than a traceback (§8.1, §9.3).
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=d3_a3_lr
#SBATCH --output=sbatch/logs/d3_a3_actor_lr-%J.out
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
    --actor_learning_rate 3e-4 \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
