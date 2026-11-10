#!/bin/bash
# D4 -- A3 with a 4x stronger length penalty.  (one variable against D1)
#
# check_z_influence.py on A3's checkpoint (kind-bush-24) vs the baseline (earthy-bird-16):
#
#                               baseline (0.432)   A3 (0.582)
#   mean_len from actor z              1.01           2.53     <- +150%
#   unique plans from actor z          5/256         17/256    <- 3.4x
#   actor cosine                      0.7783         0.7892    <- unchanged
#   decoder on *random* z (len)        1.77           1.99      <- barely moved
#
# Two readings follow.
#
# 1. The penalty worked, and the gain tracks the mechanism: longer, more varied plans
#    actually reaching execution. Note Part A (random z) barely moved while Part B (actor
#    z) moved a lot -- training only reshapes the region the actor visits. The win came
#    from the decoder spreading the actor's narrow z-region across longer plans, NOT from
#    the actor becoming more state-dependent. The §8.3 actor collapse is still intact.
#
# 2. mean_len settled at 2.53 against a band asking for >= 4. The Q term is pushing back
#    and equilibrium landed below target, i.e. the penalty is under-powered. This run
#    raises the weight 5e-4 -> 2e-3 to shift that equilibrium and find out whether the
#    lever has more to give.
#
# Decisive either way:
#   mean_len ~3.5 AND success up  -> keep pushing; sweep further
#   mean_len up, success flat/down -> 2.53 was near optimal; lever spent, move to
#                                     multi-head (needs --min_qf_value, see §9.1) or
#                                     reconstruction pretraining (§6.5, needs §4.1-4.3)
#
# Genuinely uncertain rather than a formality: §6.5 records the MLP baseline running at
# ~1.6 valid actions per sequence while A3 is already at 2.53, so longer is not obviously
# monotonically better -- there may be an optimum we are approaching or have passed.
#
# Independent of D1/D2/D3; can run alongside them.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=d4_a3_len2
#SBATCH --output=sbatch/logs/d4_a3_stronger_length-%J.out
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
