#!/bin/bash
# P1 -- the headline run: best-known config at the MLP baseline's full 1M steps.
#
# Everything in wave D ran 500k steps, because validation peaks at 265k-355k and the second
# half changes nothing (best-checkpoint saving keeps the peak). That is true but it invites
# the obvious question -- did you run as long as the baseline? This removes it.
#
# Matched to the MLP baseline (slurm 18136376, cs-1080-02, gtx_1080, 10:24:26, 0.794):
#   same 1,000,000 steps
#   same epsilon schedule (--total_steps_e 15000, identical in absolute steps)
#   same obstacle map, dataset sizes and seed
#   same determinism setting -- note the ABSENCE of --no-torch_deterministic here, unlike
#     every wave D run. Determinism costs ~1.71x (E1 vs E5, via cuBLAS, not the
#     convolutions) but the baseline paid it too, so the runtime comparison is honest only
#     if this does as well.
#
# Best-known config, from §9.10:
#   --sequence_length_weight 0.002, bands 4-10   (5e-4 -> 0.619, 2e-3 -> 0.725, 4e-3 -> 0.742;
#                                                 the last step is inside the 0.042 noise)
#   --min_qf_value -20.0                         (activates Q clipping, dead config until §9.2)
#   --n_proto_plan_candidates 1                  (5 diverges through maximisation bias, §9.1)
#   no --decoder_soft_token_feedback             (+0.009 on this config, i.e. noise -- it
#                                                 overlaps with the length penalty, §9.10)
#
# Expected ~11.5-12h, from the deterministic 1M baseline run at 11.29h (job 23430936).
# For reference the non-deterministic equivalent was 7.62h (job 23430937).
#
# One deliberate difference from the baseline: --eval_test_dataset_during_training_freq -1.
# Those periodic test evaluations feed no model selection and no training signal -- only
# logging -- and cost ~1.3h. The final test evaluation still runs. So this run's wall clock
# is ~1.3h optimistic against the baseline's 10:24:26; the accuracy comparison is unaffected.
#
# Expectation: ~0.725 (the 500k result), since validation peaks well before 500k. If 1M
# gives materially more, that invalidates the "500k loses nothing" assumption behind every
# wave D run and is worth knowing.
#
#SBATCH --partition=main
#SBATCH --time=1-00:00:00
#SBATCH --job-name=p1_full_1m
#SBATCH --output=sbatch/logs/p1_full_1m-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

# Required by torch.use_deterministic_algorithms, which Config.torch_deterministic enables
# by default. Absent, the first cuBLAS matmul raises.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

START=$(date +%s)

poetry run python gps_simplegrid_levels.py \
    --cuda \
    --track \
    --obstacle_map 16x16_corridors_v1 \
    --train_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --train_decoder_end_to_end \
    --no-initialize_decoder_from_pretrained \
    --decoder_use_cuda_graph \
    --eval_test_dataset_during_training_freq -1 \
    --total_timesteps 1000000 \
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
