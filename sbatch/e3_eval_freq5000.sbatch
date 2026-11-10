#!/bin/bash
# E3 -- Evaluation at the production schedule (every 5000 steps, 100 episodes each).
#
# E3 minus E1 is the cost of evaluation. This is also the experiment that validates the
# bug fixes, because its log carries the diagnostic that says whether the decoder is
# healthy. After the run, check:
#
#   grep "mean num valid actions in sequences" sbatch/logs/e3_*.out
#   grep "mean num decoder generations"        sbatch/logs/e3_*.out
#   grep -c "invalid - empty action sequence"  sbatch/logs/e3_*.out
#
# Healthy: valid-actions-per-sequence stays well above 1 and decoder-generations stays
# well below max_episode_steps (75). Degenerate: valid-actions trends to 1 and
# generations saturates at 75, which is what multiplied evaluation cost ~10x.
#
#SBATCH --partition=main
#SBATCH --time=0-08:00:00
#SBATCH --job-name=e3_eval_5000
#SBATCH --output=sbatch/logs/e3_eval_freq5000-%J.out
# Pin the GPU type: the partition is heterogeneous (gtx_1080, rtx_2080, rtx_3090) and
# the 10.4h baseline ran on gtx_1080, so comparisons must stay on that class.
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
source sbatch/_common.sh

EXPERIMENT_NAME="E3: eval every 5000 steps (production schedule)"
run_experiment \
    --val_eval_freq 5000 \
    --train_eval_freq 5000 \
    --eval_test_dataset_during_training_freq -1 \
    --val_dataset_size 100 \
    --train_eval_dataset_size 100
