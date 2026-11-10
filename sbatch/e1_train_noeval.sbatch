#!/bin/bash
# E1 -- Training-loop baseline, evaluation switched off.
#
# Isolates the cost of the RL update itself (target-Q decode every step + actor update
# every 4 steps) with no evaluation at all. This is the number every other experiment is
# measured against, and the one that tells you whether the decoder's per-step cost or the
# evaluation schedule is the thing to attack.
#
#SBATCH --partition=main
#SBATCH --time=0-04:00:00
#SBATCH --job-name=e1_train_noeval
#SBATCH --output=sbatch/logs/e1_train_noeval-%J.out
# Pin the GPU type: the partition is heterogeneous (gtx_1080, rtx_2080, rtx_3090) and
# the 10.4h baseline ran on gtx_1080, so comparisons must stay on that class.
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
source sbatch/_common.sh

EXPERIMENT_NAME="E1: training loop only (no eval), decoder n_layer=2"
run_experiment \
    --val_eval_freq -1 \
    --train_eval_freq -1 \
    --eval_test_dataset_during_training_freq -1 \
    --val_dataset_size 20 \
    --train_eval_dataset_size 20
