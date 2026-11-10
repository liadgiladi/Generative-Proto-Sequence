#!/bin/bash
# E4 -- Identical to E3 but evaluating 5x less often.
#
# E3 minus E4 isolates the marginal cost of one evaluation round, which lets you price the
# eval schedule directly: if E3 is much slower than E4, raising val_eval_freq and
# train_eval_freq is the single cheapest win available, and it costs nothing but
# resolution on the learning curve.
#
#SBATCH --partition=main
#SBATCH --time=0-08:00:00
#SBATCH --job-name=e4_eval_25000
#SBATCH --output=sbatch/logs/e4_eval_freq25000-%J.out
# Pin the GPU type: the partition is heterogeneous (gtx_1080, rtx_2080, rtx_3090) and
# the 10.4h baseline ran on gtx_1080, so comparisons must stay on that class.
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
source sbatch/_common.sh

EXPERIMENT_NAME="E4: eval every 25000 steps"
run_experiment \
    --val_eval_freq 25000 \
    --train_eval_freq 25000 \
    --eval_test_dataset_during_training_freq -1 \
    --val_dataset_size 100 \
    --train_eval_dataset_size 100
