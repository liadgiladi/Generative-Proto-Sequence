#!/bin/bash
# E5 -- E1 without deterministic algorithms.
#
# torch_deterministic=True calls torch.use_deterministic_algorithms(True) and sets
# cudnn.deterministic, and CUBLAS_WORKSPACE_CONFIG=:4096:8 exists only to satisfy it.
# Together these force cuDNN and cuBLAS onto reproducible but slower kernels for the
# actor/critic convolutions. Note the deliberate absence of the CUBLAS_WORKSPACE_CONFIG
# export below -- it is not needed once determinism is off.
#
# This buys speed at the cost of exact run-to-run reproducibility, so only adopt it if the
# gain is worth that trade for your experiments.
#
#SBATCH --partition=main
#SBATCH --time=0-04:00:00
#SBATCH --job-name=e5_nondet
#SBATCH --output=sbatch/logs/e5_nondeterministic-%J.out
# Pin the GPU type: the partition is heterogeneous (gtx_1080, rtx_2080, rtx_3090) and
# the 10.4h baseline ran on gtx_1080, so comparisons must stay on that class.
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

source sbatch/_common.sh

EXPERIMENT_NAME="E5: training loop only (no eval), torch_deterministic=False"
run_experiment \
    --no-torch_deterministic \
    --val_eval_freq -1 \
    --train_eval_freq -1 \
    --eval_test_dataset_during_training_freq -1 \
    --val_dataset_size 20 \
    --train_eval_dataset_size 20
