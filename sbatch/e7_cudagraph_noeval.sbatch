#!/bin/bash
# E7 -- E1 with the decoder's decode loop replayed from a captured CUDA graph.
#
# Identical to E1 in every other respect, so E1 - E7 is the in-situ value of graphing the
# gradient-free decode paths (action selection and the per-step target-Q computation).
# The actor update stays eager, so this does not capture the whole decoder cost.
#
# E0 measured the per-call win on this hardware: 10.478 -> 1.228 ms at batch 1 (8.5x) and
# 10.921 -> 5.051 ms at batch 256 (2.2x). Projected over 1M steps that is ~1.6h off the
# target-Q term plus ~1.3h off action selection; the much larger eval-side win (~3.4h)
# shows up in E8, not here, because this run has eval off.
#
# Sanity check in the log -- capture should happen once per (batch, emit mode), so expect
# a small fixed number of these lines and no fallback warning:
#
#   grep "captured decoder CUDA graph" sbatch/logs/e7_*.out
#   grep "CUDA graph capture failed"   sbatch/logs/e7_*.out
#
#SBATCH --partition=main
#SBATCH --time=0-04:00:00
#SBATCH --job-name=e7_cudagraph
#SBATCH --output=sbatch/logs/e7_cudagraph_noeval-%J.out
# Pin the GPU type: the partition is heterogeneous (gtx_1080, rtx_2080, rtx_3090) and
# the 10.4h baseline ran on gtx_1080, so comparisons must stay on that class.
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
source sbatch/_common.sh

EXPERIMENT_NAME="E7: training loop only (no eval), decoder CUDA graph"
run_experiment \
    --decoder_use_cuda_graph \
    --val_eval_freq -1 \
    --train_eval_freq -1 \
    --eval_test_dataset_during_training_freq -1 \
    --val_dataset_size 20 \
    --train_eval_dataset_size 20
