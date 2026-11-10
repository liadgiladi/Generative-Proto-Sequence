#!/bin/bash
# E8 -- E3 with the decoder CUDA graph, i.e. graphing where it actually pays.
#
# Evaluation decodes at batch 1, which is exactly where E0 measured the biggest win
# (10.478 -> 1.228 ms, 8.5x), and at ~20 decoder generations per episode it is the single
# largest decoder term in a production run (~3.84h -> ~0.41h projected over 1M steps).
# E3 - E8 is that win measured in situ.
#
# This is also the closest thing in the matrix to a production candidate: same eval
# schedule as the real run, both bug fixes, and the graph enabled. Its success rate is
# still not meaningful at 30k steps -- only its timing is.
#
#SBATCH --partition=main
#SBATCH --time=0-08:00:00
#SBATCH --job-name=e8_cudagraph_eval
#SBATCH --output=sbatch/logs/e8_cudagraph_eval-%J.out
# Pin the GPU type: the partition is heterogeneous (gtx_1080, rtx_2080, rtx_3090) and
# the 10.4h baseline ran on gtx_1080, so comparisons must stay on that class.
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
source sbatch/_common.sh

EXPERIMENT_NAME="E8: eval every 5000 steps, decoder CUDA graph"
run_experiment \
    --decoder_use_cuda_graph \
    --val_eval_freq 5000 \
    --train_eval_freq 5000 \
    --eval_test_dataset_during_training_freq -1 \
    --val_dataset_size 100 \
    --train_eval_dataset_size 100
