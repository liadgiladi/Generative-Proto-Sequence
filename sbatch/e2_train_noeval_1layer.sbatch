#!/bin/bash
# E2 -- E1 with a single-block decoder trunk.
#
# The decode loop runs one trunk pass per generated token, so halving n_layer halves the
# number of kernel launches in the decoder. Comparing E2 against E1 tells you what
# fraction of the per-step cost is the decoder trunk itself: if E2 is barely faster, the
# decoder is not the bottleneck and the remaining time is replay sampling, the CNNs and
# the environment.
#
#SBATCH --partition=main
#SBATCH --time=0-04:00:00
#SBATCH --job-name=e2_noeval_1layer
#SBATCH --output=sbatch/logs/e2_train_noeval_1layer-%J.out
# Pin the GPU type: the partition is heterogeneous (gtx_1080, rtx_2080, rtx_3090) and
# the 10.4h baseline ran on gtx_1080, so comparisons must stay on that class.
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

export CUBLAS_WORKSPACE_CONFIG=:4096:8
source sbatch/_common.sh

EXPERIMENT_NAME="E2: training loop only (no eval), decoder n_layer=1"
run_experiment \
    --decoder_n_layer 1 \
    --val_eval_freq -1 \
    --train_eval_freq -1 \
    --eval_test_dataset_during_training_freq -1 \
    --val_dataset_size 20 \
    --train_eval_dataset_size 20
