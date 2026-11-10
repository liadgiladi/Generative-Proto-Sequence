#!/bin/bash
# Evaluate a finished run's saved checkpoints on the test set.
#
# Training saves its best-on-validation actor, critic and decoder to runs/<run_name>/ every
# time the validation success rate improves, but the final test evaluation only happens if
# train() reaches its end. A run that crashed or aborted partway leaves a perfectly good
# model on disk with no test numbers attached. This recovers them -- minutes, not hours.
#
# Usage -- pass the run directory as the one argument:
#
#   sbatch sbatch/eval_checkpoint.sbatch runs/SimpleGrid-v0__gps_simplegrid_levels__earthy-bird-16
#
# The training flags below must match the run being evaluated: anything that changes network
# shape or the dataset split (obstacle_map, dataset sizes, seed, max_level, the decoder_*
# options, actor/critic layer sizes) will either fail to load the state dicts or silently
# evaluate against a different test split. These match run_gps.sbatch.
#
#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --job-name=eval_checkpoint
#SBATCH --output=sbatch/logs/eval_checkpoint-%J.out
# Same GPU class as the training runs, so the timing statistics the evaluation prints
# (inference time per decision, wall clock per optimal step) stay comparable.
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

export CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN_DIR="$1"

if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: pass the run directory, e.g."
    echo "  sbatch $0 runs/SimpleGrid-v0__gps_simplegrid_levels__earthy-bird-16"
    exit 1
fi

if [ ! -d "${RUN_DIR}" ]; then
    echo "ERROR: ${RUN_DIR} is not a directory"
    exit 1
fi

echo "=== Evaluating checkpoints in ${RUN_DIR} ==="
echo "NODE=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
ls -la "${RUN_DIR}"

poetry run python eval_checkpoint.py \
    --run_dir "${RUN_DIR}" \
    --cuda \
    --obstacle_map 16x16_corridors_v1 \
    --train_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --train_decoder_end_to_end \
    --no-initialize_decoder_from_pretrained \
    --decoder_use_cuda_graph

echo
echo "=== SUMMARY ==="
echo "(the line to compare against the 0.509 baseline is 'mean success')"
