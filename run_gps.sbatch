#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-48:00:00 
#SBATCH --job-name=gps_mingpt_maze
#SBATCH --output=gps_maze_job-%J.out
# Pin the GPU type. The partition is heterogeneous (gtx_1080, rtx_2080, rtx_3090) and a
# bare --gpus=1 will happily hand out a 2080 Ti, which makes the runtime incomparable to
# the 10.4h MLP baseline (job 18136376, cs-1080-02, gtx_1080).
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# Load conda and activate your environment
module load anaconda
source activate gps_env

# wandb credentials come from ~/.netrc, written once by running `wandb login` on the login
# node. The home directory is shared with the compute nodes, so batch jobs pick it up.
#
# Do NOT hardcode WANDB_API_KEY here. This file is committed, and wandb scans repositories
# for leaked keys and revokes them -- which is what killed jobs 23429516 and 23429517 with
# "CommError: 'NoneType' object has no attribute 'get'" twenty seconds after startup.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

START=$(date +%s)

# Run your training command
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
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"

# --decoder_use_cuda_graph                   measured 30.54h -> 15.27h (E3 vs E8)
# --eval_test_dataset_during_training_freq   the periodic test-set evaluations feed no
#     model selection and no training signal -- only logging. The final test evaluation at
#     the end of train() still runs, so the headline accuracy number is unaffected.
# --slurm_job_id                             lands in wandb.config, so a run can be found
#     by job id instead of grepping logs for its wandb URL.
