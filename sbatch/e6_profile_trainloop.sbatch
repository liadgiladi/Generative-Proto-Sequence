#!/bin/bash
# E6 -- cProfile of the training loop, to attribute the non-decoder time.
#
# E0 measured the decoder at ~5.95h per 1M steps. The 9-hour baseline therefore implies
# ~8.6h of *non-decoder* work -- a bigger block than the decoder itself, and so far a
# black box. This run attributes it by function.
#
# Two specific suspects to look for in the output:
#
#   random.sample / deque.__getitem__   -- modules/replay_memory.py:23 samples a batch of
#       256 from a deque. random.sample indexes the population, and deque indexing walks a
#       linked list of blocks (O(min(i, n-i)), averaging ~12,500 hops at buffer_size
#       50000). That is ~3.2M pointer hops per training step.
#
#   torch.Tensor / stack                -- gps_simplegrid_levels.py:1970 rebuilds the
#       action batch with 256 individual legacy tensor constructions per step.
#
# cProfile cannot see asynchronous GPU time -- it attributes that to whatever call
# synchronises -- so read this for Python-side overhead, which is exactly what is
# suspected here. Absolute times are inflated by the profiler; the ranking is what counts.
#
# Deliberately NOT sourcing _common.sh: wandb syncing and the 30k-step budget would both
# distort the profile. 3000 steps is plenty to rank the hot functions.
#
#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --job-name=e6_profile
#SBATCH --output=sbatch/logs/e6_profile_trainloop-%J.out
# Pin the GPU type so the matrix stays on one class of hardware (see e0 for why). The
# profiler only needs relative function rankings, so it tolerates node-to-node variation
# better than the timing runs do -- but absolute times still shift with node CPU, since
# what this is measuring is CPU-side dispatch.
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

export CUBLAS_WORKSPACE_CONFIG=:4096:8

PROF_OUT="sbatch/logs/e6_profile_${SLURM_JOB_ID}.prof"

echo "=== E6: cProfile of the training loop ==="
echo "NODE=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

# total_steps_e must sit between learning_starts and total_timesteps (see Config
# __post_init__), so the epsilon schedule is compressed to match the short run.
poetry run python -m cProfile -o "${PROF_OUT}" gps_simplegrid_levels.py \
    --cuda \
    --obstacle_map 16x16_corridors_v1 \
    --train_dataset_size 100 \
    --total_timesteps 3000 \
    --total_steps_e 2000 \
    --learning_starts 1000 \
    --train_decoder_end_to_end \
    --no-initialize_decoder_from_pretrained \
    --val_eval_freq -1 \
    --train_eval_freq -1 \
    --eval_test_dataset_during_training_freq -1 \
    --val_dataset_size 20 \
    --train_eval_dataset_size 20 \
    --test_dataset_size 20

echo
echo "=== TOP 40 BY CUMULATIVE TIME (who owns the call tree) ==="
poetry run python -c "
import pstats
p = pstats.Stats('${PROF_OUT}')
p.strip_dirs().sort_stats('cumulative').print_stats(40)
"

echo
echo "=== TOP 40 BY SELF TIME (where the cycles actually go) ==="
poetry run python -c "
import pstats
p = pstats.Stats('${PROF_OUT}')
p.strip_dirs().sort_stats('tottime').print_stats(40)
"

echo
echo "=== REPLAY-BUFFER SAMPLING SPECIFICALLY ==="
poetry run python -c "
import pstats
p = pstats.Stats('${PROF_OUT}')
p.strip_dirs().print_stats('sample|deque|replay_memory|stack|Tensor')
"

echo "PROFILE_SAVED_TO=${PROF_OUT}"
