#!/bin/bash
# J3 -- plain DQN (no proto-sequences at all) on any map.  The missing third arm.
#
#   sbatch sbatch/j3_dqn_map.sbatch <map> [train_pairs] [total_steps]
#   sbatch sbatch/j3_dqn_map.sbatch 16x16_obstacles_v1_25p
#
# WHY THIS EXISTS. J1 and J2 are both GPS -- Transformer decoder vs MLP decoder. Neither is
# a test of GPS itself. The sweep showed the winning arm (J2) emits plans averaging
# 1.1-1.8 actions, which is close to single-step control, so the obvious question is
# whether the sequence-level machinery is buying anything at all over a 1-step agent.
# That question needs a non-GPS baseline, and dqn_simplegrid_levels.py already is one.
#
#   J1 vs J2  -> which decoder (settled: MLP, except on corridors)
#   J2 vs J3  -> does GPS beat plain RL?   <-- this file
#   J1 vs J3  -> does our Transformer GPS beat plain RL?
#
# Read J2 vs J3 on success FIRST, then on the sequence metrics. If GPS and DQN tie on
# success, GPS's claim has to rest on the efficiency metrics (decisions per episode,
# wall-clock per optimal step), not on reaching the goal more often.
#
# STEP UNITS ARE COMPARABLE. Both scripts increment global_step once per environment step
# -- GPS does it inside its action-sequence loop (gps_simplegrid_levels.py:1882), DQN in
# its main loop. So 1M here is the same budget the J2 MLP arm got. Do NOT use the DQN
# script's own 80,000 default: that is 12.5x less environment experience and would make
# any GPS win meaningless.
#
# The paper also ships temporl_simplegrid_levels.py and dar_simplegrid_levels.py, two
# temporal-abstraction baselines. Copy this file for those if the DQN result is close --
# they are the methods GPS most needs to beat, since they also act over multiple steps.
#
#SBATCH --partition=main
#SBATCH --time=1-00:00:00
#SBATCH --job-name=j3_dqn_map
#SBATCH --output=sbatch/logs/j3_dqn_map-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

export CUBLAS_WORKSPACE_CONFIG=:4096:8

MAP="$1"
TRAIN_PAIRS="${2:-500}"
TOTAL_STEPS="${3:-1000000}"

if [ -z "${MAP}" ]; then
    echo "ERROR: pass the obstacle map, e.g."
    echo "  sbatch $0 16x16_obstacles_v1_25p"
    echo "  sbatch $0 16x16_obstacles_v1_25p 500 1000000"
    exit 1
fi

echo "=== J3: map=${MAP}  train_pairs=${TRAIN_PAIRS}  total_steps=${TOTAL_STEPS} ==="

START=$(date +%s)

poetry run python dqn_simplegrid_levels.py \
    --cuda \
    --track \
    --obstacle_map "${MAP}" \
    --train_dataset_size "${TRAIN_PAIRS}" \
    --train_eval_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --total_timesteps "${TOTAL_STEPS}" \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
