#!/bin/bash
# K1 -- check_z_influence.py as a job.  Separates actor collapse from decoder collapse.
#
#   sbatch sbatch/k1_zinfluence.sbatch <run-name-or-dir> <obstacle_map> [n_samples]
#
#   sbatch sbatch/k1_zinfluence.sbatch sunny-galaxy-68 24x24_empty
#   sbatch sbatch/k1_zinfluence.sbatch unique-snow-60  16x16_room_v1
#
# Pass either the bare wandb run name or the full runs/ directory; both resolve.
#
# WHY A JOB. The script needs torch, so it cannot run on a login node without the env, and
# dataset_setup calls precompute_path_lengths, which BFSes from every cell pair -- 576^2 on
# 24x24. That is enough to hit a login-node CPU limit.
#
# ALWAYS RUN A CONTROL ALONGSIDE. Part A/Part B numbers mean nothing in isolation: the 0.7783
# cosine in EVIDENCE.md sec. 4 came from a 100-pair corridors run and is not a valid reference
# for the 500-pair sweep. Pair every failing checkpoint with a healthy one from the same
# config -- e.g. 24x24_empty (0.110) against 16x16_room_v1 (0.961).
#
# READING IT:
#   Part B cosine near 1.0 + few unique plans -> the ACTOR collapsed. On 24x24 that is the
#     leading suspect: 484 free cells is ~2.5x the observation area the actor CNN was tuned on.
#   Part A worse than the untrained control      -> the DECODER broke. Would be new.
#   Both resemble the control                    -> neither half collapsed; look at the critic
#     or the reward scale on a larger grid.
#
#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --job-name=k1_zinfl
#SBATCH --output=sbatch/logs/k1_zinfluence-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

RUN="$1"
MAP="$2"
N_SAMPLES="${3:-256}"
# Must match the run being probed. check_z_influence.py inherits Config, whose default is
# 100 -- so without this it probes the actor on states drawn from a 100-pair dataset while
# the checkpoint was trained on 500. Comparisons between two such runs are still internally
# consistent, but the probe distribution is not the training distribution.
TRAIN_PAIRS="${4:-500}"

if [ -z "${RUN}" ] || [ -z "${MAP}" ]; then
    echo "ERROR: pass a run and its map, e.g."
    echo "  sbatch $0 sunny-galaxy-68 24x24_empty"
    exit 1
fi

# Accept a bare run name as well as a path.
case "${RUN}" in
    runs/*) RUN_DIR="${RUN}" ;;
    SimpleGrid-v0*) RUN_DIR="runs/${RUN}" ;;
    *) RUN_DIR="runs/SimpleGrid-v0__gps_simplegrid_levels__${RUN}" ;;
esac

if [ ! -d "${RUN_DIR}" ]; then
    echo "ERROR: ${RUN_DIR} does not exist. Available:"
    ls runs/ 2>/dev/null | tail -20
    exit 1
fi

echo "=== K1: run_dir=${RUN_DIR}  map=${MAP}  n_samples=${N_SAMPLES}  train_pairs=${TRAIN_PAIRS} ==="

poetry run python check_z_influence.py \
    --run_dir "${RUN_DIR}" \
    --cuda \
    --obstacle_map "${MAP}" \
    --train_dataset_size "${TRAIN_PAIRS}" \
    --n_samples "${N_SAMPLES}"
