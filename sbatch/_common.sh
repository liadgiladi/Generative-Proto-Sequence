#!/bin/bash
# Shared setup + timing harness for the runtime experiments (E1..E5).
#
# Each experiment sources this, then calls `run_experiment` with the flags that are
# specific to it. Everything else is held constant so that the difference in
# WALL_CLOCK_SECONDS between two experiments is attributable to the one flag that changed.
#
# Experiments are deliberately short (30k steps, not 1M): long enough to get past
# learning_starts=1000 and well into the epsilon decay, short enough to turn the whole
# matrix around in an afternoon. Compare steps/sec, not absolute totals.

module load anaconda
source activate gps_env

# wandb credentials come from ~/.netrc (`wandb login`, run once on the login node).
# Do NOT hardcode WANDB_API_KEY here -- see the note in run_gps.sbatch.

# Held constant across every experiment in the matrix.
STEPS=30000
COMMON_ARGS=(
    --cuda
    --track
    --obstacle_map 16x16_corridors_v1
    --train_dataset_size 100
    --total_timesteps "${STEPS}"
    --train_decoder_end_to_end
    --no-initialize_decoder_from_pretrained
    # Keep the one unavoidable end-of-training test evaluation cheap: it runs two full
    # passes over the test set and would otherwise swamp the measurement.
    --test_dataset_size 20
)

run_experiment() {
    echo "=== ${EXPERIMENT_NAME} ==="
    echo "args: ${COMMON_ARGS[*]} $*"
    nvidia-smi --query-gpu=name --format=csv,noheader

    START=$(date +%s)
    poetry run python gps_simplegrid_levels.py "${COMMON_ARGS[@]}" "$@"
    STATUS=$?
    END=$(date +%s)

    ELAPSED=$((END - START))
    echo "=== ${EXPERIMENT_NAME} RESULT ==="
    echo "EXIT_STATUS=${STATUS}"
    echo "WALL_CLOCK_SECONDS=${ELAPSED}"
    echo "TOTAL_TIMESTEPS=${STEPS}"
    if [ "${ELAPSED}" -gt 0 ]; then
        echo "STEPS_PER_SECOND=$(awk "BEGIN {printf \"%.2f\", ${STEPS}/${ELAPSED}}")"
        echo "PROJECTED_HOURS_FOR_1M_STEPS=$(awk "BEGIN {printf \"%.2f\", ${ELAPSED}*1000000/${STEPS}/3600}")"
    fi
    return ${STATUS}
}
