#!/bin/bash
# J1 -- Transformer decoder on any map.  Parameterised: the map is an argument.
#
#   sbatch sbatch/j1_gpt_map.sbatch <map> [length_weight] [train_pairs] [total_steps]
#
#   sbatch sbatch/j1_gpt_map.sbatch 16x16_obstacles_v1_25p                     # tuned (0.002)
#   sbatch sbatch/j1_gpt_map.sbatch 16x16_obstacles_v1_25p 0                   # plain, no penalty
#   sbatch sbatch/j1_gpt_map.sbatch 16x16_obstacles_v1_25p 0.002 500 1000000   # 1M, matches the MLP
#
# The paper evaluates 9 environments; this repo ships 7 of them (both 24x24 obstacle maps
# are absent):
#   8x8_empty  16x16_empty  16x16_room_v1  16x16_corridors_v1
#   16x16_obstacles_v1_15p  16x16_obstacles_v1_25p  24x24_empty
#
# TWO CONFIGS PER MAP, and both are needed:
#   weight 0.002 (default) -> our tuned method
#   weight 0               -> no length penalty, which is what the paper's MLP effectively
#                             ran, because its torch.floor bug silently disabled
#                             --sequence_length_weight. Only this one is a like-for-like
#                             decoder comparison against j2.
#
# Reporting only the tuned number against the MLP would credit the Transformer for a dead
# feature we switched on rather than for the architecture.
#
# Everything else matches the best-known config (d6). 100 training pairs, matching the
# paper's protocol -- note that on corridors this was badly data-starved (0.725 at 100
# pairs, 0.99 at 500), so expect every map here to understate what the setup can reach.
#
# CAVEAT worth checking before trusting 24x24 or 8x8: max_level defaults to 14 and
# n_actions_in_seq to 10. Those were sensible for 16x16. On 24x24 optimal paths are longer,
# so a 10-action plan covers proportionally less ground; on 8x8 the level range may clip.
# Compare "mean num decoder generations" and "agent step ratio" across maps -- if 24x24
# needs far more generations per episode, the plan length is the limit, not the decoder.
#
#SBATCH --partition=main
# 1 day, not 12h: a 500k run takes 3-7h depending on the map, so a 1M run on the slower
# maps (obstacles_15p was 5.45h at 500k) would be killed at a 12h wall mid-training.
#SBATCH --time=1-00:00:00
#SBATCH --job-name=j1_gpt_map
#SBATCH --output=sbatch/logs/j1_gpt_map-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

MAP="$1"
LENGTH_WEIGHT="${2:-0.002}"
# 500 pairs, not the paper's 100. At 100 both decoders are data-starved (0.725 vs 0.794 on
# corridors) and the comparison measures starvation; at 500 both sit at the success ceiling
# (0.99 vs 0.982) and the difference is purely sequence-level efficiency -- 3.36 vs 13.34
# decoder generations per episode, 0.936 vs 0.573 agent step ratio. That is the paper's
# actual claim, and it only becomes visible once accuracy saturates.
TRAIN_PAIRS="${3:-500}"
# 500k was validated on corridors, where validation peaks around 200k. It does NOT transfer:
# on room_v1 and obstacles_25p the best validation landed at step 499,999 -- still improving
# when the budget ran out -- and obstacles_15p at 470,000. Those three need 1M to be
# comparable against the MLP, which gets 1M everywhere. Check "best val mean global step" in
# the log: within ~10% of the budget means the run was cut short and its score is a floor.
TOTAL_STEPS="${4:-500000}"

if [ -z "${MAP}" ]; then
    echo "ERROR: pass the obstacle map, e.g."
    echo "  sbatch $0 16x16_obstacles_v1_25p"
    echo "  sbatch $0 16x16_obstacles_v1_25p 0        # no length penalty"
    echo "  sbatch $0 16x16_obstacles_v1_25p 0.002 100  # paper's 100-pair protocol"
    exit 1
fi

# The length-band flags must be omitted entirely when the weight is 0: passing them with
# the penalty disabled is harmless but misleading in the logged config.
LENGTH_ARGS=()
if [ "${LENGTH_WEIGHT}" != "0" ]; then
    LENGTH_ARGS=(--sequence_length_weight "${LENGTH_WEIGHT}"
                 --min_lengths_per_head 4 --max_lengths_per_head 10)
fi

echo "=== J1: map=${MAP}  length_weight=${LENGTH_WEIGHT}  train_pairs=${TRAIN_PAIRS}  total_steps=${TOTAL_STEPS} ==="

START=$(date +%s)

poetry run python gps_simplegrid_levels.py \
    --cuda \
    --track \
    --no-torch_deterministic \
    --obstacle_map "${MAP}" \
    --train_dataset_size "${TRAIN_PAIRS}" \
    --train_eval_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --train_decoder_end_to_end \
    --no-initialize_decoder_from_pretrained \
    --decoder_use_cuda_graph \
    --eval_test_dataset_during_training_freq -1 \
    --total_timesteps "${TOTAL_STEPS}" \
    --total_steps_e 15000 \
    --n_proto_plan_candidates 1 \
    --min_qf_value -20.0 \
    --max_qf_value 1.0 \
    "${LENGTH_ARGS[@]}" \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
