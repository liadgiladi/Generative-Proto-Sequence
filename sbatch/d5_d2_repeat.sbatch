#!/bin/bash
# D5 -- second seed of D2 (A3 + soft token feedback), the current best config.
#
# D2 (occult-horseman-28) scored 0.691, the best result so far. Its hypothesis held in a
# satisfying way:
#
#   soft feedback ALONE (B2, 2 seeds)   0.479, 0.481   -> worse than A3
#   A3 ALONE        (3 seeds)           0.582, 0.611, 0.665
#   A3 + soft feedback (D2, 1 seed)     0.691          -> best
#
# which is what D2 was built to test: gradient flowing through the decode chain has almost
# nothing to work with when plans are length 1 (check_z_influence measured mean_len = 1.01
# on the baseline), and A3 is precisely the change that makes them long. The two
# mechanisms appear to compose rather than overlap.
#
# But 0.691 is ONE seed. Three A3 seeds give std ~= 0.042, so +0.072 over the A3 mean of
# 0.619 is only ~1.7 sigma, and it sits just 0.026 above A3's best seed. This run decides
# whether that is real.
#
#   >= 0.66 again  -> A3 + soft feedback is the new best config at ~0.68; gap to the MLP's
#                     0.794 narrows to ~0.11. Promote it to the baseline and stack from there.
#   ~0.62          -> D2 was a draw from the top of the A3 distribution; soft feedback adds
#                     nothing on top of A3 and the lever is spent.
#
# Config is identical to d2_a3_soft_feedback.sbatch. --no-torch_deterministic makes this an
# independent sample without changing --seed.
#
# NOTE on reading these runs: D3 (A3 + actor_learning_rate 3e-4) halved the actor embedding
# cosine almost exactly as much as D2 did (0.375 vs 0.337, against a 0.641 control) and
# scored 0.445 -- 0.17 WORSE than the A3 mean. So actor/embedding_cosine is a mechanism
# signal, not a performance proxy: you can move it and lose. Judge on test success.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=d5_d2_repeat
#SBATCH --output=sbatch/logs/d5_d2_repeat-%J.out
#SBATCH --gpus=gtx_1080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

module load anaconda
source activate gps_env

START=$(date +%s)

poetry run python gps_simplegrid_levels.py \
    --cuda \
    --track \
    --no-torch_deterministic \
    --obstacle_map 16x16_corridors_v1 \
    --train_dataset_size 100 \
    --val_dataset_size 100 \
    --test_dataset_size 1000 \
    --train_decoder_end_to_end \
    --no-initialize_decoder_from_pretrained \
    --decoder_use_cuda_graph \
    --eval_test_dataset_during_training_freq -1 \
    --total_timesteps 500000 \
    --total_steps_e 15000 \
    --n_proto_plan_candidates 1 \
    --min_qf_value -20.0 \
    --max_qf_value 1.0 \
    --sequence_length_weight 0.0005 \
    --min_lengths_per_head 4 \
    --max_lengths_per_head 10 \
    --decoder_soft_token_feedback \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
