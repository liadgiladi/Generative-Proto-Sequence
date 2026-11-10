#!/bin/bash
# B2 -- Soft token feedback: let gradient reach z through the whole decode chain.
#
# A1-A3 push the actor directly (more heads, diversity, length bands). B2 targets why the
# actor stopped learning in the first place.
#
# Evidence (TRANSFORMER_INTEGRATION.md §8.6). Actor embedding cosine similarity over a
# full run -- lower means more state-dependent, which is what we want:
#
#   MLP decoder (0.794 test):   1.0  -> 0.507     the actor started fully collapsed and learned
#   GPT decoder (0.432 test):   0.706 -> 0.619    the actor barely moved
#
# The MLP's actor learned; ours did not. That points at the learning signal, not the
# starting point. The actor's gradient arrives from the critic *through the decoder*:
#
#   MLP:  critic -> one Gumbel -> one linear -> z                      (short, strong)
#   GPT:  critic -> Gumbel x10 -> trunk x10 -> z_proj -> z             (long, and the
#         fed-back tokens were detached, so gradient reached z only via the prefix token)
#
# Consistent with actor gradient norms: MLP 0.387 vs GPT 0.197-0.265.
#
# The change feeds the emitted one-hot row into the next step instead of a detached argmax
# index. For a one-hot row, row @ tok_emb.weight == tok_emb(argmax), so **generation is
# numerically unchanged** -- only the backward pass differs.
#
# Risk: a 10-step differentiable chain is less stable than a 1-step one. Both previous
# full runs hit NaN gradients; watch gradient/decoder_norm and gradient/actor_norm. The
# NaN abort now preserves the final test evaluation (§8.1), so an instability still yields
# a number.
#
# Single variable vs the 0.53-val baseline: one proto-plan candidate, no diversity or
# length penalty, soft token feedback the only difference.
#
# Deliberately NOT 5 heads. The multi-head runs (A1/A2/A3) peaked at step 60000 and then
# degraded for 250k steps before dying on NaN Q-values, because the Bellman target takes
# argmax over the candidates -- maximisation bias, which is a no-op at 1 candidate. Testing
# soft feedback on top of that would confound the two.
#
# --min_qf_value is set because the Q-clipping in the training loop is gated on it and has
# therefore never been active in any run: max_qf_value=1.0 was dead config. Episodic
# returns span roughly -20 to +1. Both 1-candidate full runs still hit NaN gradients (steps
# 625476 and 918000), so bounding Q is worthwhile even without extra candidates.
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=b2_soft_feedback
#SBATCH --output=sbatch/logs/b2_soft_feedback-%J.out
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
    --decoder_soft_token_feedback \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
