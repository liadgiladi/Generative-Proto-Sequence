#!/bin/bash
# G4 -- multi-head proto-plans, with the two guards that were missing last time.
#
# The one architectural idea in this codebase that has never had a fair run.
#
# WHY IT FAILED BEFORE, AND WHY THAT WAS NOT A VERDICT
#
# A1/A2/A3 ran --n_proto_plan_candidates 5 and all three peaked at step 60,000, then decayed
# for 250k steps before dying on NaN Q-values in select_action. The cause (sec. 9.1): the Bellman
# target takes argmax over the candidates --
#
#   qf_next_target = qf_next_target.view(batch, n_candidates)
#   best = torch.argmax(qf_next_target, dim=1)          # maximisation bias
#
# -- which is a no-op at 1 candidate and compounds overestimation at 5. Q diverged. Those
# runs measured instability, not the mechanism, so multi-head is UNTESTED, not refuted.
#
# Two things have changed since:
#   --min_qf_value -20.0   activates the Q clipping that was dead config until sec. 9.2
#                          (max_qf_value alone does nothing -- the clip is gated on
#                          min_qf_value being non-None, so it had never run in ANY experiment
#                          in this repo, including the MLP baseline)
#   --actor_grad_clip 1.0  added after two separate changes destabilised the actor: a larger
#                          decoder (NaN in cnn.0.weight at step 159k) and actor_learning_rate
#                          3e-4 (0.445). Typical actor gradient norms are 0.2-0.4, so 1.0
#                          only bites on spikes.
#
# WHY IT IS WORTH A RUN
#
# With one candidate the method's central mechanism is inert: there is nothing for the critic
# to select between, so eval_candidates_with_critic, select_action's sample_from_candidates
# branch and the per-head softmax weighting in the actor loss all reduce to identity. The
# Config ships five-element min/max_lengths_per_head defaults, which is what 5 heads was
# designed for. And it is flag-only.
#
# NOTE the length bands revert to the 5-element defaults [1,2,4,6,8] / [2,4,6,8,10], because
# sequence_length_weight asserts len(min_lengths_per_head) == n_proto_plan_candidates. That
# also means the bands differ from the 4-10 used in the best config -- per-head bands push
# the heads to different lengths, which is the point of having several.
#
# READING IT
#
#   survives past 60k and > 0.74  -> multi-head works once stabilised; the mechanism was
#                                    right and only the maximisation bias was in the way
#   survives, ~0.72               -> stable but no better; one candidate is sufficient for
#                                    this task and the extra machinery buys nothing
#   peaks ~60k again / NaN        -> the clipping does not fix it; the instability is the
#                                    argmax target itself and would need a double-Q or
#                                    clipped-target change, which is beyond a flag
#
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --job-name=g4_multihead
#SBATCH --output=sbatch/logs/g4_multihead_clipped-%J.out
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
    --n_proto_plan_candidates 5 \
    --min_qf_value -20.0 \
    --max_qf_value 1.0 \
    --actor_grad_clip 1.0 \
    --sequence_length_weight 0.002 \
    --slurm_job_id "${SLURM_JOB_ID}"

END=$(date +%s)
echo "WALL_CLOCK_SECONDS=$((END - START))"
echo "WALL_CLOCK_HOURS=$(awk "BEGIN {printf \"%.2f\", ($END - $START)/3600}")"
