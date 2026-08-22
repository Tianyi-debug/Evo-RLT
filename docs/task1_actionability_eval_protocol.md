# Task1 direct-Q matched real-robot evaluation

This protocol evaluates the strictly matched Task1 actors `Q0`, `Q=0.05`, and
`Q=0.25`. Only fully autonomous episodes count toward policy utility. Do not
rescue an episode and later count it as autonomous success.

## Fixed conditions

- setup: `/home/zty/catkin_ws/src/Evo-RLT/configs/so101_two_cam_manifest.json`
- locked pose: `/home/zty/catkin_ws/src/Evo-RLT/configs/so101_expert_initial_pose.json`
- task: `Put the clip into the box`
- deterministic actor mean, `always_rl`, chunk execution 25
- 60-second episode timeout, 10-second environment reset window
- `front` and `top` aliases remain those in the setup manifest
- no teleoperation during scored trials

Before every block, verify the object/box pose, lighting, both camera feeds, and
the locked reset pose. A hardware/camera failure is an invalid trial and must be
rerun; it is not a policy failure.

## Shell setup and block command

Paste this once in the robot shell:

```bash
cd /home/zty/catkin_ws/src/Evo-RLT
conda activate evo-rlt

export EVO_RLT_PI05_DEVICE_DIRECT_LOAD=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK1_VLA=/home/zty/catkin_ws/src/Evo-RLT/outputs/task1_v4/pi05_vla_ft_ep5_g8/checkpoints/003300/pretrained_model
TASK1_RL_TOKEN=/home/zty/catkin_ws/src/Evo-RLT/outputs/task1_v4/rl_token_ep10_fp32_lr1e5_g6/checkpoints/022000/pretrained_model
TASK1_Q0=/home/zty/catkin_ws/src/Evo-RLT/outputs/task1_v4/actor_refine_matched_q0_projected000228_h1_t1_projected_actor5e6_b16_updates228_seed1000/checkpoints/000228/pretrained_model
TASK1_Q005=/home/zty/catkin_ws/src/Evo-RLT/outputs/task1_v4/actor_refine_matched_fixedq005_projected000228_h1_t1_projected_actor5e6_b16_updates228_seed1000/checkpoints/000228/pretrained_model
TASK1_Q025=/home/zty/catkin_ws/src/Evo-RLT/outputs/task1_v4/actor_refine_matched_fixedq025_projected000228_h1_t1_projected_actor5e6_b16_updates228_seed1000/checkpoints/000228/pretrained_model

run_task1_block () {
  local policy_path="$1"
  local dataset_tag="$2"
  local episode_count="$3"
  evo-rlt-record full \
    --initial-source vla \
    --setup-json /home/zty/catkin_ws/src/Evo-RLT/configs/so101_two_cam_manifest.json \
    --policy-path "$policy_path" \
    --vla-path "$TASK1_VLA" \
    --rl-token-path "$TASK1_RL_TOKEN" \
    --task "Put the clip into the box" \
    --phase-mode always_rl \
    --chunk-exec-steps 25 \
    --deterministic \
    --no-teleop \
    --rtc \
    --rtc-action-queue-size-to-get-new-actions 20 \
    --num-episodes "$episode_count" \
    --episode-time-s 60 \
    --reset-time-s 10 \
    --fps 30 \
    --vcodec h264 \
    --default-episode-success failure \
    --reset-pose-path /home/zty/catkin_ws/src/Evo-RLT/configs/so101_expert_initial_pose.json \
    --no-reset-pose-recapture \
    --dataset-tag "$dataset_tag"
}
```

Mark autonomous success explicitly with the normal success key. Timeouts and
unmarked episodes remain failure through `--default-episode-success failure`.

## Screening schedule: 12 trials per method

Run one line at a time. Each row is one four-episode block. The Latin-square
order gives every method one early, middle, and late block.

```bash
run_task1_block "$TASK1_Q0"   task1_actionability_screen_r1_q0   4
run_task1_block "$TASK1_Q005" task1_actionability_screen_r1_q005 4
run_task1_block "$TASK1_Q025" task1_actionability_screen_r1_q025 4

run_task1_block "$TASK1_Q005" task1_actionability_screen_r2_q005 4
run_task1_block "$TASK1_Q025" task1_actionability_screen_r2_q025 4
run_task1_block "$TASK1_Q0"   task1_actionability_screen_r2_q0   4

run_task1_block "$TASK1_Q025" task1_actionability_screen_r3_q025 4
run_task1_block "$TASK1_Q0"   task1_actionability_screen_r3_q0   4
run_task1_block "$TASK1_Q005" task1_actionability_screen_r3_q005 4
```

Do not use the screening result as the final paper estimate. It is a smoke test
for gross regressions and hardware/protocol problems.

## Full extension: add 36 trials per method

If the screening run is operationally valid, these additional blocks bring the
total to 48 trials per method. They are separate datasets by design.

```bash
run_task1_block "$TASK1_Q0"   task1_actionability_full_r4_q0   12
run_task1_block "$TASK1_Q005" task1_actionability_full_r4_q005 12
run_task1_block "$TASK1_Q025" task1_actionability_full_r4_q025 12

run_task1_block "$TASK1_Q005" task1_actionability_full_r5_q005 12
run_task1_block "$TASK1_Q025" task1_actionability_full_r5_q025 12
run_task1_block "$TASK1_Q0"   task1_actionability_full_r5_q0   12

run_task1_block "$TASK1_Q025" task1_actionability_full_r6_q025 12
run_task1_block "$TASK1_Q0"   task1_actionability_full_r6_q0   12
run_task1_block "$TASK1_Q005" task1_actionability_full_r6_q005 12
```

## Trial log template

Keep one CSV row per attempted episode, including invalid attempts:

```text
date_time,round,block_position,method,dataset_root,episode_index,autonomous_success,intervention_used,valid_trial,invalid_reason,object_pose_id,camera_check,reset_pose_check,notes
```

The primary comparison uses only `valid_trial=1` and
`intervention_used=0`. Report success counts and binomial confidence intervals
for all three methods; do not replace these matched trials with historical
results collected on another day or pose.
