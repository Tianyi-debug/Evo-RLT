<h1 align="center">Evo-RLT</h1>

<p align="center">
  <a href="https://github.com/MINT-SJTU/Evo-RL"><img alt="parent project" src="https://img.shields.io/badge/Parent-Evo--RL-0ea5e9"/></a>
  <a href="https://github.com/huggingface/lerobot"><img alt="lerobot version" src="https://img.shields.io/badge/LeRobot-0.4.4-f59e0b"/></a>
  <a href="#relationship-to-evo-rl"><img alt="rlt branch" src="https://img.shields.io/badge/Evo--RL-rlt%20branch-6366f1"/></a>
  <a href="#citation"><img alt="paper coming soon" src="https://img.shields.io/static/v1?label=Paper&message=Coming%20Soon&color=9ca3af"/></a>
  <a href="#model--dataset"><img alt="model and dataset coming soon" src="https://img.shields.io/static/v1?label=Model%20%2F%20Dataset&message=Coming%20Soon&color=9ca3af"/></a>
  <a href="./LICENSE"><img alt="license" src="https://img.shields.io/badge/License-Apache--2.0-ef4444"/></a>
</p>

<p align="center"><strong>SJTU-MINT</strong></p>

<p align="center">
  <strong>RLT is the RL-token branch of Evo-RL: a LeRobot-based wrapper for VLA finetuning, RL-token learning, transition-cache generation, actor-critic training, and real-robot deployment.</strong>
</p>

<p align="center"><strong>Real-Robot Rollout Video</strong></p>

<p align="center">
  <a href="./website/assets/videos/rlt_rollout.mp4">
    <img alt="RLT rollout video placeholder" src="https://placehold.co/1280x720/111827/e5e7eb?text=RLT+Rollout+Video+Placeholder" width="96%"/>
  </a>
</p>

<!--
Replace the placeholder above with the final video asset once it is ready.
Recommended path: website/assets/videos/rlt_rollout.mp4
-->

## Overview

Evo-RLT packages the RLT work that was originally developed inside the Evo-RL codebase into a standalone repository. It keeps LeRobot as a strong dependency, while isolating all LeRobot-specific integration under `evo_rlt.adapters.lerobot`. The algorithmic core under `evo_rlt.core` stays torch-only, so future LeRobot updates can be absorbed at the adapter layer instead of being mixed into the algorithm implementation.

The project currently supports:

- VLA finetuning on LeRobot datasets.
- RL-token training on top of pi0.5 prefix hidden states.
- Transition-cache generation for chunk-level offline RL.
- Chunk actor-critic training for RLT policies.
- Real-robot recording and deployment wrappers for VLA, RLT, and human-in-the-loop collection.

## Relationship to Evo-RL

This repository is an SJTU-MINT project and exists as the standalone RLT sub-repository of Evo-RL. Conceptually, it corresponds to the `rlt` branch of Evo-RL: the original monorepo implementation was extracted, cleaned, and reorganized so that RLT can evolve independently while still depending on LeRobot and remaining compatible with Evo-RL's real-world robot workflow.

The intended layering is:

```text
Evo-RL real-world workflow
`-- Evo-RLT package
    |-- evo_rlt.core                  # torch-only RLT algorithm modules
    |-- evo_rlt.adapters.lerobot      # LeRobot policy, dataset, processor, and record adapters
    `-- evo_rlt.cli                   # training, cache, and deployment command-line tools
```

## Highlights

- **LeRobot wrapper, not a forked code dump:** RLT-specific code lives in `evo_rlt`, and LeRobot integration is registered at runtime.
- **Deploy-aligned training artifacts:** RL-token and actor-critic policies are saved in policy-style directories that can be consumed by deployment scripts.
- **Cache-first offline RL:** expensive VLA/RL-token encoding can be precomputed into transition caches before actor-critic training.
- **Robot-facing scripts:** the record wrapper supports VLA-only collection, pedal outcome labels, RLT deployment, RTC defaults, and human-in-the-loop workflows.

## Quick Start

```bash
git clone https://github.com/Shiki42/evo-rlt.git
cd evo-rlt

conda create -y -n evo-rl python=3.10
conda activate evo-rl

pip install -e .
```

For the current migration target, use the Evo-RL LeRobot 0.4.4 fork at commit `95360c66eff2c8adaf8bc51c892f4f0b6ed5ff86` on `PYTHONPATH` or install it in the same environment.

```bash
export PYTHONPATH=/path/to/evo-rlt/src:/path/to/evo-rl/src
```

## Runtime Registration

Evo-RLT keeps policy registration out of LeRobot source files. Before using LeRobot factory helpers with RLT policy types, register the adapter once:

```python
from evo_rlt.adapters.lerobot import register

register()
```

Registered policy types:

```text
rlt_token    # RL-token reconstruction policy
rlt_ac       # chunk actor-critic policy
rlt          # deployment policy wrapper
```

## Training Pipeline

The typical RLT workflow has four stages.

### 1. Finetune VLA

Use LeRobot's training entrypoint to finetune a pi0.5 VLA checkpoint on a LeRobot dataset.

```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=<HF_ORG>/<DATASET> \
  --dataset.root=<LOCAL_DATASET_ROOT> \
  --policy.path=<BASE_PI05_CHECKPOINT_DIR> \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=16 \
  --steps=30000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --output_dir=outputs/vla_ft \
  --job_name=vla_ft
```

### 2. Train RL Token

```bash
python -c 'from evo_rlt.adapters.lerobot import register; register(); from lerobot.scripts.lerobot_train import main; main()' \
  --dataset.repo_id=<HF_ORG>/<DATASET> \
  --dataset.root=<LOCAL_DATASET_ROOT> \
  --policy.type=rlt_token \
  --policy.vla_pretrained_path=outputs/vla_ft/checkpoints/last/pretrained_model \
  --policy.vla_dtype=bfloat16 \
  --policy.rl_token_num_rl_tokens=1 \
  --policy.token_pool_size=0 \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=10000 \
  --save_freq=2000 \
  --eval_freq=0 \
  --output_dir=outputs/rl_token \
  --job_name=rl_token
```

### 3. Build Transition Cache

```bash
evo-rlt-build-transition-cache-v2 \
  --demo-dataset-repo-id <HF_ORG>/<DATASET> \
  --demo-dataset-root <LOCAL_DATASET_ROOT> \
  --rl-token-policy-path outputs/rl_token/checkpoints/last/pretrained_model \
  --vla-pretrained-path outputs/vla_ft/checkpoints/last/pretrained_model \
  --output-dir outputs/cache \
  --task-instruction "<TASK>" \
  --chunk-length 10 \
  --frame-stride 2 \
  --batch-size 8 \
  --num-workers 2 \
  --train-ratio 0.9 \
  --device cuda
```

### 4. Train Chunk Actor-Critic

```bash
python -c 'from evo_rlt.adapters.lerobot import register; register(); from lerobot.scripts.lerobot_train import main; main()' \
  --dataset.type=rlt_chunk_transition \
  --dataset.repo_id=outputs/cache \
  --policy.type=rlt_ac \
  --policy.vla_pretrained_path=outputs/vla_ft/checkpoints/last/pretrained_model \
  --policy.rl_token_pretrained_path=outputs/rl_token/checkpoints/last/pretrained_model \
  --policy.vla_dtype=bfloat16 \
  --policy.rl_token_num_rl_tokens=1 \
  --policy.chunk_length=10 \
  --policy.chunk_exec_steps=25 \
  --policy.phase_mode=always_rl \
  --policy.device=cuda \
  --batch_size=256 \
  --steps=50000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --output_dir=outputs/ac \
  --job_name=rlt_ac
```

## Real-Robot Recording and Deployment

`evo-rlt-record` wraps the LeRobot recording stack and injects RLT-specific policy, RTC, and human-label behavior without modifying LeRobot source files.

Example VLA-only full-process recording with pedal outcome labels:

```bash
PYTHONPATH=src:/path/to/evo-rl/src HF_HUB_OFFLINE=1 evo-rlt-record full \
  --initial-source vla \
  --policy-path <AC_OR_VLA_POLICY_PATH> \
  --vla-path <BASE_OR_FINETUNED_VLA_PT> \
  --phase-mode always_vla \
  --chunk-exec-steps 25 \
  --pedal-outcome \
  --double-tap-window-s 0.6 \
  --num-episodes 5 \
  --episode-time-s 3000 \
  --reset-time-s 0 \
  --fps 30 \
  --vcodec h264 \
  --dataset-tag vla_full_pedal \
  --no-teleop
```

Pedal semantics in this mode:

```text
single tap    success, end current episode, start next episode
double tap    failure, end current episode, start next episode
```

## Repository Layout

```text
src/evo_rlt/core                  # algorithm core, torch-only
src/evo_rlt/adapters/lerobot      # LeRobot/pi0.5/dataset/policy/record adapters
src/evo_rlt/cli                   # training and cache CLIs
tests/rlt                         # focused RLT unit and integration tests
```

## Development Checks

```bash
PYTHONPATH=src pytest -q tests/rlt
PYTHONPATH=src python -m compileall -q src/evo_rlt tests/rlt
```

## Model & Dataset

Models, datasets, and demonstration videos will be linked here after public release.

## Citation

Citation information will be added with the paper release.

## License

This project follows the Evo-RL release convention and is distributed under the Apache-2.0 license.
