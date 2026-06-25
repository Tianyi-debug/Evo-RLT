<h1 align="center">Evo-RLT</h1>

<p align="center">
  <a href="https://github.com/huggingface/lerobot"><img alt="lerobot version" src="https://img.shields.io/badge/LeRobot-0.4.4-f59e0b"/></a>
  <a href="#citation"><img alt="paper coming soon" src="https://img.shields.io/static/v1?label=Paper&message=Coming%20Soon&color=9ca3af"/></a>
  <a href="#model-dataset"><img alt="model and dataset available" src="https://img.shields.io/static/v1?label=Model%20%2F%20Dataset&message=Available&color=22c55e"/></a>
  <a href="./LICENSE"><img alt="license" src="https://img.shields.io/badge/License-Apache--2.0-ef4444"/></a>
</p>

<p align="center"><strong>SJTU-MINT</strong></p>

<p align="center">
  <strong>An independent LeRobot-based reproduction of RLT for the pi paper, covering RL-token learning, transition-cache generation, actor-critic training, and real-robot rollout.</strong>
</p>

<p align="center"><strong>Real-Robot Rollout Demo</strong></p>

<p align="center">
  <img alt="RLT real-robot rollout demo" src="./website/assets/images/rlt_rollout.gif" width="96%"/>
</p>

## 🎯 Evo-RLT Focus

- **Standalone RLT reproduction:** this repository presents RLT as an independent LeRobot-based reproduction for the pi paper, rather than as a branch or submodule of another project.
- **Open training path:** the code covers VLA finetuning, RL-token learning, transition-cache generation, and chunk actor-critic training.
- **Real-robot deployment path:** the recording wrapper supports VLA/RLT rollout, RTC defaults, pedal labels, and human-in-the-loop collection.

## 📰 News

- **[2026-06-26]** Added a higher-quality rollout GIF and affiliations.
- **[2026-06-26]** Added training dataset and checkpoint links, and folded runtime registration into installation.
- **[2026-06-25]** Added a GitHub-rendered rollout GIF for the repository homepage.
- **[2026-06-25]** Refreshed the README to describe Evo-RLT as a standalone RLT reproduction and added hardware setup differences from Evo-RL.

## 🧭 Table of Contents

| Getting Started | Training Pipeline | Project Info |
| -------------------------------------- | -------------------------------------------- | ------------------------------------------- |
| [⚡ Quick Start](#quick-start) | [🧪 Training Pipeline](#training-pipeline) | [🤗 Model & Dataset](#model-dataset) |
| [1) Installation](#installation) | [3) Finetune VLA](#finetune-vla) | [🗂️ Repository Layout](#repository-layout) |
| [2) Hardware Setup](#hardware-setup) | [4) Train RL Token](#train-rl-token) | [✅ Development Checks](#development-checks) |
| [🤖 Real-Robot Recording and Deployment](#real-robot-recording-and-deployment) | [5) Build Transition Cache](#build-transition-cache) | [🏫 Affiliations](#affiliations) |
| | [6) Train Chunk Actor-Critic](#train-chunk-actor-critic) | [📖 Citation](#citation) / [📄 License](#license) |

<a id="quick-start"></a>

## ⚡ Quick Start

<a id="installation"></a>

### 1) Installation

```bash
git clone https://github.com/Shiki42/evo-rlt.git
cd evo-rlt

conda create -y -n evo-rlt python=3.10
conda activate evo-rlt

pip install -e ".[lerobot]"
```

If you are using a local LeRobot checkout instead of the package dependency, put this repository and that checkout on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/evo-rlt/src:/path/to/lerobot/src
```

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

<a id="hardware-setup"></a>

### 2) Hardware Setup

Use the [Evo-RL hardware setup](https://github.com/MINT-SJTU/Evo-RL#2-hardware-setup) for the shared robot bring-up steps: SO-series assembly, stable serial/camera paths, camera validation, PiPER/PiPER-X CAN setup, and basic teleoperation checks.

This repository only differs at the recording/deployment configuration layer:

- `evo-rlt-record` reads a setup manifest from `--setup-json`, or from `~/.roboclaw/workspace/embodied/manifest.json` when the flag is omitted.
- Arm entries point to per-device `calibration_dir` folders. The wrapper looks for `<calibration_dir>/<folder-name>.json`, then stages those files into temporary LeRobot-compatible names at runtime.
- Follower calibrations are staged as `bimanual_left.json` and `bimanual_right.json` under a temporary robot calibration directory.
- Leader calibrations are staged as `bimanual_leader_left.json` and `bimanual_leader_right.json` under a temporary teleop calibration directory.
- Dataset paths are created under `<datasets.root>/<MMDD>_<dataset-tag>/<prefix>_<HHMMSS>`. If `datasets.root` is omitted, the default is `~/.roboclaw/workspace/embodied/datasets`.

Example setup manifest:

```json
{
  "datasets": {"root": "/path/to/lerobot_datasets"},
  "arms": [
    {
      "alias": "left_follower",
      "type": "follower",
      "port": "/dev/serial/by-id/<left-follower>",
      "calibration_dir": "/path/to/calibration/<left-follower-serial>"
    },
    {
      "alias": "right_follower",
      "type": "follower",
      "port": "/dev/serial/by-id/<right-follower>",
      "calibration_dir": "/path/to/calibration/<right-follower-serial>"
    },
    {
      "alias": "left_leader",
      "type": "leader",
      "port": "/dev/serial/by-id/<left-leader>",
      "calibration_dir": "/path/to/calibration/<left-leader-serial>"
    },
    {
      "alias": "right_leader",
      "type": "leader",
      "port": "/dev/serial/by-id/<right-leader>",
      "calibration_dir": "/path/to/calibration/<right-leader-serial>"
    }
  ],
  "cameras": [
    {
      "alias": "left_wrist",
      "port": "/dev/v4l/by-path/<left-wrist>",
      "width": 640,
      "height": 480,
      "fps": 30,
      "fourcc": "MJPG"
    },
    {
      "alias": "right_wrist",
      "port": "/dev/v4l/by-path/<right-wrist>",
      "width": 640,
      "height": 480,
      "fps": 30,
      "fourcc": "MJPG"
    },
    {
      "alias": "right_front",
      "port": "/dev/v4l/by-path/<right-front>",
      "width": 640,
      "height": 480,
      "fps": 30,
      "fourcc": "MJPG"
    }
  ]
}
```

<a id="training-pipeline"></a>

## 🧪 Training Pipeline

The typical RLT workflow has four stages.

<a id="finetune-vla"></a>

### 3) Finetune VLA

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

<a id="train-rl-token"></a>

### 4) Train RL Token

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

<a id="build-transition-cache"></a>

### 5) Build Transition Cache

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

<a id="train-chunk-actor-critic"></a>

### 6) Train Chunk Actor-Critic

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

<a id="real-robot-recording-and-deployment"></a>

## 🤖 Real-Robot Recording and Deployment

Set up the environment before running robot commands:

```bash
cd /path/to/evo-rlt
source ~/miniconda3/etc/profile.d/conda.sh
conda activate evo-rlt
export PYTHONPATH=/path/to/evo-rlt/src:/path/to/lerobot/src
export HF_HUB_OFFLINE=1
```

Default VLA-RLT-VLA real-robot collection:

```bash
evo-rlt-record collect \
  --setup-json /path/to/robot_manifest.json \
  --policy-path /path/to/rlt_ac_policy \
  --vla-path /path/to/pi05_vla_checkpoint_or_dir \
  --rl-token-path /path/to/rl_token_policy \
  --dataset-tag vla_rlt_vla_test \
  --rlt-toggle-key r \
  --teleop-toggle-key space \
  --episode-outcome-key e
```

The same collection entrypoint is exposed as `evo-rlt-collect-default` after reinstalling package entry points, but checkpoint and setup paths still need to be supplied by the caller.

Default collection controls:

```text
r              toggle RLT critical phase
space          toggle teleop intervention; pressing again exits teleop
e              save the episode as success after the double-tap window
e+e            save the episode as failure
left arrow     rerecord the current episode
Esc            stop data collection
```

Validated RTC defaults for this collection mode:

```text
RLT RTC execution horizon: 10
VLA RTC execution horizon: 25
RTC action queue refill threshold: 30
```

VLA-only full-process recording with pedal outcome labels:

```bash
evo-rlt-record full \
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

<a id="repository-layout"></a>

## 🗂️ Repository Layout

```text
src/evo_rlt/core                  # algorithm core, torch-only
src/evo_rlt/adapters/lerobot      # LeRobot/pi0.5/dataset/policy/record adapters
src/evo_rlt/cli                   # training and cache CLIs
tests/rlt                         # focused RLT unit and integration tests
```

<a id="development-checks"></a>

## ✅ Development Checks

```bash
PYTHONPATH=src pytest -q tests/rlt
PYTHONPATH=src python -m compileall -q src/evo_rlt tests/rlt
```

<a id="model-dataset"></a>

## 🤗 Model & Dataset

- Training dataset: [Elvinky/bi-so101-insert-screw-562ep](https://huggingface.co/datasets/Elvinky/bi-so101-insert-screw-562ep).
- Checkpoint repo: [Shiki42/pi05_screw_c_mix_cont15k_fp16](https://huggingface.co/Shiki42/pi05_screw_c_mix_cont15k_fp16/tree/main).
- Rollout demo: displayed above from `website/assets/images/rlt_rollout.gif`.

<a id="affiliations"></a>

## 🏫 Affiliations

<p align="center">
  <img alt="SJTU community visual" src="./website/assets/images/sjtu.png" height="68"/>
  <img alt="EvoMind" src="./website/assets/images/evomind1.png" height="60"/>
</p>

<a id="citation"></a>

## 📖 Citation

Citation information will be added with the paper release.

<a id="license"></a>

## 📄 License

Apache-2.0. See [LICENSE](./LICENSE).
