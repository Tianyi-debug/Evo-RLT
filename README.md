# Evo-RLT

Evo-RLT is a LeRobot wrapper package for RL Token training, chunk-level actor-critic training,
and deployment adapters. It depends on LeRobot but keeps all LeRobot coupling in
`evo_rlt.adapters.lerobot`; the RLT algorithm core under `evo_rlt.core` does not import LeRobot.

## Layout

```text
src/evo_rlt/core                  # torch-only RLT core
src/evo_rlt/adapters/lerobot      # LeRobot/pi0.5/dataset/policy adapters
src/evo_rlt/cli                   # training and cache CLIs
```

## Development smoke checks

```bash
PYTHONPATH=src pytest tests/rlt/test_rl_token.py tests/rlt/test_losses.py
```

For LeRobot policy factory integration, import and call:

```python
from evo_rlt.adapters.lerobot import register
register()
```

## LeRobot Compatibility

The current migration target is the Evo-RL LeRobot 0.4.4 fork at commit
`95360c66eff2c8adaf8bc51c892f4f0b6ed5ff86`. Upstream LeRobot integration is intentionally isolated
behind `evo_rlt.adapters.lerobot` so future LeRobot updates can be handled as adapter updates.
