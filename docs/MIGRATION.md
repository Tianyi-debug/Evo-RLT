# Migration Notes

Source branch: `fork/shuyuan/train_rlt_with_lerobot`
Source commit: `95360c66eff2c8adaf8bc51c892f4f0b6ed5ff86`

This repository is structured as a LeRobot wrapper rather than a LeRobot fork.
RLT core code lives in `evo_rlt.core`; all LeRobot-dependent code lives in
`evo_rlt.adapters.lerobot`.
