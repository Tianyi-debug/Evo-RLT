from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

DEFAULT_SETUP_PATH = Path.home() / ".roboclaw/workspace/embodied/manifest.json"
DEFAULT_DATASET_ROOT = Path.home() / ".roboclaw/workspace/embodied/datasets"


def load_setup_json(path: str | None = None) -> dict[str, Any]:
    setup_path = Path(path).expanduser() if path else DEFAULT_SETUP_PATH
    with open(setup_path) as fh:
        return json.load(fh)


def resolve_dataset_root(setup: dict[str, Any]) -> Path:
    dataset_root = setup.get("datasets", {}).get("root", "")
    if not dataset_root:
        return DEFAULT_DATASET_ROOT
    return Path(dataset_root).expanduser()


def get_sorted_followers(setup: dict[str, Any]) -> list[dict[str, Any]]:
    followers = [arm for arm in setup["arms"] if "follower" in arm["type"]]
    followers.sort(key=lambda arm: 0 if "left" in arm.get("alias", "") else 1)
    return followers


def get_sorted_leaders(setup: dict[str, Any]) -> list[dict[str, Any]]:
    leaders = [arm for arm in setup["arms"] if "leader" in arm["type"]]
    leaders.sort(key=lambda arm: 0 if "left" in arm.get("alias", "") else 1)
    return leaders

log = logging.getLogger(__name__)

CAMERA_RENAME = {"left_wrist": "wrist", "right_wrist": "wrist", "right_front": "front"}
LEFT_CAMERA_ALIASES = {"left_wrist"}
RIGHT_CAMERA_ALIASES = {"right_wrist", "right_front"}
TELEOP_ID = "bimanual_leader"
DEFAULT_SO_VARIANT = "so101"


@dataclass(frozen=True)
class RobotSetup:
    setup: dict[str, Any]
    followers: list[dict[str, Any]]
    leaders: list[dict[str, Any]]
    left_cameras: dict[str, Any]
    right_cameras: dict[str, Any]


@dataclass(frozen=True)
class RunPaths:
    dataset_name: str
    dataset_root: Path
    day_dir: Path
    log_file: Path


def load_robot_setup(setup_json: str | None) -> RobotSetup:
    setup = load_setup_json(setup_json)
    followers = get_sorted_followers(setup)
    leaders = get_sorted_leaders(setup)
    if not followers:
        raise ValueError("Need at least 1 follower arm, got 0")
    left_cameras, right_cameras = build_camera_configs(
        setup.get("cameras", []),
        single_arm=len(followers) == 1,
    )
    return RobotSetup(setup, followers, leaders, left_cameras, right_cameras)


def build_camera_configs(
    cameras: list[dict[str, Any]],
    *,
    single_arm: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    left_cameras: dict[str, Any] = {}
    right_cameras: dict[str, Any] = {}
    for camera in cameras:
        alias = camera["alias"]
        camera_config: dict[str, Any] = {
            "type": "opencv",
            "index_or_path": camera["port"],
            "width": camera.get("width", 640),
            "height": camera.get("height", 480),
            "fps": camera.get("fps", 30),
        }
        if camera.get("fourcc"):
            camera_config["fourcc"] = camera["fourcc"]
        if camera.get("backend"):
            camera_config["backend"] = camera["backend"]
        target_name = CAMERA_RENAME.get(alias, alias)
        if single_arm:
            left_cameras[target_name] = camera_config
        elif alias in LEFT_CAMERA_ALIASES:
            left_cameras[target_name] = camera_config
        elif alias in RIGHT_CAMERA_ALIASES:
            right_cameras[target_name] = camera_config
    return left_cameras, right_cameras


def resolve_run_paths(setup: dict[str, Any], dataset_tag: str, dataset_prefix: str) -> RunPaths:
    now = datetime.now()
    date_folder = f"{now:%m%d}_{dataset_tag}"
    dataset_leaf = f"{dataset_prefix}_{now:%H%M%S}"
    day_dir = resolve_dataset_root(setup) / date_folder
    dataset_root = day_dir / dataset_leaf
    return RunPaths(
        dataset_name=f"local/{dataset_leaf}",
        dataset_root=dataset_root,
        day_dir=day_dir,
        log_file=day_dir / f"{dataset_leaf}.log",
    )


def configure_logging(log_file: Path, log_level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)


def remove_existing_dataset(dataset_root: Path) -> None:
    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)


def stage_arm_calibration(arm: dict[str, Any], dst: Path) -> None:
    cal_dir = Path(arm["calibration_dir"]).expanduser()
    serial = cal_dir.name
    candidate_names = [f"{serial}.json"]
    if arm.get("lerobot_id"):
        candidate_names.append(f"{arm['lerobot_id']}.json")
    for name in dict.fromkeys(candidate_names):
        src = cal_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            log.info("Calibration staged: %s -> %s", src, dst)
            return
    log.warning("Calibration file not found under %s; tried %s", cal_dir, candidate_names)


def _so_variant(arm: dict[str, Any]) -> str:
    for key in ("robot_type", "teleop_type", "model", "variant", "type", "alias"):
        value = str(arm.get(key, "")).lower()
        if "so100" in value:
            return "so100"
        if "so101" in value:
            return "so101"
    return DEFAULT_SO_VARIANT


def _single_follower_type(arm: dict[str, Any]) -> str:
    return f"{_so_variant(arm)}_follower"


def _single_leader_type(arm: dict[str, Any]) -> str:
    return f"{_so_variant(arm)}_leader"


def _single_device_id(arm: dict[str, Any], role: str) -> str:
    if arm.get("lerobot_id"):
        return str(arm["lerobot_id"])
    if role == "follower":
        return _single_follower_type(arm)
    if role == "leader":
        return _single_leader_type(arm)
    raise ValueError(f"Unknown single device role: {role}")


def stage_follower_calibrations(followers: list[dict[str, Any]], cal_dir: str) -> None:
    if len(followers) == 1:
        stage_arm_calibration(
            followers[0],
            Path(cal_dir) / f"{_single_device_id(followers[0], 'follower')}.json",
        )
        return
    for side, arm in (("left", followers[0]), ("right", followers[1])):
        stage_arm_calibration(arm, Path(cal_dir) / f"bimanual_{side}.json")


def build_teleop_argv(leaders: list[dict[str, Any]], no_teleop: bool) -> list[str]:
    if no_teleop:
        log.warning("Teleop disabled by --no-teleop")
        return []
    if not leaders:
        log.warning("Teleop disabled: need at least 1 leader arm, got 0")
        return []
    if len(leaders) == 1:
        leader = leaders[0]
        log.info("Single-arm teleop enabled: port=%s", leader["port"])
        return [
            f"--teleop.type={_single_leader_type(leader)}",
            f"--teleop.port={leader['port']}",
            "--teleop.use_degrees=true",
            f"--teleop.id={_single_device_id(leader, 'leader')}",
        ]
    log.info("Teleop enabled: left=%s, right=%s", leaders[0]["port"], leaders[1]["port"])
    return [
        "--teleop.type=bi_so_leader",
        f"--teleop.left_arm_config.port={leaders[0]['port']}",
        "--teleop.left_arm_config.use_degrees=true",
        f"--teleop.right_arm_config.port={leaders[1]['port']}",
        "--teleop.right_arm_config.use_degrees=true",
        f"--teleop.id={TELEOP_ID}",
    ]


def stage_leader_calibrations(
    leaders: list[dict[str, Any]], teleop_argv: list[str]
) -> TemporaryDirectory[str] | None:
    if not teleop_argv:
        return None
    leader_cal_dir = TemporaryDirectory(prefix="record-leader-cal-")
    if len(leaders) == 1:
        stage_arm_calibration(
            leaders[0],
            Path(leader_cal_dir.name) / f"{_single_device_id(leaders[0], 'leader')}.json",
        )
        teleop_argv.append(f"--teleop.calibration_dir={leader_cal_dir.name}")
        return leader_cal_dir
    for side, arm in (("left", leaders[0]), ("right", leaders[1])):
        stage_arm_calibration(arm, Path(leader_cal_dir.name) / f"{TELEOP_ID}_{side}.json")
    teleop_argv.append(f"--teleop.calibration_dir={leader_cal_dir.name}")
    return leader_cal_dir


def build_robot_argv(
    followers: list[dict[str, Any]],
    left_cameras: dict[str, Any],
    right_cameras: dict[str, Any],
    cal_dir: str,
) -> list[str]:
    if len(followers) == 1:
        follower = followers[0]
        cameras = {**left_cameras, **right_cameras}
        return [
            f"--robot.type={_single_follower_type(follower)}",
            f"--robot.id={_single_device_id(follower, 'follower')}",
            f"--robot.calibration_dir={cal_dir}",
            f"--robot.port={follower['port']}",
            "--robot.use_degrees=true",
            f"--robot.cameras={json.dumps(cameras)}",
        ]
    return [
        "--robot.type=bi_so_follower",
        "--robot.id=bimanual",
        f"--robot.calibration_dir={cal_dir}",
        f"--robot.left_arm_config.port={followers[0]['port']}",
        "--robot.left_arm_config.use_degrees=true",
        f"--robot.left_arm_config.cameras={json.dumps(left_cameras)}",
        f"--robot.right_arm_config.port={followers[1]['port']}",
        "--robot.right_arm_config.use_degrees=true",
        f"--robot.right_arm_config.cameras={json.dumps(right_cameras)}",
    ]


def build_dataset_argv(
    *,
    dataset_name: str,
    dataset_root: Path,
    task: str,
    num_episodes: int,
    episode_time_s: int,
    fps: int,
    vcodec: str,
) -> list[str]:
    return [
        f"--dataset.repo_id={dataset_name}",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={task}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.fps={fps}",
        f"--dataset.vcodec={vcodec}",
        "--dataset.push_to_hub=false",
        f"--dataset.video_encoding_batch_size={num_episodes + 1}",
        "--dataset.streaming_encoding=true",
    ]


def build_policy_overrides(
    *,
    policy_path: str | None,
    vla_path: str | None,
    rl_token_path: str | None,
    phase_mode: str | None = None,
    chunk_exec_steps: int | None = None,
) -> list[str]:
    if policy_path is None:
        return []
    overrides = [f"--policy.path={policy_path}"]
    if phase_mode is not None:
        overrides.append(f"--policy.phase_mode={phase_mode}")
    if chunk_exec_steps is not None:
        overrides.append(f"--policy.chunk_exec_steps={chunk_exec_steps}")
    if vla_path is not None:
        overrides.append(f"--policy.vla_pretrained_path={vla_path}")
    if rl_token_path is not None:
        overrides.append(f"--policy.rl_token_pretrained_path={rl_token_path}")
    return overrides


def build_rtc_argv(
    *,
    enabled: bool,
    execution_horizon: int,
    max_guidance_weight: float,
    prefix_attention_schedule: str,
    vla_execution_horizon: int | None,
    action_queue_size_to_get_new_actions: int | None,
) -> list[str]:
    argv = [
        f"--rlt.rtc_enabled={'true' if enabled else 'false'}",
        f"--rlt.rtc_execution_horizon={execution_horizon}",
        f"--rlt.rtc_max_guidance_weight={max_guidance_weight}",
        f"--rlt.rtc_prefix_attention_schedule={prefix_attention_schedule}",
    ]
    if vla_execution_horizon is not None:
        argv.append(f"--rlt.vla_rtc_execution_horizon={vla_execution_horizon}")
    if action_queue_size_to_get_new_actions is not None:
        argv.append(
            "--rlt.rtc_action_queue_size_to_get_new_actions="
            f"{action_queue_size_to_get_new_actions}"
        )
    return argv


def preflight_motor_connections(
    followers: list[dict[str, Any]],
    leaders: list[dict[str, Any]],
    cal_dir: str,
    leader_cal_dir: str | None,
) -> None:
    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
    from lerobot.robots.so_follower import SOFollower, SOFollowerConfig, SOFollowerRobotConfig
    from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderConfig, SOLeaderTeleopConfig

    def disconnect(device: Any) -> None:
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(device, arm_name, None)
            if arm is not None and arm.is_connected:
                arm.disconnect()
        if getattr(device, "is_connected", False):
            device.disconnect()

    log.info("Preflight checking follower motor connections before loading policy")
    if len(followers) == 1:
        robot = SOFollower(
            SOFollowerRobotConfig(
                id=_single_device_id(followers[0], "follower"),
                calibration_dir=Path(cal_dir),
                port=followers[0]["port"],
                use_degrees=True,
            )
        )
    else:
        robot = BiSOFollower(
            BiSOFollowerConfig(
                id="bimanual",
                calibration_dir=Path(cal_dir),
                left_arm_config=SOFollowerConfig(port=followers[0]["port"], use_degrees=True),
                right_arm_config=SOFollowerConfig(port=followers[1]["port"], use_degrees=True),
            )
        )
    try:
        robot.connect(calibrate=True)
        log.info("Preflight follower motor check passed")
    finally:
        disconnect(robot)

    if not leaders or leader_cal_dir is None:
        return

    log.info("Preflight checking leader motor connections before loading policy")
    if len(leaders) == 1:
        teleop = SOLeader(
            SOLeaderTeleopConfig(
                id=_single_device_id(leaders[0], "leader"),
                calibration_dir=Path(leader_cal_dir),
                port=leaders[0]["port"],
                use_degrees=True,
            )
        )
    else:
        teleop = BiSOLeader(
            BiSOLeaderConfig(
                id=TELEOP_ID,
                calibration_dir=Path(leader_cal_dir),
                left_arm_config=SOLeaderConfig(port=leaders[0]["port"], use_degrees=True),
                right_arm_config=SOLeaderConfig(port=leaders[1]["port"], use_degrees=True),
            )
        )
    try:
        teleop.connect(calibrate=True)
        log.info("Preflight leader motor check passed")
    finally:
        disconnect(teleop)


def set_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
