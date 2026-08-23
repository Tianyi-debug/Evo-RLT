from __future__ import annotations

import json
import logging
import select
import sys
import time
from pathlib import Path
from typing import Any

from lerobot.utils.constants import HF_LEROBOT_HOME

from evo_rlt.adapters.lerobot.record.annotations import EPISODE_FAILURE, EPISODE_SUCCESS
from evo_rlt.adapters.lerobot.record.hil import send_teleop_feedback


def default_reset_pose_path(cfg: Any) -> Path:
    robot_cfg = getattr(cfg, "robot", None)
    robot_id = getattr(robot_cfg, "id", None) or "default"
    robot_type = getattr(robot_cfg, "type", None) or type(robot_cfg).__name__
    return HF_LEROBOT_HOME / "failure_reset_pose" / f"{robot_type}_{robot_id}.json"


def extract_joint_pos_from_observation(observation: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in observation.items() if key.endswith(".pos")}


def save_reset_pose(robot: Any, pose_path: Path) -> dict[str, float]:
    observation = robot.get_observation()
    joint_pos = extract_joint_pos_from_observation(observation)
    if not joint_pos:
        raise ValueError("Could not capture reset pose: no '.pos' joints found in observation.")

    pose_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"robot_type": getattr(robot, "robot_type", type(robot).__name__), "joint_pos": joint_pos}
    with open(pose_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logging.info("Saved reset pose to %s", pose_path)
    return joint_pos


def load_reset_pose(pose_path: Path) -> dict[str, float]:
    with open(pose_path) as f:
        payload = json.load(f)
    joint_pos_raw = payload["joint_pos"] if isinstance(payload, dict) and "joint_pos" in payload else payload
    if not isinstance(joint_pos_raw, dict):
        raise ValueError(f"Invalid reset pose payload in {pose_path}: expected dict, got {type(joint_pos_raw)}")
    joint_pos = {str(key): float(value) for key, value in joint_pos_raw.items() if str(key).endswith(".pos")}
    if not joint_pos:
        raise ValueError(f"Invalid reset pose payload in {pose_path}: no '.pos' joints found.")
    logging.info("Loaded reset pose from %s", pose_path)
    return joint_pos


def _stdin_ready() -> bool:
    return sys.stdin.isatty() and bool(select.select([sys.stdin], [], [], 0.0)[0])


def _read_stdin_line() -> None:
    sys.stdin.readline()


def capture_reset_pose_with_teleop(
    robot: Any,
    teleop: Any,
    pose_path: Path,
    *,
    fps: float = 30.0,
) -> dict[str, float]:
    if not sys.stdin.isatty():
        logging.warning("No interactive stdin available; capturing the current follower pose as reset pose.")
        return save_reset_pose(robot=robot, pose_path=pose_path)

    print(
        "Episode reset pose capture is enabled.\n"
        "Use the leader arm to move the follower to the episode initial position.\n"
        "Press ENTER when the follower is ready; episode recording will start after capture.\n"
        f"Saving reset pose to: {pose_path}",
        flush=True,
    )

    action_keys = set(robot.action_features)
    sleep_s = 1.0 / max(float(fps), 1.0)
    while True:
        if _stdin_ready():
            _read_stdin_line()
            return save_reset_pose(robot=robot, pose_path=pose_path)

        action = teleop.get_action()
        robot_action = {key: value for key, value in action.items() if key in action_keys}
        if not robot_action:
            logging.warning(
                "Teleop action keys do not match robot action keys; capture the current follower pose manually."
            )
            input("Move the follower to the episode initial position, then press ENTER to capture.\n")
            return save_reset_pose(robot=robot, pose_path=pose_path)

        robot.send_action(robot_action)
        time.sleep(sleep_s)


def slow_reset_all_arms_to_pose(
    robot: Any,
    teleop: Any,
    target_pose: dict[str, float],
    duration_s: float = 3.0,
    *,
    gripper_joint: str | None = None,
    gripper_open_position: float | None = None,
    gripper_open_fraction: float = 0.25,
    gripper_close_fraction: float = 0.60,
) -> None:
    joint_keys = [key for key in robot.action_features if key.endswith(".pos") and key in target_pose]
    if not joint_keys:
        logging.warning("No matching '.pos' joints found for the stored reset pose.")
        return

    current_pose = extract_joint_pos_from_observation(robot.get_observation())
    start_pose = {key: current_pose.get(key, float(target_pose[key])) for key in joint_keys}
    goal_pose = {key: float(target_pose[key]) for key in joint_keys}

    if gripper_joint is not None and gripper_open_position is not None:
        if gripper_joint not in joint_keys:
            logging.warning(
                "Reset gripper release requested, but joint %s is absent from robot actions or reset pose.",
                gripper_joint,
            )
            gripper_joint = None
            gripper_open_position = None
        else:
            logging.info(
                "Episode reset will open %s to %.2f, then close it to the stored pose value %.2f.",
                gripper_joint,
                gripper_open_position,
                goal_pose[gripper_joint],
            )

    if teleop is not None and not isinstance(teleop, list) and hasattr(teleop, "set_manual_control"):
        teleop.set_manual_control(False)

    step_dt_s = 0.05
    steps = max(int(duration_s / step_dt_s), 1)
    teleop_feedback_enabled = teleop is not None and not isinstance(teleop, list)
    for idx in range(1, steps + 1):
        alpha = idx / steps
        action = {key: start_pose[key] + (goal_pose[key] - start_pose[key]) * alpha for key in joint_keys}
        if (
            gripper_joint is not None
            and gripper_open_position is not None
            and gripper_joint in action
        ):
            if alpha <= gripper_open_fraction:
                gripper_alpha = alpha / gripper_open_fraction
                action[gripper_joint] = start_pose[gripper_joint] + (
                    gripper_open_position - start_pose[gripper_joint]
                ) * gripper_alpha
            elif alpha < gripper_close_fraction:
                action[gripper_joint] = gripper_open_position
            else:
                gripper_alpha = (alpha - gripper_close_fraction) / (1.0 - gripper_close_fraction)
                action[gripper_joint] = gripper_open_position + (
                    goal_pose[gripper_joint] - gripper_open_position
                ) * gripper_alpha
        robot.send_action(action)
        if teleop_feedback_enabled:
            try:
                send_teleop_feedback(teleop, action)
            except NotImplementedError:
                teleop_feedback_enabled = False
                logging.info("Teleop device does not support reset-pose feedback; continuing follower-only reset.")
        time.sleep(step_dt_s)

    logging.info("Episode ended. Arms returned to the stored reset pose in %.1fs.", duration_s)


class EpisodeResetPoseController:
    def __init__(
        self,
        cfg: Any,
        *,
        pose_path: str | Path | None = None,
        duration_s: float = 3.0,
        capture_if_missing: bool = True,
        recapture: bool = False,
        capture_fps: float = 30.0,
        gripper_release_enabled: bool = False,
        gripper_joint: str = "gripper.pos",
        gripper_open_position: float = 40.0,
        gripper_open_fraction: float = 0.25,
        gripper_close_fraction: float = 0.60,
    ):
        self.pose_path = Path(pose_path).expanduser() if pose_path is not None else default_reset_pose_path(cfg)
        self.duration_s = float(duration_s)
        self.capture_if_missing = capture_if_missing
        self.recapture = recapture
        self.capture_fps = float(capture_fps)
        self.gripper_release_enabled = bool(gripper_release_enabled)
        self.gripper_joint = str(gripper_joint)
        self.gripper_open_position = float(gripper_open_position)
        self.gripper_open_fraction = float(gripper_open_fraction)
        self.gripper_close_fraction = float(gripper_close_fraction)
        self.reset_pose: dict[str, float] | None = None

    def on_record_connected(self, robot: Any, teleop: Any) -> None:
        if self.pose_path.is_file() and not self.recapture:
            self.reset_pose = load_reset_pose(self.pose_path)
            print(f"Loaded episode reset pose: {self.pose_path}", flush=True)
            logging.info("Moving arms to the loaded reset pose before the first episode.")
            slow_reset_all_arms_to_pose(
                robot=robot,
                teleop=teleop,
                target_pose=self.reset_pose,
                duration_s=self.duration_s,
            )
            return

        if not self.capture_if_missing:
            logging.warning("Reset pose is enabled but %s does not exist.", self.pose_path)
            return

        if teleop is not None and not isinstance(teleop, list):
            self.reset_pose = capture_reset_pose_with_teleop(
                robot=robot,
                teleop=teleop,
                pose_path=self.pose_path,
                fps=self.capture_fps,
            )
            return

        input(
            "Episode reset pose is enabled.\n"
            "Move the follower arm to the episode initial position, then press ENTER to capture:\n"
            f"{self.pose_path}\n"
        )
        self.reset_pose = save_reset_pose(robot=robot, pose_path=self.pose_path)

    def on_episode_outcome(self, robot: Any, teleop: Any, episode_success: str | None) -> None:
        if episode_success in {EPISODE_FAILURE, EPISODE_SUCCESS} and self.reset_pose is not None:
            slow_reset_all_arms_to_pose(
                robot=robot,
                teleop=teleop,
                target_pose=self.reset_pose,
                duration_s=self.duration_s,
                gripper_joint=self.gripper_joint if self.gripper_release_enabled else None,
                gripper_open_position=(
                    self.gripper_open_position if self.gripper_release_enabled else None
                ),
                gripper_open_fraction=self.gripper_open_fraction,
                gripper_close_fraction=self.gripper_close_fraction,
            )
