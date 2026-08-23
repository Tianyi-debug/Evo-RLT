import json

import pytest

from evo_rlt.adapters.lerobot.record.annotations import EPISODE_FAILURE, EPISODE_SUCCESS
from evo_rlt.adapters.lerobot.record import reset_pose


class FakeRobot:
    robot_type = "fake_robot"
    action_features = {"motor_1.pos": float, "motor_2.pos": float}

    def __init__(self, observation=None):
        self.observation = observation or {"motor_1.pos": 0.0, "motor_2.pos": 0.0}
        self.sent_actions = []

    def get_observation(self):
        return dict(self.observation)

    def send_action(self, action):
        self.sent_actions.append(dict(action))
        self.observation.update(action)
        return action


class FakeTeleop:
    def __init__(self, action=None):
        self.action = action or {"motor_1.pos": 0.0, "motor_2.pos": 0.0}
        self.feedback = []

    def get_action(self):
        return dict(self.action)

    def send_feedback(self, feedback):
        self.feedback.append(dict(feedback))


class FakeStdin:
    def isatty(self):
        return True


def test_save_and_load_reset_pose(tmp_path):
    robot = FakeRobot({"motor_1.pos": 12.5, "motor_2.pos": -3.0, "front": object()})
    pose_path = tmp_path / "reset_pose.json"

    saved_pose = reset_pose.save_reset_pose(robot=robot, pose_path=pose_path)
    loaded_pose = reset_pose.load_reset_pose(pose_path)

    with open(pose_path) as f:
        payload = json.load(f)
    assert saved_pose == {"motor_1.pos": 12.5, "motor_2.pos": -3.0}
    assert loaded_pose == saved_pose
    assert payload["joint_pos"] == saved_pose


def test_controller_captures_missing_pose(tmp_path, monkeypatch):
    robot = FakeRobot({"motor_1.pos": 1.0, "motor_2.pos": 2.0})
    controller = reset_pose.EpisodeResetPoseController(
        cfg=object(),
        pose_path=tmp_path / "new_pose.json",
    )
    monkeypatch.setattr("builtins.input", lambda _: "")

    controller.on_record_connected(robot=robot, teleop=None)

    assert controller.reset_pose == {"motor_1.pos": 1.0, "motor_2.pos": 2.0}
    assert controller.pose_path.is_file()


def test_capture_reset_pose_with_teleop_moves_follower_until_enter(tmp_path, monkeypatch):
    robot = FakeRobot({"motor_1.pos": 0.0, "motor_2.pos": 0.0})
    teleop = FakeTeleop({"motor_1.pos": 5.0, "motor_2.pos": -6.0})
    pose_path = tmp_path / "teleop_pose.json"
    ready_values = iter([False, False, True])
    monkeypatch.setattr(reset_pose.sys, "stdin", FakeStdin())
    monkeypatch.setattr(reset_pose, "_stdin_ready", lambda: next(ready_values))
    monkeypatch.setattr(reset_pose, "_read_stdin_line", lambda: None)
    monkeypatch.setattr(reset_pose.time, "sleep", lambda _: None)

    pose = reset_pose.capture_reset_pose_with_teleop(robot=robot, teleop=teleop, pose_path=pose_path)

    assert len(robot.sent_actions) == 2
    assert pose == {"motor_1.pos": 5.0, "motor_2.pos": -6.0}
    assert reset_pose.load_reset_pose(pose_path) == pose


def test_controller_reuses_existing_pose(tmp_path, monkeypatch):
    pose_path = tmp_path / "existing_pose.json"
    pose_path.write_text(json.dumps({"joint_pos": {"motor_1.pos": 3.0, "motor_2.pos": -4.0}}))
    controller = reset_pose.EpisodeResetPoseController(cfg=object(), pose_path=pose_path)
    calls = []
    monkeypatch.setattr("builtins.input", lambda _: calls.append("input"))
    monkeypatch.setattr(
        reset_pose,
        "slow_reset_all_arms_to_pose",
        lambda **kwargs: calls.append(kwargs),
    )
    robot = FakeRobot()

    controller.on_record_connected(robot=robot, teleop=None)

    assert controller.reset_pose == {"motor_1.pos": 3.0, "motor_2.pos": -4.0}
    assert calls == [
        {
            "robot": robot,
            "teleop": None,
            "target_pose": {"motor_1.pos": 3.0, "motor_2.pos": -4.0},
            "duration_s": 3.0,
        }
    ]


def test_controller_recaptures_existing_pose_with_teleop(tmp_path, monkeypatch):
    pose_path = tmp_path / "existing_pose.json"
    pose_path.write_text(json.dumps({"joint_pos": {"motor_1.pos": 3.0, "motor_2.pos": -4.0}}))
    robot = FakeRobot({"motor_1.pos": 0.0, "motor_2.pos": 0.0})
    teleop = FakeTeleop({"motor_1.pos": 7.0, "motor_2.pos": 8.0})
    controller = reset_pose.EpisodeResetPoseController(cfg=object(), pose_path=pose_path, recapture=True)
    ready_values = iter([False, True])
    monkeypatch.setattr(reset_pose.sys, "stdin", FakeStdin())
    monkeypatch.setattr(reset_pose, "_stdin_ready", lambda: next(ready_values))
    monkeypatch.setattr(reset_pose, "_read_stdin_line", lambda: None)
    monkeypatch.setattr(reset_pose.time, "sleep", lambda _: None)

    controller.on_record_connected(robot=robot, teleop=teleop)

    assert controller.reset_pose == {"motor_1.pos": 7.0, "motor_2.pos": 8.0}
    assert reset_pose.load_reset_pose(pose_path) == controller.reset_pose


def test_slow_reset_all_arms_to_pose_interpolates_and_feedbacks(monkeypatch):
    robot = FakeRobot({"motor_1.pos": 0.0, "motor_2.pos": 0.0})
    teleop = FakeTeleop()
    monkeypatch.setattr(reset_pose.time, "sleep", lambda _: None)

    reset_pose.slow_reset_all_arms_to_pose(
        robot=robot,
        teleop=teleop,
        target_pose={"motor_1.pos": 1.0, "motor_2.pos": -2.0},
        duration_s=0.1,
    )

    assert robot.sent_actions[-1] == {"motor_1.pos": 1.0, "motor_2.pos": -2.0}
    assert len(robot.sent_actions) > 1
    assert teleop.feedback[-1] == robot.sent_actions[-1]


def test_slow_reset_can_open_then_close_gripper_during_return(monkeypatch):
    robot = FakeRobot({"motor_1.pos": 0.0, "motor_2.pos": 0.0})
    robot.action_features = {**robot.action_features, "gripper.pos": float}
    robot.observation["gripper.pos"] = 3.0
    monkeypatch.setattr(reset_pose.time, "sleep", lambda _: None)

    reset_pose.slow_reset_all_arms_to_pose(
        robot=robot,
        teleop=None,
        target_pose={"motor_1.pos": 10.0, "motor_2.pos": -5.0, "gripper.pos": 2.9},
        duration_s=1.0,
        gripper_joint="gripper.pos",
        gripper_open_position=40.0,
        gripper_open_fraction=0.25,
        gripper_close_fraction=0.60,
    )

    gripper_positions = [action["gripper.pos"] for action in robot.sent_actions]
    assert max(gripper_positions) == pytest.approx(40.0)
    assert gripper_positions[-1] == pytest.approx(2.9)
    assert robot.sent_actions[-1]["motor_1.pos"] == pytest.approx(10.0)
    assert robot.sent_actions[-1]["motor_2.pos"] == pytest.approx(-5.0)


def test_controller_only_releases_gripper_after_episode_outcome(tmp_path, monkeypatch):
    pose_path = tmp_path / "existing_pose.json"
    pose_path.write_text(json.dumps({"joint_pos": {"motor_1.pos": 3.0, "gripper.pos": 2.9}}))
    controller = reset_pose.EpisodeResetPoseController(
        cfg=object(),
        pose_path=pose_path,
        gripper_release_enabled=True,
        gripper_open_position=40.0,
    )
    calls = []
    monkeypatch.setattr(reset_pose, "slow_reset_all_arms_to_pose", lambda **kwargs: calls.append(kwargs))
    robot = FakeRobot()

    controller.on_record_connected(robot=robot, teleop=None)
    controller.on_episode_outcome(robot=robot, teleop=None, episode_success=EPISODE_SUCCESS)

    assert "gripper_joint" not in calls[0]
    assert calls[1]["gripper_joint"] == "gripper.pos"
    assert calls[1]["gripper_open_position"] == pytest.approx(40.0)


def test_controller_resets_only_on_final_outcome(monkeypatch):
    robot = FakeRobot({"motor_1.pos": 0.0, "motor_2.pos": 0.0})
    controller = reset_pose.EpisodeResetPoseController(cfg=object())
    controller.reset_pose = {"motor_1.pos": 1.0, "motor_2.pos": -2.0}
    calls = []

    def fake_slow_reset_all_arms_to_pose(**kwargs):
        calls.append(kwargs["target_pose"])

    monkeypatch.setattr(
        reset_pose,
        "slow_reset_all_arms_to_pose",
        fake_slow_reset_all_arms_to_pose,
    )

    controller.on_episode_outcome(robot=robot, teleop=None, episode_success=None)
    controller.on_episode_outcome(robot=robot, teleop=None, episode_success=EPISODE_SUCCESS)
    controller.on_episode_outcome(robot=robot, teleop=None, episode_success=EPISODE_FAILURE)

    assert calls == [
        {"motor_1.pos": 1.0, "motor_2.pos": -2.0},
        {"motor_1.pos": 1.0, "motor_2.pos": -2.0},
    ]
