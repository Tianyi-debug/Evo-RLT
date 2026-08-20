from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from evo_rlt.adapters.lerobot.record import loop
from evo_rlt.adapters.lerobot.record.loop import _extract_hold_action


def test_extract_hold_action_uses_observation_and_complete_fallback():
    hold = _extract_hold_action(
        {
            "shoulder_pan.pos": np.array([12.5], dtype=np.float32),
            "shoulder_lift.pos": -7.0,
        },
        ["shoulder_pan.pos", "shoulder_lift.pos", "gripper.pos"],
        fallback_action={"gripper.pos": 31.0},
    )

    assert hold == {
        "shoulder_pan.pos": 12.5,
        "shoulder_lift.pos": -7.0,
        "gripper.pos": 31.0,
    }


@pytest.mark.parametrize(
    ("event_name", "expected_reason"),
    [
        ("toggle_intervention", loop.INTERVENTION_REASON_CORRECTIVE),
        ("toggle_proactive_intervention", loop.INTERVENTION_REASON_PROACTIVE),
    ],
)
def test_two_stage_intervention_holds_then_teleops_then_releases(
    monkeypatch,
    event_name,
    expected_reason,
):
    class IdentityPipeline:
        def __call__(self, value):
            return value[0] if isinstance(value, tuple) else value

        def reset(self):
            pass

    class FakeTeleop:
        def __init__(self):
            self.manual_control_calls = []

        def set_manual_control(self, enabled):
            self.manual_control_calls.append(enabled)

        def get_action(self):
            return {"joint.pos": 99.0}

    class FakeRobot:
        name = "so101_follower"
        robot_type = "so101_follower"
        action_features = {"joint.pos": float}

        def __init__(self):
            self.actions = []

        def get_observation(self):
            return {"joint.pos": 42.0}

        def send_action(self, action):
            self.actions.append(action)
            return {"joint.pos": 50.0}

    class FakePolicy:
        config = SimpleNamespace(input_features={}, device="cpu", use_amp=False)

        def reset(self):
            pass

    class FakeDataset:
        fps = 30
        root = None
        features = {
            "action": {"names": ["joint.pos"]},
            "complementary_info.policy_action": {"names": ["joint.pos"]},
            "complementary_info.requested_action": {"names": ["joint.pos"]},
            "complementary_info.is_intervention": {},
            "complementary_info.state": {},
            "complementary_info.intervention_stage": {},
            "complementary_info.intervention_reason": {},
            "complementary_info.collector_policy_id": {},
            "complementary_info.phase": {},
        }

        def __init__(self, events):
            self.events = events
            self.frames = []
            self.episode_buffer = {"size": 0}

        def add_frame(self, frame):
            self.frames.append(frame)
            self.episode_buffer["size"] += 1
            if len(self.frames) < 3:
                self.events[event_name] = True
            else:
                self.events["exit_early"] = True

    class FakeSyncExecutor:
        def __init__(self):
            self.actions = []

        def send_action(self, action):
            self.actions.append(action)
            return {"joint.pos": float(action["joint.pos"]) - 1.0}

    monkeypatch.setattr(loop, "Teleoperator", FakeTeleop)
    monkeypatch.setattr(loop, "precise_sleep", lambda _: None)
    monkeypatch.setattr(loop, "log_say", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        loop,
        "_predict_policy_action_with_acp_inference",
        lambda **kwargs: np.array([7.0], dtype=np.float32),
    )
    monkeypatch.setattr(loop, "make_robot_action", lambda action, features: {"joint.pos": 7.0})
    monkeypatch.setattr(
        loop,
        "build_dataset_frame",
        lambda features, values, prefix: {prefix: dict(values)},
    )

    teleop = FakeTeleop()
    events = defaultdict(bool)
    events[event_name] = True
    dataset = FakeDataset(events)
    robot = FakeRobot()
    sync_executor = FakeSyncExecutor()
    loop.record_loop(
        robot=robot,
        events=events,
        fps=30,
        teleop_action_processor=IdentityPipeline(),
        robot_action_processor=IdentityPipeline(),
        robot_observation_processor=IdentityPipeline(),
        dataset=dataset,
        teleop=teleop,
        policy=FakePolicy(),
        preprocessor=IdentityPipeline(),
        postprocessor=IdentityPipeline(),
        control_time_s=1,
        single_task="task",
        policy_sync_executor=sync_executor,
        two_stage_intervention=True,
    )

    assert teleop.manual_control_calls == [False, False, True, False]
    assert len(sync_executor.actions) == 2
    assert sync_executor.actions[0]["joint.pos"] == 42.0
    assert sync_executor.actions[1]["joint.pos"] == 7.0
    assert robot.actions == [{"joint.pos": 99.0}]
    assert [frame["action"]["joint.pos"] for frame in dataset.frames] == [41.0, 50.0, 6.0]
    assert [
        frame["complementary_info.requested_action"]["joint.pos"]
        for frame in dataset.frames
    ] == [42.0, 99.0, 7.0]
    assert [
        frame["complementary_info.policy_action"]["joint.pos"]
        for frame in dataset.frames
    ] == [0.0, 0.0, 7.0]
    assert [
        frame["complementary_info.is_intervention"].item()
        for frame in dataset.frames
    ] == [1.0, 1.0, 0.0]
    assert [
        frame["complementary_info.intervention_stage"].item()
        for frame in dataset.frames
    ] == [
        loop.INTERVENTION_STAGE_HOLD,
        loop.INTERVENTION_STAGE_TELEOP,
        loop.INTERVENTION_STAGE_RELEASE,
    ]
    assert [
        frame["complementary_info.intervention_reason"].item()
        for frame in dataset.frames
    ] == [expected_reason, expected_reason, expected_reason]
