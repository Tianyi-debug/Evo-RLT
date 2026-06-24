import sys
from types import SimpleNamespace

import pytest
import time

from evo_rlt.adapters.lerobot.record.cli import build_parser
from evo_rlt.adapters.lerobot.record import runner
from evo_rlt.adapters.lerobot.record.runner import (
    _patch_double_tap_episode_outcome_listener,
    _patch_skip_policyless_reset_loop,
    build_default_collect_record_argv,
    build_segment_record_argv,
)


def test_initial_source_rejects_rlt():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "segment",
            "--initial-source",
            "rlt",
            "--critical-source",
            "rlt",
        ])


def test_segment_defaults_to_rtc_enabled():
    parser = build_parser()
    args = parser.parse_args([
        "segment",
        "--initial-source",
        "teleop",
        "--critical-source",
        "rlt",
        "--policy-path",
        "/tmp/ac",
    ])
    assert args.rtc is True


def test_segment_rlt_argv_marks_key_segment_with_teleop_start_and_rtc():
    args = SimpleNamespace(
        critical_source="rlt",
        initial_source="teleop",
        policy_path="/tmp/ac",
        vla_path="/tmp/vla",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        reset_time_s=None,
        fps=30,
        vcodec="h264",
        double_tap_window_s=0.6,
        intervention_action_blend_time_s=0.4,
        rtc=True,
        rtc_execution_horizon=10,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=None,
        vla_rtc_execution_horizon=None,
        vla_ref=True,
        chunk_exec_steps=25,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={},
        right_cameras={},
    )
    paths = SimpleNamespace(
        dataset_name="local/test",
        dataset_root="/tmp/dataset",
    )
    argv = build_segment_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--rlt.rl_phase_key_toggles_episode=true" in argv
    assert "--rlt.start_in_teleop=true" in argv
    assert "--rlt.rtc_enabled=true" in argv
    assert "--enable_episode_outcome_labeling=true" in argv
    assert "--policy_sync_to_teleop=true" in argv
    assert "--policy.path=/tmp/ac" in argv


def test_double_tap_episode_outcome_key_marks_success_and_failure(monkeypatch):
    control_utils = pytest.importorskip("lerobot.utils.control_utils")
    pedal_listener = pytest.importorskip("lerobot.utils.pedal_listener")

    captured = {}

    class FakePedalListener:
        def __init__(self, on_press):
            captured["on_press"] = on_press

        def start(self):
            return True

        def stop(self):
            captured["stopped"] = True

    def original_init_keyboard_listener(*args, **kwargs):
        return None, {"episode_outcome": None, "exit_early": False}

    monkeypatch.setattr(control_utils, "is_headless", lambda: True)
    monkeypatch.setattr(control_utils, "init_keyboard_listener", original_init_keyboard_listener)
    monkeypatch.setattr(pedal_listener, "PedalListener", FakePedalListener)

    _patch_double_tap_episode_outcome_listener(0.01, "e")
    listener, events = control_utils.init_keyboard_listener()

    captured["on_press"]("e")
    time.sleep(0.03)
    assert events["episode_outcome"] == "success"
    assert events["exit_early"] is True

    events["episode_outcome"] = None
    events["exit_early"] = False
    captured["on_press"]("e")
    captured["on_press"]("e")
    assert events["episode_outcome"] == "failure"
    assert events["exit_early"] is True
    listener.stop()


def test_skip_policyless_reset_loop_keeps_recording_loop(monkeypatch):
    lerobot_rlt_record = pytest.importorskip("lerobot.scripts.lerobot_rlt_record")

    calls = []

    def original_record_loop(*args, **kwargs):
        calls.append((args, kwargs))
        return "called"

    monkeypatch.setattr(lerobot_rlt_record, "record_loop", original_record_loop)

    _patch_skip_policyless_reset_loop()

    assert lerobot_rlt_record.record_loop(teleop=object(), control_time_s=10) is None
    assert calls == []
    assert lerobot_rlt_record.record_loop(policy=object(), dataset=object()) == "called"
    assert len(calls) == 1


def test_background_episode_video_encoding_submits_after_save(monkeypatch):
    calls = []

    class FakeFuture:
        def __init__(self, fn, args):
            self.fn = fn
            self.args = args

        def result(self):
            return self.fn(*self.args)

    class FakeExecutor:
        def __init__(self, max_workers, thread_name_prefix):
            calls.append(("executor", max_workers, thread_name_prefix))

        def submit(self, fn, *args):
            calls.append(("submit", args))
            return FakeFuture(fn, args)

        def shutdown(self, wait):
            calls.append(("shutdown", wait))

    class FakeMeta:
        total_episodes = 0
        video_keys = ["observation.images.wrist"]

        def _close_writer(self):
            calls.append(("close_writer", self.total_episodes))

    class FakeLeRobotDataset:
        def __init__(self):
            self.meta = FakeMeta()
            self.batch_encoding_size = 6
            self.episodes_since_last_encoding = 0

        def save_episode(self):
            calls.append(("save", self.batch_encoding_size))
            self.meta.total_episodes += 1
            self.episodes_since_last_encoding += 1
            return "saved"

        def _batch_save_episode_video(self, start_episode, end_episode):
            calls.append(("batch", start_episode, end_episode))

    class FakeVideoEncodingManager:
        def __init__(self, dataset):
            self.dataset = dataset

        def __exit__(self, exc_type, exc_val, exc_tb):
            calls.append(("exit", self.dataset.episodes_since_last_encoding))
            return False

    fake_lerobot_dataset = type(sys)("lerobot.datasets.lerobot_dataset")
    fake_lerobot_dataset.LeRobotDataset = FakeLeRobotDataset
    fake_video_utils = type(sys)("lerobot.datasets.video_utils")
    fake_video_utils.VideoEncodingManager = FakeVideoEncodingManager
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", fake_lerobot_dataset)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.video_utils", fake_video_utils)
    monkeypatch.setattr(runner.concurrent.futures, "ThreadPoolExecutor", FakeExecutor)

    runner._patch_background_episode_video_encoding()

    dataset = FakeLeRobotDataset()
    assert dataset.save_episode() == "saved"
    assert ("close_writer", 1) in calls
    assert ("submit", (0, 1)) in calls
    assert not any(call[0] == "batch" for call in calls)
    assert dataset.batch_encoding_size == 6
    assert dataset.episodes_since_last_encoding == 0

    FakeVideoEncodingManager(dataset).__exit__(None, None, None)
    assert ("batch", 0, 1) in calls
    assert ("shutdown", True) in calls
    assert ("exit", 0) in calls


def test_default_collect_parser_requires_user_policy_path():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["collect"])


def test_default_collect_parser_uses_open_source_safe_defaults():
    parser = build_parser()
    args = parser.parse_args(["collect", "--policy-path", "/tmp/ac"])

    assert args.policy_path == "/tmp/ac"
    assert args.vla_path is None
    assert args.rl_token_path is None
    assert args.dataset_tag == "vla_rlt_vla_test"
    assert args.num_episodes == 5
    assert args.rlt_toggle_key == "r"
    assert args.teleop_toggle_key == "space"
    assert args.episode_outcome_key == "e"
    assert args.start_with_teleop is False
    assert args.only_critical is False
    assert args.rtc is True
    assert args.rtc_execution_horizon == 10
    assert args.vla_rtc_execution_horizon == 25
    assert args.rtc_action_queue_size_to_get_new_actions == 30


def test_default_collect_argv_matches_best_real_robot_rtc_chunks():
    args = SimpleNamespace(
        policy_path="/tmp/ac",
        vla_path="/tmp/vla.pt",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        fps=30,
        vcodec="h264",
        double_tap_window_s=0.6,
        rtc=True,
        rtc_execution_horizon=10,
        vla_rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=30,
        vla_ref=True,
        play_sounds=True,
        rlt_toggle_key="r",
        teleop_toggle_key="space",
        episode_outcome_key="e",
        start_with_teleop=False,
        only_critical=False,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={
            "wrist": {
                "type": "opencv",
                "index_or_path": "/tmp/left-camera",
                "width": 640,
                "height": 480,
                "fps": 30,
                "fourcc": "MJPG",
            }
        },
        right_cameras={"wrist": {}, "front": {}},
    )
    paths = SimpleNamespace(dataset_name="local/eval_vla_rlt_vla_123456", dataset_root="/tmp/dataset")

    argv = build_default_collect_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--policy.phase_mode=manual" in argv
    assert "--rlt.enable=true" in argv
    assert "--rlt.rl_phase_key=r" in argv
    assert "--rlt.start_in_teleop=false" in argv
    assert "--rlt.rl_phase_key_toggles_critical_phase=true" in argv
    assert "--rlt.rl_phase_key_toggles_episode=true" not in argv
    assert "--rlt.skip_prefix_recording=true" not in argv
    assert "--rlt.rtc_execution_horizon=10" in argv
    assert "--rlt.vla_rtc_execution_horizon=25" in argv
    assert "--rlt.rtc_action_queue_size_to_get_new_actions=30" in argv
    assert "--enable_episode_outcome_labeling=true" in argv
    assert "--require_episode_success_label=true" in argv
    assert "--dataset.video_encoding_batch_size=6" in argv
    assert "--policy_sync_to_teleop=true" in argv
    assert "--vla_ref=true" in argv


def test_default_collect_only_critical_starts_recording_on_first_r_and_ends_on_second_r():
    args = SimpleNamespace(
        policy_path="/tmp/ac",
        vla_path="/tmp/vla.pt",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        fps=30,
        vcodec="h264",
        double_tap_window_s=0.6,
        rtc=True,
        rtc_execution_horizon=10,
        vla_rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=30,
        vla_ref=True,
        play_sounds=True,
        rlt_toggle_key="r",
        teleop_toggle_key="space",
        episode_outcome_key="e",
        start_with_teleop=False,
        only_critical=True,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={},
        right_cameras={},
    )
    paths = SimpleNamespace(dataset_name="local/eval_vla_rlt_vla_123456", dataset_root="/tmp/dataset")

    argv = build_default_collect_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--rlt.skip_prefix_recording=true" in argv
    assert "--rlt.rl_phase_key_toggles_episode=true" in argv
    assert "--rlt.start_in_teleop=false" in argv
    assert "--rlt.rl_phase_key_toggles_critical_phase=true" not in argv
    assert "--enable_episode_outcome_labeling=true" in argv
    assert "--require_episode_success_label=true" in argv
    assert "--policy_sync_to_teleop=true" in argv


def test_default_collect_start_with_teleop_sets_episode_initial_source():
    args = SimpleNamespace(
        policy_path="/tmp/ac",
        vla_path="/tmp/vla.pt",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        fps=30,
        vcodec="h264",
        double_tap_window_s=0.6,
        rtc=True,
        rtc_execution_horizon=10,
        vla_rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=30,
        vla_ref=True,
        play_sounds=True,
        rlt_toggle_key="r",
        teleop_toggle_key="space",
        episode_outcome_key="e",
        start_with_teleop=True,
        only_critical=False,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={},
        right_cameras={},
    )
    paths = SimpleNamespace(dataset_name="local/eval_vla_rlt_vla_123456", dataset_root="/tmp/dataset")

    argv = build_default_collect_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--rlt.start_in_teleop=true" in argv
    assert "--rlt.rl_phase_key_toggles_critical_phase=true" in argv
    assert "--rlt.rl_phase_key_toggles_episode=true" not in argv


def test_full_vla_pedal_outcome_parser():
    parser = build_parser()
    args = parser.parse_args([
        "full",
        "--initial-source",
        "vla",
        "--policy-path",
        "/tmp/ac",
        "--vla-path",
        "/tmp/base.pt",
        "--phase-mode",
        "always_vla",
        "--chunk-exec-steps",
        "25",
        "--pedal-outcome",
        "--episode-outcome-key",
        "e",
        "--reset-time-s",
        "0",
    ])

    assert args.rtc is True
    assert args.pedal_outcome is True
    assert args.episode_outcome_key == "e"
    assert args.phase_mode == "always_vla"
    assert args.chunk_exec_steps == 25
    assert args.reset_time_s == 0
