from types import SimpleNamespace

import pytest

from evo_rlt.adapters.lerobot.record.cli import build_parser
from evo_rlt.adapters.lerobot.record.runner import (
    _patch_episode_end_pedal_listener,
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


def test_episode_end_pedal_maps_space_to_exit_early(monkeypatch):
    import lerobot.utils.control_utils as control_utils
    import lerobot.utils.pedal_listener as pedal_listener

    captured = {}

    class FakePedalListener:
        def __init__(self, on_press):
            captured["on_press"] = on_press

        def start(self):
            return True

        def stop(self):
            captured["stopped"] = True

    def original_start_pedal_listener(events, *args, **kwargs):
        from lerobot.utils.pedal_listener import PedalListener

        return PedalListener(lambda key: events.setdefault("forwarded", []).append(key))

    monkeypatch.setattr(control_utils, "_start_pedal_listener", original_start_pedal_listener)
    monkeypatch.setattr(pedal_listener, "PedalListener", FakePedalListener)

    _patch_episode_end_pedal_listener("space")
    events = {"exit_early": False}
    control_utils._start_pedal_listener(
        events,
        " ",
        None,
        "r",
        None,
        None,
        "s",
        "f",
        None,
        None,
    )

    captured["on_press"]("space")
    assert events["exit_early"] is True
    assert "forwarded" not in events

    captured["on_press"]("r")
    assert events["forwarded"] == ["r"]


def test_default_collect_parser_uses_vla_rlt_vla_defaults():
    parser = build_parser()
    args = parser.parse_args(["collect"])

    assert args.policy_path == "/home/kye/rlt_deploy/ac_online_base_0528"
    assert args.vla_path.endswith("/online_base_vla_0528.pt")
    assert args.rl_token_path == "/home/kye/rlt_deploy/rlt_online_base_0528"
    assert args.dataset_tag == "vla_rlt_vla_test"
    assert args.num_episodes == 5
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
    assert "--rlt.rl_phase_key_toggles_critical_phase=true" in argv
    assert "--rlt.rtc_execution_horizon=10" in argv
    assert "--rlt.vla_rtc_execution_horizon=25" in argv
    assert "--rlt.rtc_action_queue_size_to_get_new_actions=30" in argv
    assert "--dataset.video_encoding_batch_size=6" in argv
    assert "--policy_sync_to_teleop=true" in argv
    assert "--vla_ref=true" in argv


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
        "--reset-time-s",
        "0",
    ])

    assert args.rtc is True
    assert args.pedal_outcome is True
    assert args.phase_mode == "always_vla"
    assert args.chunk_exec_steps == 25
    assert args.reset_time_s == 0
