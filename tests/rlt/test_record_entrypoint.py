from types import SimpleNamespace

import pytest

from evo_rlt.adapters.lerobot.record.cli import build_parser
from evo_rlt.adapters.lerobot.record.runner import build_segment_record_argv


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


def test_segment_defaults_to_rtc_disabled():
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
    assert args.rtc is False


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
        fps=30,
        vcodec="h264",
        double_tap_window_s=0.6,
        intervention_action_blend_time_s=0.4,
        rtc=True,
        rtc_execution_horizon=10,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=None,
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
