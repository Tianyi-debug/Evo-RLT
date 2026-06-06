from __future__ import annotations

import argparse
import logging
import runpy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from evo_rlt.adapters.lerobot import register
from evo_rlt.adapters.lerobot.record.common import (
    build_dataset_argv,
    build_policy_overrides,
    build_robot_argv,
    build_rtc_argv,
    build_teleop_argv,
    configure_logging,
    load_robot_setup,
    preflight_motor_connections,
    remove_existing_dataset,
    resolve_run_paths,
    set_offline_env,
    stage_follower_calibrations,
    stage_leader_calibrations,
)

log = logging.getLogger(__name__)


def prepare_lerobot_runtime() -> None:
    register()
    # lerobot_rlt_record imports ChunkACPolicy through this concrete module path.
    import evo_rlt.adapters.lerobot.policies.modeling_rlt_ac as modeling_rlt_ac

    sys.modules["lerobot.policies.rlt.modeling_rlt_ac"] = modeling_rlt_ac


def run_segment(args: argparse.Namespace) -> None:
    set_offline_env()
    setup = load_robot_setup(args.setup_json)
    paths = resolve_run_paths(setup.setup, args.dataset_tag, f"eval_{args.critical_source}_segment")
    configure_logging(paths.log_file, args.log_level)
    remove_existing_dataset(paths.dataset_root)

    if args.critical_source in {"rlt", "vla"} and args.policy_path is None:
        raise ValueError("segment recording with critical-source rlt/vla requires --policy-path")
    if args.rtc and not args.vla_ref and args.critical_source == "rlt":
        raise ValueError("RTC RLT recording requires --vla-ref; --no-vla-ref hides the guided reference")

    teleop_argv = build_teleop_argv(setup.leaders, args.no_teleop)
    if args.initial_source == "teleop" and not teleop_argv:
        raise ValueError("--initial-source teleop requires leader teleop arms")

    leader_cal_dir = None
    with TemporaryDirectory(prefix="record-segment-") as cal_dir:
        stage_follower_calibrations(setup.followers, cal_dir)
        leader_cal_dir = stage_leader_calibrations(setup.leaders, teleop_argv)
        if not args.dry_run and args.preflight:
            preflight_motor_connections(
                setup.followers,
                setup.leaders if teleop_argv else [],
                cal_dir,
                leader_cal_dir.name if leader_cal_dir is not None else None,
            )

        sys.argv = build_segment_record_argv(args, setup, paths, cal_dir, teleop_argv)
        print_segment_summary(args, paths)
        if args.dry_run:
            print("\nDry run argv:")
            print(" ".join(sys.argv))
            return

        prepare_lerobot_runtime()
        from lerobot.scripts.lerobot_rlt_record import record

        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()


def build_segment_record_argv(args, setup, paths, cal_dir: str, teleop_argv: list[str]) -> list[str]:
    argv = [
        "record_segment",
        *build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, cal_dir),
        *teleop_argv,
        *build_segment_policy_argv(args),
        *build_dataset_argv(
            dataset_name=paths.dataset_name,
            dataset_root=paths.dataset_root,
            task=args.task,
            num_episodes=args.num_episodes,
            episode_time_s=args.episode_time_s,
            fps=args.fps,
            vcodec=args.vcodec,
        ),
        "--rlt.enable=true",
        "--rlt.skip_prefix_recording=true",
        "--rlt.rl_phase_key_toggles_episode=true",
        f"--rlt.start_in_teleop={'true' if args.initial_source == 'teleop' else 'false'}",
        f"--rlt.rl_phase_double_tap_window_s={args.double_tap_window_s}",
        f"--rlt.intervention_action_blend_time_s={args.intervention_action_blend_time_s}",
        *build_rtc_argv(
            enabled=args.rtc,
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            action_queue_size_to_get_new_actions=args.rtc_action_queue_size_to_get_new_actions,
        ),
        "--enable_episode_outcome_labeling=true",
        "--intervention_state_machine_enabled=true",
        f"--policy_sync_to_teleop={'true' if teleop_argv and args.critical_source in {'rlt', 'vla'} else 'false'}",
        f"--vla_ref={'true' if args.vla_ref else 'false'}",
        "--play_sounds=true",
    ]
    return argv


def build_segment_policy_argv(args: argparse.Namespace) -> list[str]:
    if args.critical_source == "vla":
        return build_policy_overrides(
            policy_path=args.policy_path,
            vla_path=args.vla_path,
            rl_token_path=args.rl_token_path,
            phase_mode="always_vla",
            chunk_exec_steps=args.chunk_exec_steps,
        )
    return build_policy_overrides(
        policy_path=args.policy_path,
        vla_path=args.vla_path,
        rl_token_path=args.rl_token_path,
    )


def print_segment_summary(args: argparse.Namespace, paths) -> None:
    print(f"\nDataset: {paths.dataset_name} -> {paths.dataset_root}")
    print(f"Log: {paths.log_file}")
    print(f"Initial source: {args.initial_source}")
    print(f"Critical source: {args.critical_source}")
    print(
        "RTC: "
        f"enabled={args.rtc} horizon={args.rtc_execution_horizon} "
        f"guidance={args.rtc_max_guidance_weight} "
        f"schedule={args.rtc_prefix_attention_schedule} "
        f"refill_threshold={args.rtc_action_queue_size_to_get_new_actions or 'chunk_length-1'}"
    )
    if args.policy_path is not None:
        print(f"Policy: {args.policy_path}")
    print(
        "Segment outcome mode: r starts the recorded critical segment; "
        "r ends it as success; r+r inside "
        f"{args.double_tap_window_s:.1f}s marks failure."
    )


def run_full(args: argparse.Namespace) -> None:
    set_offline_env()
    setup = load_robot_setup(args.setup_json)
    paths = resolve_run_paths(setup.setup, args.dataset_tag, f"eval_{args.initial_source}_full")
    configure_logging(paths.log_file, args.log_level)
    remove_existing_dataset(paths.dataset_root)
    teleop_argv = build_teleop_argv(setup.leaders, args.no_teleop)

    if args.initial_source == "vla" and args.policy_path is None:
        raise ValueError("full recording with --initial-source vla requires --policy-path")
    if args.initial_source == "teleop" and not teleop_argv:
        raise ValueError("full recording with --initial-source teleop requires leader teleop arms")

    leader_cal_dir = None
    with TemporaryDirectory(prefix="record-full-") as cal_dir:
        stage_follower_calibrations(setup.followers, cal_dir)
        leader_cal_dir = stage_leader_calibrations(setup.leaders, teleop_argv)
        sys.argv = [
            "record_full",
            *build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, cal_dir),
            *teleop_argv,
            *build_policy_overrides(
                policy_path=args.policy_path,
                vla_path=args.vla_path,
                rl_token_path=args.rl_token_path,
            ),
            *build_dataset_argv(
                dataset_name=paths.dataset_name,
                dataset_root=paths.dataset_root,
                task=args.task,
                num_episodes=args.num_episodes,
                episode_time_s=args.episode_time_s,
                fps=args.fps,
                vcodec=args.vcodec,
            ),
            *build_rtc_argv(
                enabled=args.rtc,
                execution_horizon=args.rtc_execution_horizon,
                max_guidance_weight=args.rtc_max_guidance_weight,
                prefix_attention_schedule=args.rtc_prefix_attention_schedule,
                action_queue_size_to_get_new_actions=args.rtc_action_queue_size_to_get_new_actions,
            ),
            "--enable_episode_outcome_labeling=true",
            "--require_episode_success_label=true",
            "--intervention_state_machine_enabled=true",
            f"--policy_sync_to_teleop={'true' if teleop_argv and args.initial_source == 'vla' else 'false'}",
            "--play_sounds=true",
        ]
        if args.dry_run:
            print(" ".join(sys.argv))
            return
        prepare_lerobot_runtime()
        from lerobot.scripts.lerobot_rlt_record import record

        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()


def run_live(args: argparse.Namespace) -> None:
    set_offline_env()
    setup = load_robot_setup(args.setup_json)
    eval_script = Path(args.eval_script).expanduser()
    if not eval_script.exists():
        raise FileNotFoundError(f"RTC eval script not found: {eval_script}")

    with TemporaryDirectory(prefix="record-live-cal-") as cal_dir:
        stage_follower_calibrations(setup.followers, cal_dir)
        sys.argv = [
            "eval_with_real_robot",
            *build_policy_overrides(
                policy_path=args.policy_path,
                vla_path=args.vla_path,
                rl_token_path=args.rl_token_path,
                phase_mode=args.phase_mode,
                chunk_exec_steps=args.chunk_exec_steps,
            ),
            "--policy.device=cuda",
            "--device=cuda",
            f"--rtc.enabled={'true' if args.rtc else 'false'}",
            f"--rtc.execution_horizon={args.rtc_execution_horizon}",
            f"--rtc.max_guidance_weight={args.rtc_max_guidance_weight}",
            f"--rtc.prefix_attention_schedule={args.rtc_prefix_attention_schedule}",
            *build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, cal_dir),
            f"--task={args.task}",
            f"--duration={args.duration}",
            f"--fps={args.fps}",
        ]
        if args.dry_run:
            print(" ".join(sys.argv))
            return
        prepare_lerobot_runtime()
        runpy.run_path(str(eval_script), run_name="__main__")
