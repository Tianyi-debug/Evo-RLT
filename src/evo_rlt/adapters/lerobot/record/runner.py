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


def prepare_lerobot_runtime(
    *,
    full_pedal_outcome_window_s: float | None = None,
    episode_end_pedal_key: str | None = None,
) -> None:
    register()
    # lerobot_rlt_record imports ChunkACPolicy through this concrete module path.
    import evo_rlt.adapters.lerobot.policies.modeling_rlt_ac as modeling_rlt_ac

    sys.modules["lerobot.policies.rlt.modeling_rlt_ac"] = modeling_rlt_ac
    if full_pedal_outcome_window_s is not None:
        _patch_full_pedal_outcome_listener(full_pedal_outcome_window_s)
    if episode_end_pedal_key is not None:
        _patch_episode_end_pedal_listener(episode_end_pedal_key)


class _CompositeListener:
    def __init__(self, *listeners):
        self._listeners = [listener for listener in listeners if listener is not None]

    def stop(self) -> None:
        for listener in self._listeners:
            stop = getattr(listener, "stop", None)
            if callable(stop):
                stop()


class _EpisodeEndPedalListener:
    def __init__(
        self, listener_cls, events: dict, pedal_key: str, on_press, *args, **kwargs
    ):
        self._events = events
        self._pedal_key = pedal_key
        self._on_press = on_press
        self._listener = listener_cls(self._handle_press, *args, **kwargs)

    def _handle_press(self, key_name: str) -> None:
        if key_name == self._pedal_key:
            self._events["exit_early"] = True
            logging.info("Pedal '%s' pressed -> exit current episode", key_name)
            return
        self._on_press(key_name)

    def start(self) -> bool:
        return self._listener.start()

    def stop(self) -> None:
        self._listener.stop()


def _patch_episode_end_pedal_listener(pedal_key: str) -> None:
    import lerobot.utils.control_utils as control_utils
    import lerobot.utils.pedal_listener as pedal_listener

    if getattr(control_utils._start_pedal_listener, "_evo_rlt_episode_end_pedal", False):
        return

    original_start_pedal_listener = control_utils._start_pedal_listener

    def _start_pedal_listener(events, *args, **kwargs):
        original_pedal_listener = pedal_listener.PedalListener

        def listener_factory(on_press, *listener_args, **listener_kwargs):
            return _EpisodeEndPedalListener(
                original_pedal_listener,
                events,
                pedal_key,
                on_press,
                *listener_args,
                **listener_kwargs,
            )

        pedal_listener.PedalListener = listener_factory
        try:
            return original_start_pedal_listener(events, *args, **kwargs)
        finally:
            pedal_listener.PedalListener = original_pedal_listener

    _start_pedal_listener._evo_rlt_episode_end_pedal = True
    control_utils._start_pedal_listener = _start_pedal_listener


def _patch_full_pedal_outcome_listener(double_tap_window_s: float) -> None:
    import threading

    import lerobot.utils.control_utils as control_utils
    from lerobot.utils.recording_annotations import EPISODE_FAILURE, EPISODE_SUCCESS

    if getattr(control_utils.init_keyboard_listener, "_evo_rlt_full_pedal_outcome", False):
        return

    original_init_keyboard_listener = control_utils.init_keyboard_listener

    def init_keyboard_listener(*args, **kwargs):
        kwargs["episode_success_key"] = None
        kwargs["episode_failure_key"] = None
        keyboard_listener, events = original_init_keyboard_listener(*args, **kwargs)

        try:
            from lerobot.utils.pedal_listener import PedalListener
        except ImportError as error:
            logging.info("Full-episode pedal outcome unavailable: %s", error)
            return keyboard_listener, events

        state = {"timer": None}
        lock = threading.Lock()

        def mark_success() -> None:
            with lock:
                if state["timer"] is None:
                    return
                state["timer"] = None
                events["episode_outcome"] = EPISODE_SUCCESS
                events["exit_early"] = True
            logging.info("Full-episode pedal outcome resolved as success")

        def on_pedal(key_name: str) -> None:
            if key_name != "r":
                return
            with lock:
                timer = state["timer"]
                if timer is None:
                    timer = threading.Timer(double_tap_window_s, mark_success)
                    timer.daemon = True
                    state["timer"] = timer
                    timer.start()
                    logging.info(
                        "Full-episode pedal end pending; tap again within %.1fs for failure",
                        double_tap_window_s,
                    )
                    return
                timer.cancel()
                state["timer"] = None
                events["episode_outcome"] = EPISODE_FAILURE
                events["exit_early"] = True
            logging.info("Full-episode pedal outcome resolved as failure")

        pedal_listener = PedalListener(on_press=on_pedal)
        if not pedal_listener.start():
            return keyboard_listener, events
        return _CompositeListener(keyboard_listener, pedal_listener), events

    init_keyboard_listener._evo_rlt_full_pedal_outcome = True
    control_utils.init_keyboard_listener = init_keyboard_listener


def run_collect(args: argparse.Namespace) -> None:
    set_offline_env()
    setup = load_robot_setup(args.setup_json)
    paths = resolve_run_paths(setup.setup, args.dataset_tag, "eval_vla_rlt_vla")
    configure_logging(paths.log_file, args.log_level)
    remove_existing_dataset(paths.dataset_root)
    teleop_argv = build_teleop_argv(setup.leaders, args.no_teleop)

    if args.policy_path is None:
        raise ValueError("default collection requires --policy-path")
    if not teleop_argv:
        raise ValueError("default VLA-RLT-VLA collection requires leader teleop arms")

    leader_cal_dir = None
    with TemporaryDirectory(prefix="record-collect-") as cal_dir:
        stage_follower_calibrations(setup.followers, cal_dir)
        leader_cal_dir = stage_leader_calibrations(setup.leaders, teleop_argv)
        sys.argv = build_default_collect_record_argv(args, setup, paths, cal_dir, teleop_argv)
        print_collect_summary(args, paths)
        if args.dry_run:
            print("\nDry run argv:")
            print(" ".join(sys.argv))
            return

        prepare_lerobot_runtime(episode_end_pedal_key="space")
        from lerobot.scripts.lerobot_rlt_record import record

        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()


def build_default_collect_record_argv(
    args: argparse.Namespace,
    setup,
    paths,
    cal_dir: str,
    teleop_argv: list[str],
) -> list[str]:
    argv = [
        "record_collect",
        *build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, cal_dir),
        *teleop_argv,
        *build_policy_overrides(
            policy_path=args.policy_path,
            vla_path=args.vla_path,
            rl_token_path=args.rl_token_path,
            phase_mode="manual",
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
        "--rlt.enable=true",
        "--rlt.rl_phase_key_toggles_critical_phase=true",
        f"--rlt.rl_phase_double_tap_window_s={args.double_tap_window_s}",
        *build_rtc_argv(
            enabled=args.rtc,
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            vla_execution_horizon=args.vla_rtc_execution_horizon,
            action_queue_size_to_get_new_actions=args.rtc_action_queue_size_to_get_new_actions,
        ),
        "--intervention_state_machine_enabled=true",
        f"--policy_sync_to_teleop={'true' if teleop_argv else 'false'}",
        f"--vla_ref={'true' if args.vla_ref else 'false'}",
        f"--play_sounds={'true' if args.play_sounds else 'false'}",
    ]
    return argv


def print_collect_summary(args: argparse.Namespace, paths) -> None:
    vla_horizon = args.vla_rtc_execution_horizon or args.rtc_execution_horizon
    print("\nDefault VLA-RLT-VLA collection")
    print(f"Dataset: {paths.dataset_name} -> {paths.dataset_root}")
    print(f"Log: {paths.log_file}")
    print(f"Policy: {args.policy_path}")
    print(f"VLA: {args.vla_path}")
    print(f"RL token: {args.rl_token_path}")
    print(
        "RTC: "
        f"enabled={args.rtc} rlt_horizon={args.rtc_execution_horizon} "
        f"vla_horizon={vla_horizon} guidance={args.rtc_max_guidance_weight} "
        f"schedule={args.rtc_prefix_attention_schedule} "
        f"refill_threshold={args.rtc_action_queue_size_to_get_new_actions}"
    )
    print(
        "Critical phase mode: r starts RLT; r ends the critical phase as success; "
        "r+r inside "
        f"{args.double_tap_window_s:.1f}s marks failure. The episode then continues in VLA mode."
    )


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
        *build_reset_time_argv(args),
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
            vla_execution_horizon=args.vla_rtc_execution_horizon,
            action_queue_size_to_get_new_actions=args.rtc_action_queue_size_to_get_new_actions,
        ),
        "--enable_episode_outcome_labeling=true",
        "--intervention_state_machine_enabled=true",
        (
            "--policy_sync_to_teleop="
            f"{'true' if teleop_argv and args.critical_source in {'rlt', 'vla'} else 'false'}"
        ),
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


def build_reset_time_argv(args: argparse.Namespace) -> list[str]:
    if args.reset_time_s is None:
        return []
    return [f"--dataset.reset_time_s={args.reset_time_s}"]


def print_segment_summary(args: argparse.Namespace, paths) -> None:
    vla_horizon = args.vla_rtc_execution_horizon or args.rtc_execution_horizon
    print(f"\nDataset: {paths.dataset_name} -> {paths.dataset_root}")
    print(f"Log: {paths.log_file}")
    print(f"Initial source: {args.initial_source}")
    print(f"Critical source: {args.critical_source}")
    print(
        "RTC: "
        f"enabled={args.rtc} rlt_horizon={args.rtc_execution_horizon} "
        f"vla_horizon={vla_horizon} "
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
                phase_mode=args.phase_mode,
                chunk_exec_steps=args.chunk_exec_steps,
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
            *build_reset_time_argv(args),
            *build_rtc_argv(
                enabled=args.rtc,
                execution_horizon=args.rtc_execution_horizon,
                max_guidance_weight=args.rtc_max_guidance_weight,
                prefix_attention_schedule=args.rtc_prefix_attention_schedule,
                vla_execution_horizon=args.vla_rtc_execution_horizon,
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
        prepare_lerobot_runtime(
            full_pedal_outcome_window_s=args.double_tap_window_s if args.pedal_outcome else None
        )
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
