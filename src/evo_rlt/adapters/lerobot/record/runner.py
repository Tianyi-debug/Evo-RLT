from __future__ import annotations

import argparse
import json
import logging
import os
import runpy
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from evo_rlt.adapters.lerobot import register
from evo_rlt.adapters.lerobot.record.common import (
    RunPaths,
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


def _resolve_full_recording_target(
    args: argparse.Namespace,
    setup: dict[str, Any],
    dataset_prefix: str,
) -> tuple[RunPaths, int, bool]:
    resume_root_arg = getattr(args, "resume_dataset_root", None)
    if resume_root_arg is None:
        return resolve_run_paths(setup, args.dataset_tag, dataset_prefix), args.num_episodes, False

    dataset_root = Path(resume_root_arg).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not dataset_root.is_dir() or not info_path.is_file():
        raise FileNotFoundError(
            f"Resume dataset must contain meta/info.json: {dataset_root}"
        )

    try:
        info = json.loads(info_path.read_text())
        existing_episodes = int(info["total_episodes"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid resume dataset metadata: {info_path}") from error

    if existing_episodes < 0:
        raise ValueError(f"Invalid total_episodes={existing_episodes} in {info_path}")
    episodes_to_record = args.num_episodes - existing_episodes
    if episodes_to_record <= 0:
        raise ValueError(
            f"Resume target is already reached: dataset has {existing_episodes} episodes, "
            f"target total is {args.num_episodes}."
        )

    dataset_fps = info.get("fps")
    if dataset_fps is not None and float(dataset_fps) != float(args.fps):
        raise ValueError(
            f"Resume FPS mismatch: dataset uses {dataset_fps}, command requested {args.fps}."
        )

    resume_stamp = time.strftime("%H%M%S")
    paths = RunPaths(
        dataset_name=f"local/{dataset_root.name}",
        dataset_root=dataset_root,
        day_dir=dataset_root.parent,
        log_file=dataset_root.parent / f"{dataset_root.name}_resume_{resume_stamp}.log",
    )
    return paths, episodes_to_record, True


def prepare_lerobot_runtime(
    *,
    double_tap_episode_outcome_key: str | None = None,
    double_tap_episode_outcome_window_s: float | None = None,
    intervention_toggle_key: str | None = None,
    skip_policyless_reset_loop: bool = False,
    background_episode_video_encoding: bool = False,
) -> None:
    _patch_keyboard_backend_selection()
    if double_tap_episode_outcome_key is not None:
        if double_tap_episode_outcome_window_s is None:
            raise ValueError("double_tap_episode_outcome_window_s is required")
        _patch_double_tap_episode_outcome_listener(
            double_tap_episode_outcome_window_s,
            double_tap_episode_outcome_key,
        )
    register()
    if intervention_toggle_key is not None:
        _patch_record_intervention_toggle_key(intervention_toggle_key)
    if skip_policyless_reset_loop:
        _patch_skip_policyless_reset_loop()
    if background_episode_video_encoding:
        _patch_append_safe_episode_metadata_flush()
        _patch_save_episode_extra_metadata()


class _CompositeListener:
    def __init__(self, *listeners):
        self._listeners = [listener for listener in listeners if listener is not None]

    def stop(self) -> None:
        for listener in self._listeners:
            stop = getattr(listener, "stop", None)
            if callable(stop):
                stop()


def _event_key_name(key: str | None) -> str | None:
    if key is None:
        return None
    normalized = key.lower()
    return "space" if normalized == " " else normalized


def _keyboard_key_arg(key: str) -> str:
    normalized = _event_key_name(key)
    if normalized is None:
        raise ValueError("key must not be None")
    return " " if normalized == "space" else normalized


def _pynput_key_name(key: Any, keyboard_module: Any) -> str | None:
    if key == keyboard_module.Key.space:
        return "space"
    if key == keyboard_module.Key.enter:
        return "enter"
    if key == keyboard_module.Key.right:
        return "right"
    if key == keyboard_module.Key.left:
        return "left"
    if key == keyboard_module.Key.up:
        return "up"
    if key == keyboard_module.Key.down:
        return "down"
    if key == keyboard_module.Key.esc:
        return "esc"
    if hasattr(key, "char") and key.char:
        return key.char.lower()
    return None


class _DoubleTapEpisodeOutcomeRouter:
    def __init__(
        self,
        events: dict[str, Any],
        outcome_key: str,
        double_tap_window_s: float,
    ) -> None:
        self._events = events
        self._outcome_key = _event_key_name(outcome_key)
        self._double_tap_window_s = double_tap_window_s
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_press(self, key_name: str) -> None:
        if _event_key_name(key_name) != self._outcome_key:
            return
        with self._lock:
            if self._timer is None:
                timer = threading.Timer(self._double_tap_window_s, self._mark_success)
                timer.daemon = True
                self._timer = timer
                timer.start()
                logging.info(
                    "Episode outcome pending; tap '%s' again within %.1fs for failure",
                    self._outcome_key,
                    self._double_tap_window_s,
                )
                return
            self._timer.cancel()
            self._timer = None
            self._events["episode_outcome"] = "failure"
            self._events["exit_early"] = True
        logging.info("Episode outcome resolved as failure")

    def _mark_success(self) -> None:
        with self._lock:
            if self._timer is None:
                return
            self._timer = None
            self._events["episode_outcome"] = "success"
            self._events["exit_early"] = True
        logging.info("Episode outcome resolved as success")

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


PEDAL_TOGGLE_COOLDOWN_S = 0.5


def _start_record_event_pedal_listener(
    events: dict[str, Any],
    key_bindings: dict[str | None, str | None],
    outcome_router: _DoubleTapEpisodeOutcomeRouter | None = None,
):
    from evo_rlt.adapters.lerobot.record.pedal_listener import PedalListener

    event_by_key: dict[str, str] = {}
    for key, event_name in key_bindings.items():
        key_name = _event_key_name(key)
        if key_name is not None and event_name is not None and key_name not in event_by_key:
            event_by_key[key_name] = event_name

    if outcome_router is None and not event_by_key:
        logging.info("No pedal key bindings configured; pedal listener skipped")
        return None

    cooldown_events = {
        "toggle_intervention",
        "toggle_proactive_intervention",
        "toggle_critical_phase",
    }
    last_event_times: dict[str, float] = {}

    def on_press(key_name: str) -> None:
        normalized_key = _event_key_name(key_name)
        if normalized_key is None:
            return
        if outcome_router is not None:
            outcome_router.on_press(normalized_key)
        event_name = event_by_key.get(normalized_key)
        if event_name is None:
            return
        if event_name in cooldown_events:
            now = time.monotonic()
            if now - last_event_times.get(event_name, 0.0) < PEDAL_TOGGLE_COOLDOWN_S:
                return
            last_event_times[event_name] = now
        logging.info("Pedal '%s' pressed -> events[%s] = True", normalized_key, event_name)
        events[event_name] = True

    listener = PedalListener(on_press=on_press)
    if listener.start():
        return listener
    return None


def _start_double_tap_keyboard_listener(
    outcome_key: str,
    router: _DoubleTapEpisodeOutcomeRouter,
):
    import lerobot.utils.control_utils as control_utils

    if control_utils.is_headless():
        return None
    from pynput import keyboard

    normalized_outcome_key = _event_key_name(outcome_key)

    def on_press(key: Any) -> None:
        key_name = _pynput_key_name(key, keyboard)
        if key_name == normalized_outcome_key:
            router.on_press(key_name)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _start_record_event_keyboard_listener(events: dict[str, Any], key_bindings: dict[str | None, str | None]):
    import lerobot.utils.control_utils as control_utils

    if control_utils.is_headless():
        return None
    from pynput import keyboard

    event_by_key = {
        _event_key_name(key): event_name
        for key, event_name in key_bindings.items()
        if _event_key_name(key) is not None and event_name is not None
    }

    def on_press(key: Any) -> None:
        key_name = _pynput_key_name(key, keyboard)
        event_name = event_by_key.get(key_name)
        if event_name is not None:
            events[event_name] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _ensure_record_events(events: dict[str, Any]) -> None:
    events.setdefault("exit_early", False)
    events.setdefault("rerecord_episode", False)
    events.setdefault("stop_recording", False)
    events.setdefault("episode_outcome", None)
    for event_name in [
        "toggle_intervention",
        "toggle_proactive_intervention",
        "toggle_critical_phase",
        "cp_mark_success",
        "cp_mark_failure",
        "start_rl_phase",
        "end_phase_success",
        "end_phase_failure",
    ]:
        events.setdefault(event_name, False)


KEYBOARD_TOGGLE_COOLDOWN_S = 0.5


def _get_keyboard_backend_preference() -> str:
    backend = os.environ.get("LEROBOT_KEYBOARD_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "pynput", "tty"}:
        logging.warning(
            "Unknown LEROBOT_KEYBOARD_BACKEND=%r. Falling back to automatic keyboard backend selection.",
            backend,
        )
        return "auto"
    return backend


def _should_prefer_tty_keyboard(backend: str) -> bool:
    if not sys.stdin.isatty():
        return False
    if backend == "tty":
        return True
    if backend == "pynput":
        return False
    return any(os.environ.get(name) for name in ("SSH_CONNECTION", "SSH_TTY", "TMUX"))


def _build_keyboard_event_bindings(
    *,
    intervention_toggle_key: str | None = "i",
    proactive_intervention_key: str | None = "p",
    critical_phase_toggle_key: str | None = None,
    episode_success_key: str | None = None,
    episode_failure_key: str | None = None,
    cp_success_key: str | None = None,
    cp_failure_key: str | None = None,
    rl_phase_key: str | None = None,
    end_success_key: str | None = None,
    end_failure_key: str | None = None,
    extra_bindings: dict[str | None, str | None] | None = None,
) -> dict[str, str]:
    raw_bindings: dict[str | None, str | None] = {
        intervention_toggle_key: "toggle_intervention",
        proactive_intervention_key: "toggle_proactive_intervention",
        critical_phase_toggle_key: "toggle_critical_phase",
        episode_success_key: "episode_success",
        episode_failure_key: "episode_failure",
        cp_success_key: "cp_mark_success",
        cp_failure_key: "cp_mark_failure",
        rl_phase_key: "start_rl_phase",
        end_success_key: "end_phase_success",
        end_failure_key: "end_phase_failure",
    }
    if extra_bindings:
        raw_bindings.update(extra_bindings)

    event_by_key: dict[str, str] = {}
    for key, event_name in raw_bindings.items():
        key_name = _event_key_name(key)
        if key_name is not None and event_name is not None:
            event_by_key[key_name] = event_name
    return event_by_key


def _dispatch_keyboard_event(
    events: dict[str, Any],
    key_name: str,
    event_by_key: dict[str, str],
    last_event_times: dict[str, float],
    outcome_router: _DoubleTapEpisodeOutcomeRouter | None = None,
) -> None:
    normalized_key = _event_key_name(key_name)
    if normalized_key is None:
        return

    if normalized_key == "right":
        print("Right arrow key pressed. Exiting loop...")
        events["exit_early"] = True
        return
    if normalized_key == "left":
        print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
        events["rerecord_episode"] = True
        events["exit_early"] = True
        return
    if normalized_key == "esc":
        print("Escape key pressed. Stopping data recording...")
        events["stop_recording"] = True
        events["exit_early"] = True
        return

    if outcome_router is not None:
        outcome_router.on_press(normalized_key)

    event_name = event_by_key.get(normalized_key)
    if event_name is None:
        return

    if event_name in {
        "toggle_intervention",
        "toggle_proactive_intervention",
        "toggle_critical_phase",
    }:
        now = time.monotonic()
        if now - last_event_times.get(event_name, 0.0) < KEYBOARD_TOGGLE_COOLDOWN_S:
            return
        last_event_times[event_name] = now

    if event_name == "episode_success":
        print(f"'{normalized_key}' key pressed. Marking episode as success and exiting loop...")
        events["episode_outcome"] = "success"
        events["exit_early"] = True
        return
    if event_name == "episode_failure":
        print(f"'{normalized_key}' key pressed. Marking episode as failure and exiting loop...")
        events["episode_outcome"] = "failure"
        events["exit_early"] = True
        return

    logging.info("Keyboard '%s' pressed -> events[%s] = True", normalized_key, event_name)
    events[event_name] = True


class _EvoRLTTerminalKeyboardListener:
    def __init__(
        self,
        events: dict[str, Any],
        event_by_key: dict[str, str],
        outcome_router: _DoubleTapEpisodeOutcomeRouter | None = None,
    ) -> None:
        self._events = events
        self._event_by_key = event_by_key
        self._outcome_router = outcome_router
        self._fd = sys.stdin.fileno()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._original_termios: list[Any] | None = None
        self._last_event_times: dict[str, float] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="evo-rlt-terminal-keyboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            self._original_termios = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            while not self._stop_event.is_set():
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                key_name = self._read_key_name()
                if key_name is not None:
                    _dispatch_keyboard_event(
                        self._events,
                        key_name,
                        self._event_by_key,
                        self._last_event_times,
                        self._outcome_router,
                    )
        except Exception:
            logging.exception("Terminal keyboard listener stopped unexpectedly.")
        finally:
            if self._original_termios is not None:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_termios)

    def _read_key_name(self) -> str | None:
        chunk = os.read(self._fd, 1)
        if not chunk:
            return None

        if chunk == b"\x1b":
            sequence = bytearray(chunk)
            while True:
                ready, _, _ = select.select([self._fd], [], [], 0.01)
                if not ready:
                    break
                sequence.extend(os.read(self._fd, 1))
                if len(sequence) >= 3 and bytes(sequence[-1:]) in {b"A", b"B", b"C", b"D", b"~"}:
                    break
            sequence_bytes = bytes(sequence)
            if sequence_bytes in {b"\x1b[C", b"\x1bOC"}:
                return "right"
            if sequence_bytes in {b"\x1b[D", b"\x1bOD"}:
                return "left"
            if sequence_bytes in {b"\x1b[A", b"\x1bOA"}:
                return "up"
            if sequence_bytes in {b"\x1b[B", b"\x1bOB"}:
                return "down"
            if sequence_bytes == b"\x1b":
                return "esc"
            return None

        if chunk in {b"\r", b"\n"}:
            return "enter"
        if chunk == b" ":
            return "space"
        try:
            decoded = chunk.decode("utf-8", errors="ignore")
        except Exception:
            return None
        return decoded.lower() if decoded else None


def _start_pynput_record_keyboard_listener(
    events: dict[str, Any],
    event_by_key: dict[str, str],
    outcome_router: _DoubleTapEpisodeOutcomeRouter | None = None,
):
    from pynput import keyboard

    last_event_times: dict[str, float] = {}

    def on_press(key: Any) -> None:
        key_name = _pynput_key_name(key, keyboard)
        if key_name is None:
            return
        _dispatch_keyboard_event(events, key_name, event_by_key, last_event_times, outcome_router)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _start_terminal_record_keyboard_listener(
    events: dict[str, Any],
    event_by_key: dict[str, str],
    outcome_router: _DoubleTapEpisodeOutcomeRouter | None = None,
) -> _EvoRLTTerminalKeyboardListener:
    listener = _EvoRLTTerminalKeyboardListener(events, event_by_key, outcome_router)
    listener.start()
    logging.warning(
        "Using terminal keyboard controls over the current TTY. Keep this terminal focused for "
        "intervention/success/failure/RLT keys to be captured."
    )
    return listener


def _patch_keyboard_backend_selection() -> None:
    import lerobot.utils.control_utils as control_utils

    if getattr(control_utils.init_keyboard_listener, "_evo_rlt_keyboard_backend_selection", False):
        return

    def init_keyboard_listener(*args, **kwargs):
        if args:
            logging.warning("Ignoring positional arguments passed to init_keyboard_listener: %s", args)

        events = kwargs.pop("_evo_rlt_events", None)
        if events is None:
            events = {}
        _ensure_record_events(events)

        outcome_router = kwargs.pop("_evo_rlt_outcome_router", None)
        extra_bindings = kwargs.pop("_evo_rlt_extra_bindings", None)
        event_by_key = _build_keyboard_event_bindings(
            intervention_toggle_key=kwargs.pop("intervention_toggle_key", "i"),
            proactive_intervention_key=kwargs.pop("proactive_intervention_key", "p"),
            critical_phase_toggle_key=kwargs.pop("critical_phase_toggle_key", None),
            episode_success_key=kwargs.pop("episode_success_key", None),
            episode_failure_key=kwargs.pop("episode_failure_key", None),
            cp_success_key=kwargs.pop("cp_success_key", None),
            cp_failure_key=kwargs.pop("cp_failure_key", None),
            rl_phase_key=kwargs.pop("rl_phase_key", None),
            end_success_key=kwargs.pop("end_success_key", None),
            end_failure_key=kwargs.pop("end_failure_key", None),
            extra_bindings=extra_bindings,
        )
        if kwargs:
            logging.warning("Ignoring unsupported keyboard listener kwargs: %s", sorted(kwargs))

        backend = _get_keyboard_backend_preference()
        if _should_prefer_tty_keyboard(backend):
            return _start_terminal_record_keyboard_listener(events, event_by_key, outcome_router), events

        if backend != "tty" and not control_utils.is_headless():
            return _start_pynput_record_keyboard_listener(events, event_by_key, outcome_router), events

        if sys.stdin.isatty():
            return _start_terminal_record_keyboard_listener(events, event_by_key, outcome_router), events

        logging.warning(
            "Headless environment detected without an interactive TTY. Keyboard inputs will not be available."
        )
        return None, events

    init_keyboard_listener._evo_rlt_keyboard_backend_selection = True
    control_utils.init_keyboard_listener = init_keyboard_listener
    backend_module = sys.modules.get("evo_rlt.adapters.lerobot.record.backend")
    if backend_module is not None:
        backend_module.init_keyboard_listener = init_keyboard_listener


def _patch_double_tap_episode_outcome_listener(
    double_tap_window_s: float,
    outcome_key: str,
) -> None:
    import lerobot.utils.control_utils as control_utils

    if getattr(control_utils.init_keyboard_listener, "_evo_rlt_double_tap_episode_outcome", False):
        return

    original_init_keyboard_listener = control_utils.init_keyboard_listener

    def init_keyboard_listener(*args, **kwargs):
        intervention_toggle_key = kwargs.pop("intervention_toggle_key", None)
        proactive_intervention_key = kwargs.pop("proactive_intervention_key", None)
        critical_phase_toggle_key = kwargs.pop("critical_phase_toggle_key", None)
        kwargs.pop("episode_success_key", None)
        kwargs.pop("episode_failure_key", None)
        cp_success_key = kwargs.pop("cp_success_key", None)
        cp_failure_key = kwargs.pop("cp_failure_key", None)
        rl_phase_key = kwargs.pop("rl_phase_key", None)
        end_success_key = kwargs.pop("end_success_key", None)
        end_failure_key = kwargs.pop("end_failure_key", None)
        record_key_bindings = {
            intervention_toggle_key: "toggle_intervention",
            proactive_intervention_key: "toggle_proactive_intervention",
            critical_phase_toggle_key: "toggle_critical_phase",
            cp_success_key: "cp_mark_success",
            cp_failure_key: "cp_mark_failure",
            rl_phase_key: "start_rl_phase",
            end_success_key: "end_phase_success",
            end_failure_key: "end_phase_failure",
        }
        if getattr(original_init_keyboard_listener, "_evo_rlt_keyboard_backend_selection", False):
            events = {}
            _ensure_record_events(events)
            router = _DoubleTapEpisodeOutcomeRouter(events, outcome_key, double_tap_window_s)
            keyboard_listener, events = original_init_keyboard_listener(
                _evo_rlt_events=events,
                _evo_rlt_outcome_router=router,
                _evo_rlt_extra_bindings=record_key_bindings,
            )
            pedal_listener = _start_record_event_pedal_listener(events, record_key_bindings, router)
            return _CompositeListener(keyboard_listener, pedal_listener, router), events

        keyboard_listener, events = original_init_keyboard_listener()
        _ensure_record_events(events)
        router = _DoubleTapEpisodeOutcomeRouter(events, outcome_key, double_tap_window_s)
        pedal_listener = _start_record_event_pedal_listener(events, record_key_bindings, router)
        extra_keyboard_listener = _start_double_tap_keyboard_listener(outcome_key, router)
        record_event_listener = _start_record_event_keyboard_listener(events, record_key_bindings)
        return (
            _CompositeListener(keyboard_listener, pedal_listener, extra_keyboard_listener, record_event_listener, router),
            events,
        )

    init_keyboard_listener._evo_rlt_double_tap_episode_outcome = True
    if getattr(original_init_keyboard_listener, "_evo_rlt_keyboard_backend_selection", False):
        init_keyboard_listener._evo_rlt_keyboard_backend_selection = True
    control_utils.init_keyboard_listener = init_keyboard_listener
    backend_module = sys.modules.get("evo_rlt.adapters.lerobot.record.backend")
    if backend_module is not None:
        backend_module.init_keyboard_listener = init_keyboard_listener


def _patch_record_intervention_toggle_key(toggle_key: str) -> None:
    from evo_rlt.adapters.lerobot.record import backend

    if getattr(backend.init_keyboard_listener, "_evo_rlt_intervention_key", None):
        return

    original_init_keyboard_listener = backend.init_keyboard_listener
    keyboard_toggle_key = _keyboard_key_arg(toggle_key)

    def init_keyboard_listener(*args, **kwargs):
        kwargs["intervention_toggle_key"] = keyboard_toggle_key
        return original_init_keyboard_listener(*args, **kwargs)

    init_keyboard_listener._evo_rlt_intervention_key = _event_key_name(toggle_key)
    backend.init_keyboard_listener = init_keyboard_listener


def _patch_skip_policyless_reset_loop() -> None:
    from evo_rlt.adapters.lerobot.record import backend

    if getattr(backend.record_loop, "_evo_rlt_skip_policyless_reset_loop", False):
        return

    original_record_loop = backend.record_loop

    def record_loop(*args, **kwargs):
        if kwargs.get("policy") is None and kwargs.get("dataset") is None:
            logging.info("Skipping policyless between-episode reset loop.")
            return None
        return original_record_loop(*args, **kwargs)

    record_loop._evo_rlt_skip_policyless_reset_loop = True
    backend.record_loop = record_loop


def _metadata_buffer_to_pydict(metadata_buffer: list[dict], numpy_module: Any) -> dict:
    combined_dict = {}
    for episode_dict in metadata_buffer:
        for key, value in episode_dict.items():
            if key not in combined_dict:
                combined_dict[key] = []
            val = value[0] if isinstance(value, list) else value
            combined_dict[key].append(val.tolist() if isinstance(val, numpy_module.ndarray) else val)
    return combined_dict


def _patch_append_safe_episode_metadata_flush() -> None:
    import lerobot.datasets.dataset_metadata as dataset_metadata

    metadata_cls = dataset_metadata.LeRobotDatasetMetadata
    if getattr(metadata_cls._flush_metadata_buffer, "_evo_rlt_append_safe", False):
        return

    original_flush_metadata_buffer = metadata_cls._flush_metadata_buffer

    def _flush_metadata_buffer(self) -> None:
        metadata_buffer = getattr(self, "_metadata_buffer", None)
        if not metadata_buffer:
            return
        if getattr(self, "_pq_writer", None) is not None:
            return original_flush_metadata_buffer(self)

        first_ep = metadata_buffer[0]
        chunk_idx = first_ep["meta/episodes/chunk_index"][0]
        file_idx = first_ep["meta/episodes/file_index"][0]
        path = self.root / dataset_metadata.DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        if not Path(path).exists():
            return original_flush_metadata_buffer(self)

        incoming_dict = _metadata_buffer_to_pydict(metadata_buffer, dataset_metadata.np)
        incoming_table = dataset_metadata.pa.Table.from_pydict(incoming_dict)
        existing_df = dataset_metadata.pd.read_parquet(path)
        incoming_df = incoming_table.to_pandas()
        combined_df = dataset_metadata.pd.concat([existing_df, incoming_df], ignore_index=True)
        combined_df.to_parquet(path)
        self.latest_episode = metadata_buffer[-1]
        metadata_buffer.clear()

    _flush_metadata_buffer._evo_rlt_append_safe = True
    metadata_cls._flush_metadata_buffer = _flush_metadata_buffer


def _patch_save_episode_extra_metadata() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if getattr(LeRobotDataset.save_episode, "_evo_rlt_save_episode_patch", False):
        return

    original_save_episode = LeRobotDataset.save_episode

    def save_episode_with_extra_metadata(dataset, args, kwargs, extra_episode_metadata):
        if extra_episode_metadata is None:
            return original_save_episode(dataset, *args, **kwargs)

        original_meta_save_episode = dataset.meta.save_episode

        def meta_save_episode(episode_index, episode_length, episode_tasks, episode_stats, episode_metadata):
            merged_episode_metadata = dict(episode_metadata)
            merged_episode_metadata.update(extra_episode_metadata)
            return original_meta_save_episode(
                episode_index,
                episode_length,
                episode_tasks,
                episode_stats,
                merged_episode_metadata,
            )

        dataset.meta.save_episode = meta_save_episode
        try:
            return original_save_episode(dataset, *args, **kwargs)
        finally:
            dataset.meta.save_episode = original_meta_save_episode

    def save_episode(self, *args, **kwargs):
        extra_episode_metadata = kwargs.pop("extra_episode_metadata", None)
        return save_episode_with_extra_metadata(self, args, kwargs, extra_episode_metadata)

    save_episode._evo_rlt_save_episode_patch = True
    LeRobotDataset.save_episode = save_episode


def _validate_distinct_keys(**keys: str | None) -> None:
    seen: dict[str, str] = {}
    for label, key in keys.items():
        normalized = _event_key_name(key)
        if normalized is None:
            continue
        if normalized in seen:
            raise ValueError(f"{label} conflicts with {seen[normalized]} on key {normalized!r}")
        seen[normalized] = label


def _collect_external_episode_outcome_key(args: argparse.Namespace) -> str | None:
    if args.only_critical:
        return None
    return args.rlt_toggle_key


def _episode_outcome_argv(enabled: bool, default_episode_success: str | None = None) -> list[str]:
    if not enabled:
        return []
    argv = [
        "--enable_episode_outcome_labeling=true",
        "--require_episode_success_label=true",
    ]
    if default_episode_success is not None:
        argv.append(f"--default_episode_success={default_episode_success}")
    return argv


def _policy_dataset_prefix(base: str, policy_path: str | None) -> str:
    return f"eval_{base}" if policy_path is not None else base


def _collect_rlt_phase_argv(args: argparse.Namespace) -> list[str]:
    common = [
        "--rlt.enable=true",
        f"--rlt.rl_phase_key={_keyboard_key_arg(args.rlt_toggle_key)}",
        f"--rlt.rl_phase_double_tap_window_s={args.double_tap_window_s}",
    ]
    if args.only_critical:
        return [
            *common,
            "--rlt.skip_prefix_recording=true",
            "--rlt.rl_phase_key_toggles_episode=true",
            f"--rlt.start_in_teleop={'true' if args.start_with_teleop else 'false'}",
        ]
    return [
        *common,
        f"--rlt.start_in_teleop={'true' if args.start_with_teleop else 'false'}",
    ]


def run_collect(args: argparse.Namespace) -> None:
    set_offline_env()
    episode_outcome_key = _collect_external_episode_outcome_key(args)
    validation_keys = {"teleop_toggle_key": args.teleop_toggle_key}
    if args.only_critical:
        validation_keys["rlt_toggle_key"] = args.rlt_toggle_key
    else:
        validation_keys["episode_outcome_key"] = episode_outcome_key
    _validate_distinct_keys(**validation_keys)
    setup = load_robot_setup(args.setup_json)
    apply_manifest_reset_pose(args, setup.setup)
    paths, episodes_to_record, resume = _resolve_full_recording_target(
        args,
        setup.setup,
        "eval_vla_rlt_vla",
    )
    configure_logging(paths.log_file, args.log_level)
    if not resume:
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
        sys.argv = build_default_collect_record_argv(
            args,
            setup,
            paths,
            cal_dir,
            teleop_argv,
            num_episodes=episodes_to_record,
            resume=resume,
        )
        print_collect_summary(
            args,
            paths,
            episodes_to_record=episodes_to_record,
            resume=resume,
        )
        if args.dry_run:
            print("\nDry run argv:")
            print(" ".join(sys.argv))
            return

        prepare_lerobot_runtime(
            double_tap_episode_outcome_key=episode_outcome_key,
            double_tap_episode_outcome_window_s=(
                args.double_tap_window_s if episode_outcome_key is not None else None
            ),
            intervention_toggle_key=args.teleop_toggle_key,
            skip_policyless_reset_loop=(
                args.only_critical
                and not args.start_with_teleop
                and args.reset_time_s is None
            ),
            background_episode_video_encoding=True,
        )
        from evo_rlt.adapters.lerobot.record.backend import record

        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()


def build_default_collect_record_argv(
    args: argparse.Namespace,
    setup,
    paths,
    cal_dir: str,
    teleop_argv: list[str],
    *,
    num_episodes: int | None = None,
    resume: bool = False,
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
            deterministic=getattr(args, "deterministic", None),
        ),
        *build_dataset_argv(
            dataset_name=paths.dataset_name,
            dataset_root=paths.dataset_root,
            task=args.task,
            num_episodes=args.num_episodes if num_episodes is None else num_episodes,
            episode_time_s=args.episode_time_s,
            fps=args.fps,
            vcodec=args.vcodec,
        ),
        *(["--resume=true"] if resume else []),
        *build_reset_time_argv(args),
        *build_auto_reset_pose_argv(args),
        *_collect_rlt_phase_argv(args),
        *build_rtc_argv(
            enabled=args.rtc,
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            vla_execution_horizon=args.vla_rtc_execution_horizon,
            action_queue_size_to_get_new_actions=args.rtc_action_queue_size_to_get_new_actions,
        ),
        *_episode_outcome_argv(
            args.only_critical or _collect_external_episode_outcome_key(args) is not None,
            getattr(args, "default_episode_success", None),
        ),
        "--intervention_state_machine_enabled=true",
        f"--proactive_intervention_key={getattr(args, 'proactive_intervention_key', 'p')}",
        f"--policy_sync_to_teleop={'true' if teleop_argv else 'false'}",
        f"--vla_ref={'true' if args.vla_ref else 'false'}",
        f"--play_sounds={'true' if args.play_sounds else 'false'}",
    ]
    return argv


def print_collect_summary(
    args: argparse.Namespace,
    paths,
    *,
    episodes_to_record: int | None = None,
    resume: bool = False,
) -> None:
    vla_horizon = args.vla_rtc_execution_horizon or args.rtc_execution_horizon
    print("\nDefault VLA-RLT-VLA collection")
    print(f"Dataset: {paths.dataset_name} -> {paths.dataset_root}")
    print(f"Log: {paths.log_file}")
    print(f"Policy: {args.policy_path}")
    print(f"VLA: {args.vla_path}")
    print(f"RL token: {args.rl_token_path}")
    if resume:
        print(
            "Resume: "
            f"target_total={args.num_episodes} "
            f"episodes_to_record={episodes_to_record}"
        )
    if args.auto_reset_pose:
        print(
            "Episode reset: "
            f"pose={args.reset_pose_path or 'default cached pose'} "
            f"duration={args.reset_pose_duration_s:.1f}s "
            f"environment_window={args.reset_time_s or 0}s"
        )
    else:
        print("Episode reset: automatic pose return disabled")
    print(
        "RTC: "
        f"enabled={args.rtc} rlt_horizon={args.rtc_execution_horizon} "
        f"vla_horizon={vla_horizon} guidance={args.rtc_max_guidance_weight} "
        f"schedule={args.rtc_prefix_attention_schedule} "
        f"refill_threshold={args.rtc_action_queue_size_to_get_new_actions}"
    )
    if args.only_critical:
        print(
            f"Recording mode: RLT critical segment only. {args.rlt_toggle_key} enters RLT and starts recording; "
            f"the next {args.rlt_toggle_key} saves success; "
            f"{args.rlt_toggle_key}+{args.rlt_toggle_key} inside "
            f"{args.double_tap_window_s:.1f}s saves failure."
        )
    else:
        print(
            f"Recording mode: full trajectory. Recording starts immediately; "
            f"{args.rlt_toggle_key} saves success after {args.double_tap_window_s:.1f}s; "
            f"double-tap {args.rlt_toggle_key} saves failure."
        )
    print(f"Episode starts with: {'teleop' if args.start_with_teleop else 'VLA'}")
    if args.only_critical:
        rlt_control = f"{args.rlt_toggle_key}=start/end RLT critical recording"
    else:
        rlt_control = f"{args.rlt_toggle_key}=save full-episode outcome"
    print(
        "Controls: "
        f"{rlt_control}, "
        f"{args.teleop_toggle_key}=toggle teleop intervention"
        f", {getattr(args, 'proactive_intervention_key', 'p')}=planned/proactive intervention"
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

        prepare_lerobot_runtime(background_episode_video_encoding=True)
        from evo_rlt.adapters.lerobot.record.backend import record

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
        f"--rlt.two_stage_intervention={'true' if args.two_stage_intervention else 'false'}",
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
        f"--proactive_intervention_key={getattr(args, 'proactive_intervention_key', 'p')}",
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
            deterministic=getattr(args, "deterministic", None),
        )
    return build_policy_overrides(
        policy_path=args.policy_path,
        vla_path=args.vla_path,
        rl_token_path=args.rl_token_path,
        deterministic=getattr(args, "deterministic", None),
    )


def build_reset_time_argv(args: argparse.Namespace) -> list[str]:
    reset_time_s = getattr(args, "reset_time_s", None)
    if reset_time_s is None:
        return []
    return [f"--dataset.reset_time_s={reset_time_s}"]


def build_display_data_argv(display_data: bool) -> list[str]:
    if not display_data:
        return []
    return ["--display_data=true"]


def build_auto_reset_pose_argv(args: argparse.Namespace) -> list[str]:
    if not getattr(args, "auto_reset_pose", False):
        return ["--auto_reset_pose=false"]
    argv = [
        "--auto_reset_pose=true",
        f"--reset_pose_duration_s={args.reset_pose_duration_s}",
        f"--reset_pose_recapture={'true' if args.reset_pose_recapture else 'false'}",
    ]
    if args.reset_pose_path is not None:
        argv.append(f"--reset_pose_path={args.reset_pose_path}")
    return argv


def apply_manifest_reset_pose(args: argparse.Namespace, setup: dict[str, Any]) -> None:
    reset_pose = setup.get("reset_pose")
    if not isinstance(reset_pose, dict):
        return

    locked = bool(reset_pose.get("locked", False))
    pose_path_raw = reset_pose.get("path")
    if locked and not pose_path_raw:
        raise ValueError("Locked manifest reset_pose requires a non-empty 'path'.")
    if not pose_path_raw:
        return

    pose_path = Path(str(pose_path_raw)).expanduser()
    if locked:
        if not pose_path.is_file():
            raise FileNotFoundError(f"Locked manifest reset pose does not exist: {pose_path}")
        args.reset_pose_path = str(pose_path)
        args.reset_pose_recapture = False
        print(f"Locked episode reset pose: {pose_path}", flush=True)
    elif args.reset_pose_path is None:
        args.reset_pose_path = str(pose_path)


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
    apply_manifest_reset_pose(args, setup.setup)
    dataset_prefix = _policy_dataset_prefix(f"{args.initial_source}_full", args.policy_path)
    paths, episodes_to_record, resume = _resolve_full_recording_target(
        args,
        setup.setup,
        dataset_prefix,
    )
    configure_logging(paths.log_file, args.log_level)
    if not resume:
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
                deterministic=args.deterministic,
            ),
            *build_dataset_argv(
                dataset_name=paths.dataset_name,
                dataset_root=paths.dataset_root,
                task=args.task,
                num_episodes=episodes_to_record,
                episode_time_s=args.episode_time_s,
                fps=args.fps,
                vcodec=args.vcodec,
            ),
            *(["--resume=true"] if resume else []),
            *build_reset_time_argv(args),
            *build_auto_reset_pose_argv(args),
            *build_display_data_argv(args.display_data),
            f"--rlt.intervention_action_blend_time_s={args.intervention_action_blend_time_s}",
            f"--rlt.two_stage_intervention={'true' if args.two_stage_intervention else 'false'}",
            *build_rtc_argv(
                enabled=args.rtc,
                execution_horizon=args.rtc_execution_horizon,
                max_guidance_weight=args.rtc_max_guidance_weight,
                prefix_attention_schedule=args.rtc_prefix_attention_schedule,
                vla_execution_horizon=args.vla_rtc_execution_horizon,
                action_queue_size_to_get_new_actions=args.rtc_action_queue_size_to_get_new_actions,
            ),
            *_episode_outcome_argv(True, args.default_episode_success),
            "--intervention_state_machine_enabled=true",
            f"--proactive_intervention_key={getattr(args, 'proactive_intervention_key', 'p')}",
            f"--policy_sync_to_teleop={'true' if teleop_argv and args.initial_source == 'vla' else 'false'}",
            "--play_sounds=true",
        ]
        if args.dry_run:
            print(" ".join(sys.argv))
            return
        prepare_lerobot_runtime(
            double_tap_episode_outcome_key=(
                args.episode_outcome_key if args.pedal_outcome else None
            ),
            double_tap_episode_outcome_window_s=(
                args.double_tap_window_s if args.pedal_outcome else None
            ),
            background_episode_video_encoding=True,
        )
        from evo_rlt.adapters.lerobot.record.backend import record

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
                deterministic=args.deterministic,
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
