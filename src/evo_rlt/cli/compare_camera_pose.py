from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)
FEATURE_PREFIX = "observation.images."


@dataclass(frozen=True)
class FrameSource:
    kind: str
    description: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RegistrationResult:
    matrix: list[list[float]]
    rotation_deg: float
    scale: float
    scale_error_pct: float
    center_shift_x_px: float
    center_shift_y_px: float
    center_shift_px: float
    median_reprojection_error_px: float
    matches: int
    inliers: int
    inlier_ratio: float
    status: str
    confidence: str


@dataclass(frozen=True)
class RegistrationArtifacts:
    reference: Path
    target: Path
    side_by_side: Path
    raw_false_color: Path
    target_aligned: Path
    aligned_false_color: Path
    matches: Path
    mask: Path
    report: Path


@dataclass
class RegistrationDetails:
    result: RegistrationResult
    matrix: np.ndarray
    reference_keypoints: list[Any]
    target_keypoints: list[Any]
    matches: list[Any]
    inlier_mask: np.ndarray


class _MjpegState:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._version = 0
        self._stopped = False
        self.report_json = b"{}\n"

    def publish(self, jpeg: bytes, report: dict[str, Any]) -> None:
        with self._condition:
            self._jpeg = jpeg
            self.report_json = (
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            ).encode()
            self._version += 1
            self._condition.notify_all()

    def wait_for_frame(
        self,
        version: int,
        timeout_s: float = 1.0,
    ) -> tuple[int, bytes | None, bool]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._version != version or self._stopped,
                timeout=timeout_s,
            )
            return self._version, self._jpeg, self._stopped

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()


def _require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Camera comparison requires OpenCV. Install Evo-RLT's LeRobot environment first."
        ) from error
    return cv2


def _opencv_has_gui(cv2_module) -> bool:
    for line in cv2_module.getBuildInformation().splitlines():
        stripped = line.strip()
        if stripped.startswith("GUI:"):
            value = stripped.split(":", 1)[1].strip().upper()
            return value not in {"", "NONE", "NO"}
    return False


def _safe_destroy_windows(cv2_module) -> None:
    try:
        cv2_module.destroyAllWindows()
    except cv2_module.error:
        pass


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Dataset frame loading requires pyarrow. Run this command in the Evo-RLT environment."
        ) from error
    return pa, pq


def _resolve_dataset_root(root: str | Path) -> Path:
    candidate = Path(root).expanduser().resolve()
    if (candidate / "meta/info.json").is_file():
        return candidate
    matches = sorted(candidate.glob("*/meta/info.json"))
    if len(matches) == 1:
        return matches[0].parents[1]
    if not matches:
        raise FileNotFoundError(
            f"No LeRobot meta/info.json found at {candidate} or one directory below it."
        )
    roots = ", ".join(str(path.parents[1]) for path in matches)
    raise ValueError(f"Dataset wrapper contains multiple datasets; choose one explicitly: {roots}")


def _load_episode_rows(dataset_root: Path) -> dict[str, list[Any]]:
    pa, pq = _require_pyarrow()
    files = sorted((dataset_root / "meta/episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found below {dataset_root}")
    tables = [pq.read_table(path) for path in files]
    return pa.concat_tables(tables).to_pydict()


def _dataset_frame(
    dataset_root: str | Path,
    *,
    camera_key: str,
    episode_index: int,
    frame_offset: int,
) -> tuple[np.ndarray, FrameSource]:
    cv2 = _require_cv2()
    root = _resolve_dataset_root(dataset_root)
    info = json.loads((root / "meta/info.json").read_text())
    fps = float(info["fps"])
    feature_key = camera_key if camera_key.startswith(FEATURE_PREFIX) else f"{FEATURE_PREFIX}{camera_key}"
    if feature_key not in info.get("features", {}):
        available = sorted(
            key for key in info.get("features", {}) if key.startswith(FEATURE_PREFIX)
        )
        raise KeyError(f"Camera feature {feature_key!r} is absent. Available: {available}")

    episodes = _load_episode_rows(root)
    episode_ids = [int(value) for value in episodes["episode_index"]]
    try:
        row_index = episode_ids.index(episode_index)
    except ValueError as error:
        raise IndexError(
            f"Episode {episode_index} is absent from {root}; available range is "
            f"{min(episode_ids)}..{max(episode_ids)}"
        ) from error

    length = int(episodes["length"][row_index])
    if not 0 <= frame_offset < length:
        raise IndexError(
            f"frame_offset={frame_offset} is outside episode {episode_index} length={length}"
        )

    chunk_column = f"videos/{feature_key}/chunk_index"
    file_column = f"videos/{feature_key}/file_index"
    timestamp_column = f"videos/{feature_key}/from_timestamp"
    chunk_index = int(episodes[chunk_column][row_index])
    file_index = int(episodes[file_column][row_index])
    from_timestamp = float(episodes[timestamp_column][row_index])
    video_path = (
        root
        / "videos"
        / feature_key
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )
    video_frame = int(round(from_timestamp * fps)) + frame_offset

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open dataset video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {video_frame} from {video_path}")

    metadata = {
        "dataset_root": str(root),
        "camera_feature": feature_key,
        "episode_index": episode_index,
        "frame_offset": frame_offset,
        "video_path": str(video_path),
        "video_frame": video_frame,
        "fps": fps,
    }
    return frame, FrameSource("dataset", f"{root.name}:ep{episode_index}+{frame_offset}", metadata)


def _image_frame(path: str | Path) -> tuple[np.ndarray, FrameSource]:
    cv2 = _require_cv2()
    image_path = Path(path).expanduser().resolve()
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return frame, FrameSource("image", str(image_path), {"path": str(image_path)})


def _camera_source(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def _open_camera(args: argparse.Namespace):
    cv2 = _require_cv2()
    source = _camera_source(args.target_camera)
    backend = cv2.CAP_V4L2 if isinstance(source, str) and source.startswith("/dev/") else 0
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera: {args.target_camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if args.camera_fourcc:
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*args.camera_fourcc),
        )
    return capture


def _camera_frame(args: argparse.Namespace) -> tuple[np.ndarray, FrameSource]:
    capture = _open_camera(args)
    frame = None
    for _ in range(max(args.camera_warmup_frames, 1)):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not read from camera: {args.target_camera}")
    capture.release()
    assert frame is not None
    metadata = {
        "camera": args.target_camera,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "warmup_frames": args.camera_warmup_frames,
    }
    return frame, FrameSource("camera", args.target_camera, metadata)


def _task1_top_mask(height: int, width: int) -> np.ndarray:
    mask = np.full((height, width), 255, dtype=np.uint8)
    # Movable clip/sticker workspace.
    mask[
        round(20 / 480 * height) : round(230 / 480 * height),
        round(235 / 640 * width) : round(475 / 640 * width),
    ] = 0
    # Robot and loose cables at the bottom of the global view.
    mask[
        round(300 / 480 * height) : height,
        round(130 / 640 * width) : round(520 / 640 * width),
    ] = 0
    return mask


def _parse_rect(value: str) -> tuple[int, int, int, int]:
    try:
        x, y, width, height = (int(part) for part in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Rectangle must be x,y,width,height") from error
    if min(x, y, width, height) < 0 or width == 0 or height == 0:
        raise argparse.ArgumentTypeError("Rectangle values must be non-negative with non-zero size")
    return x, y, width, height


def build_static_mask(
    height: int,
    width: int,
    *,
    profile: str,
    excluded_rectangles: list[tuple[int, int, int, int]],
) -> np.ndarray:
    if profile == "task1_top":
        mask = _task1_top_mask(height, width)
    else:
        mask = np.full((height, width), 255, dtype=np.uint8)
    for x, y, rect_width, rect_height in excluded_rectangles:
        x_to = min(x + rect_width, width)
        y_to = min(y + rect_height, height)
        mask[min(y, height) : y_to, min(x, width) : x_to] = 0
    return mask


def _status(rotation_deg: float, scale: float, center_shift_px: float) -> str:
    scale_error = abs(scale - 1.0)
    if abs(rotation_deg) <= 0.5 and scale_error <= 0.005 and center_shift_px <= 5.0:
        return "aligned"
    if abs(rotation_deg) <= 3.0 and scale_error <= 0.02 and center_shift_px <= 30.0:
        return "close"
    return "misaligned"


def register_camera_frames(
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    ratio_threshold: float = 0.78,
    ransac_threshold_px: float = 3.0,
) -> RegistrationDetails:
    cv2 = _require_cv2()
    if target.shape[:2] != reference.shape[:2]:
        target = cv2.resize(target, (reference.shape[1], reference.shape[0]))
    if mask.shape != reference.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match reference shape {reference.shape[:2]}"
        )

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(
        reference_gray,
        mask,
    )
    target_keypoints, target_descriptors = detector.detectAndCompute(target_gray, mask)
    if reference_descriptors is None or target_descriptors is None:
        raise RuntimeError("Not enough static image texture to compute camera alignment.")

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        target_descriptors,
        reference_descriptors,
        k=2,
    )
    good_matches = [
        nearest
        for nearest, second in pairs
        if nearest.distance < ratio_threshold * second.distance
    ]
    if len(good_matches) < 6:
        raise RuntimeError(
            f"Only {len(good_matches)} static feature matches found; at least 6 are required."
        )

    target_points = np.float32(
        [target_keypoints[match.queryIdx].pt for match in good_matches]
    )
    reference_points = np.float32(
        [reference_keypoints[match.trainIdx].pt for match in good_matches]
    )
    matrix, inliers = cv2.estimateAffinePartial2D(
        target_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
        maxIters=5000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None or inliers is None:
        raise RuntimeError("Static feature registration failed.")
    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 4:
        raise RuntimeError(f"Only {inlier_count} geometrically consistent matches found.")

    a = float(matrix[0, 0])
    b = float(matrix[0, 1])
    scale = float(np.hypot(a, b))
    rotation_deg = float(np.degrees(np.arctan2(b, a)))
    center = np.array([target.shape[1] / 2, target.shape[0] / 2, 1.0])
    mapped_center = matrix @ center
    center_shift = mapped_center - center[:2]
    center_shift_px = float(np.linalg.norm(center_shift))

    predicted = cv2.transform(target_points[:, None, :], matrix)[:, 0, :]
    reprojection_errors = np.linalg.norm(predicted - reference_points, axis=1)
    median_error = float(np.median(reprojection_errors[inlier_mask]))
    inlier_ratio = inlier_count / len(good_matches)
    confidence = (
        "high"
        if inlier_count >= 20 and inlier_ratio >= 0.35
        else "medium"
        if inlier_count >= 8 and inlier_ratio >= 0.15
        else "low"
    )
    result = RegistrationResult(
        matrix=matrix.tolist(),
        rotation_deg=rotation_deg,
        scale=scale,
        scale_error_pct=(scale - 1.0) * 100.0,
        center_shift_x_px=float(center_shift[0]),
        center_shift_y_px=float(center_shift[1]),
        center_shift_px=center_shift_px,
        median_reprojection_error_px=median_error,
        matches=len(good_matches),
        inliers=inlier_count,
        inlier_ratio=inlier_ratio,
        status=_status(rotation_deg, scale, center_shift_px),
        confidence=confidence,
    )
    return RegistrationDetails(
        result=result,
        matrix=matrix,
        reference_keypoints=reference_keypoints,
        target_keypoints=target_keypoints,
        matches=good_matches,
        inlier_mask=inlier_mask,
    )


def _false_color(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros_like(reference)
    canvas[:, :, 1] = target_gray
    canvas[:, :, 2] = reference_gray
    return canvas


def _annotate(frame: np.ndarray, result: RegistrationResult) -> np.ndarray:
    cv2 = _require_cv2()
    output = frame.copy()
    lines = [
        f"status={result.status} confidence={result.confidence}",
        (
            f"target->reference rotation={result.rotation_deg:+.2f} deg "
            f"scale={result.scale:.4f}"
        ),
        (
            f"center shift=({result.center_shift_x_px:+.1f},"
            f"{result.center_shift_y_px:+.1f}) px "
            f"norm={result.center_shift_px:.1f}"
        ),
        (
            f"inliers={result.inliers}/{result.matches} "
            f"median error={result.median_reprojection_error_px:.2f} px"
        ),
        "false color: reference=red, target=green; aligned areas become yellow",
    ]
    for index, line in enumerate(lines):
        y = 26 + index * 25
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def save_registration_artifacts(
    output_dir: str | Path,
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    details: RegistrationDetails,
    reference_source: FrameSource,
    target_source: FrameSource,
) -> RegistrationArtifacts:
    cv2 = _require_cv2()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if target.shape[:2] != reference.shape[:2]:
        target = cv2.resize(target, (reference.shape[1], reference.shape[0]))
    aligned = cv2.warpAffine(
        target,
        details.matrix,
        (reference.shape[1], reference.shape[0]),
    )
    raw_false_color = _annotate(_false_color(reference, target), details.result)
    aligned_false_color = _annotate(_false_color(reference, aligned), details.result)
    matches_image = cv2.drawMatches(
        target,
        details.target_keypoints,
        reference,
        details.reference_keypoints,
        [
            match
            for match, is_inlier in zip(
                details.matches,
                details.inlier_mask,
                strict=True,
            )
            if is_inlier
        ],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    side_by_side = np.hstack([reference, target])

    paths = RegistrationArtifacts(
        reference=root / "reference.png",
        target=root / "target.png",
        side_by_side=root / "side_by_side.png",
        raw_false_color=root / "raw_false_color.png",
        target_aligned=root / "target_aligned.png",
        aligned_false_color=root / "aligned_false_color.png",
        matches=root / "inlier_matches.png",
        mask=root / "static_mask.png",
        report=root / "report.json",
    )
    images = {
        paths.reference: reference,
        paths.target: target,
        paths.side_by_side: side_by_side,
        paths.raw_false_color: raw_false_color,
        paths.target_aligned: aligned,
        paths.aligned_false_color: aligned_false_color,
        paths.matches: matches_image,
        paths.mask: mask,
    }
    for path, image in images.items():
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write image: {path}")

    report = {
        "reference": asdict(reference_source),
        "target": asdict(target_source),
        "registration": asdict(details.result),
        "interpretation": (
            "The reported affine transform maps target image coordinates into the reference image. "
            "Use raw_false_color.png while physically adjusting the camera: reference edges are red, "
            "target edges are green, and overlap appears yellow."
        ),
        "artifacts": {key: str(value) for key, value in asdict(paths).items()},
    }
    paths.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return paths


def _load_reference(args: argparse.Namespace) -> tuple[np.ndarray, FrameSource]:
    if args.reference_dataset_root:
        return _dataset_frame(
            args.reference_dataset_root,
            camera_key=args.camera_key,
            episode_index=args.reference_episode,
            frame_offset=args.reference_frame_offset,
        )
    return _image_frame(args.reference_image)


def _load_target(args: argparse.Namespace) -> tuple[np.ndarray, FrameSource]:
    if args.target_dataset_root:
        return _dataset_frame(
            args.target_dataset_root,
            camera_key=args.camera_key,
            episode_index=args.target_episode,
            frame_offset=args.target_frame_offset,
        )
    if args.target_image:
        return _image_frame(args.target_image)
    return _camera_frame(args)


def _compare_once(
    args: argparse.Namespace,
    reference: np.ndarray,
    reference_source: FrameSource,
    target: np.ndarray,
    target_source: FrameSource,
) -> tuple[RegistrationDetails, RegistrationArtifacts]:
    if target.shape[:2] != reference.shape[:2]:
        cv2 = _require_cv2()
        LOGGER.warning(
            "Resizing target from %s to reference size %s",
            target.shape[:2],
            reference.shape[:2],
        )
        target = cv2.resize(target, (reference.shape[1], reference.shape[0]))
    mask = build_static_mask(
        reference.shape[0],
        reference.shape[1],
        profile=args.mask_profile,
        excluded_rectangles=args.exclude_rect,
    )
    details = register_camera_frames(
        reference,
        target,
        mask,
        ratio_threshold=args.ratio_threshold,
        ransac_threshold_px=args.ransac_threshold_px,
    )
    artifacts = save_registration_artifacts(
        args.output_dir,
        reference,
        target,
        mask,
        details,
        reference_source,
        target_source,
    )
    return details, artifacts


def _print_summary(
    details: RegistrationDetails,
    artifacts: RegistrationArtifacts,
    reference_source: FrameSource,
    target_source: FrameSource,
) -> None:
    result = details.result
    print(f"Reference: {reference_source.description}")
    print(f"Target:    {target_source.description}")
    print(f"Status: {result.status} (confidence={result.confidence})")
    print(
        "Target -> reference transform: "
        f"rotation={result.rotation_deg:+.2f} deg, "
        f"scale={result.scale:.4f} ({result.scale_error_pct:+.2f}%), "
        f"center_shift=({result.center_shift_x_px:+.1f}, "
        f"{result.center_shift_y_px:+.1f}) px, "
        f"norm={result.center_shift_px:.1f} px"
    )
    print(
        f"Static matches: {result.inliers}/{result.matches} inliers, "
        f"median reprojection error={result.median_reprojection_error_px:.2f} px"
    )
    print(f"Physical adjustment view: {artifacts.raw_false_color}")
    print(f"Aligned validation view:  {artifacts.aligned_false_color}")
    print(f"JSON report:              {artifacts.report}")


def _live_report(
    result: RegistrationResult | None,
    *,
    camera: str,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "camera": camera,
        "timestamp": time.time(),
        "registration": asdict(result) if result is not None else None,
        "error": error,
    }


def _start_mjpeg_server(
    state: _MjpegState,
    host: str,
    port: int,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                body = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>Evo-RLT camera alignment</title>"
                    "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}"
                    "main{max-width:1100px;margin:auto;padding:16px}"
                    "img{width:100%;height:auto;border:1px solid #555}"
                    "code{color:#9fe}</style></head><body><main>"
                    "<h2>Evo-RLT camera pose alignment</h2>"
                    "<p>Reference edges are red, live target edges are green, "
                    "and overlap appears yellow. Stop with <code>Ctrl+C</code> "
                    "in the terminal.</p>"
                    "<img src='/stream.mjpg' alt='live camera alignment'>"
                    "<p><a href='/report.json'>Latest JSON metrics</a></p>"
                    "</main></body></html>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/report.json":
                body = state.report_json
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            version = -1
            try:
                while True:
                    version, jpeg, stopped = state.wait_for_frame(version)
                    if stopped:
                        return
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever,
        name="camera-alignment-http",
        daemon=True,
    )
    thread.start()
    return server, thread


def _write_live_latest(
    output_dir: str | Path,
    dashboard: np.ndarray,
    report: dict[str, Any],
) -> tuple[Path, Path]:
    cv2 = _require_cv2()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "live_latest.jpg"
    report_path = root / "live_report.json"
    if not cv2.imwrite(str(image_path), dashboard):
        raise RuntimeError(f"Could not write live preview: {image_path}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return image_path, report_path


def _resolve_live_display(args: argparse.Namespace, cv2_module) -> str:
    if args.live_display != "auto":
        return args.live_display
    return "window" if _opencv_has_gui(cv2_module) else "browser"


def _live_compare(
    args: argparse.Namespace,
    reference: np.ndarray,
    reference_source: FrameSource,
) -> None:
    cv2 = _require_cv2()
    capture = _open_camera(args)
    mask = build_static_mask(
        reference.shape[0],
        reference.shape[1],
        profile=args.mask_profile,
        excluded_rectangles=args.exclude_rect,
    )
    for _ in range(max(args.camera_warmup_frames, 1)):
        ok, _ = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not warm up camera: {args.target_camera}")

    display_mode = _resolve_live_display(args, cv2)
    mjpeg_state = _MjpegState() if display_mode == "browser" else None
    server = None
    server_thread = None
    if mjpeg_state is not None:
        server, server_thread = _start_mjpeg_server(
            mjpeg_state,
            args.live_host,
            args.live_port,
        )
        actual_port = server.server_address[1]
        print(
            f"Open http://{args.live_host}:{actual_port} in a browser for the live alignment view."
        )
        print("Stop the live comparison with Ctrl+C in this terminal.")
    elif display_mode == "window":
        print("Live controls: q/Esc=quit, s=save the current comparison")
    else:
        print("Terminal live mode: stop with Ctrl+C.")

    last_target = None
    last_details = None
    last_preview_write_t = 0.0
    last_terminal_print_t = 0.0
    try:
        while True:
            ok, target = capture.read()
            if not ok:
                raise RuntimeError(f"Camera read failed: {args.target_camera}")
            if target.shape[:2] != reference.shape[:2]:
                target = cv2.resize(target, (reference.shape[1], reference.shape[0]))
            current_result = None
            try:
                details = register_camera_frames(
                    reference,
                    target,
                    mask,
                    ratio_threshold=args.ratio_threshold,
                    ransac_threshold_px=args.ransac_threshold_px,
                )
                dashboard = _annotate(_false_color(reference, target), details.result)
                last_target = target
                last_details = details
                current_result = details.result
                live_report = _live_report(
                    details.result,
                    camera=args.target_camera,
                )
            except RuntimeError as error:
                dashboard = target.copy()
                live_report = _live_report(
                    None,
                    camera=args.target_camera,
                    error=str(error),
                )
                cv2.putText(
                    dashboard,
                    str(error),
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            now = time.monotonic()
            if now - last_preview_write_t >= args.live_update_interval_s:
                image_path, _ = _write_live_latest(
                    args.output_dir,
                    dashboard,
                    live_report,
                )
                last_preview_write_t = now
            if mjpeg_state is not None:
                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    dashboard,
                    [cv2.IMWRITE_JPEG_QUALITY, args.live_jpeg_quality],
                )
                if encoded:
                    mjpeg_state.publish(jpeg.tobytes(), live_report)
            if now - last_terminal_print_t >= args.live_print_interval_s:
                if current_result is not None:
                    result = current_result
                    print(
                        f"\rstatus={result.status:<10} confidence={result.confidence:<6} "
                        f"rot={result.rotation_deg:+6.2f}deg "
                        f"shift={result.center_shift_px:5.1f}px "
                        f"scale_err={result.scale_error_pct:+5.2f}% "
                        f"preview={image_path}",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\rregistration unavailable: {live_report['error']}",
                        end="",
                        flush=True,
                    )
                last_terminal_print_t = now

            if display_mode == "window":
                try:
                    cv2.imshow("Evo-RLT camera pose alignment", dashboard)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as error:
                    raise RuntimeError(
                        "This OpenCV build has no GUI support. Re-run with "
                        "--live-display browser."
                    ) from error
                if key in (ord("q"), 27):
                    break
                if key == ord("s") and last_target is not None and last_details is not None:
                    source = FrameSource(
                        "camera",
                        args.target_camera,
                        {"camera": args.target_camera, "mode": "live"},
                    )
                    artifacts = save_registration_artifacts(
                        args.output_dir,
                        reference,
                        last_target,
                        mask,
                        last_details,
                        reference_source,
                        source,
                    )
                    _print_summary(last_details, artifacts, reference_source, source)
    except KeyboardInterrupt:
        print("\nStopping live camera comparison.")
    finally:
        capture.release()
        if mjpeg_state is not None:
            mjpeg_state.stop()
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=1.0)
        if display_mode == "window":
            _safe_destroy_windows(cv2)

    if last_target is not None and last_details is not None:
        source = FrameSource(
            "camera",
            args.target_camera,
            {"camera": args.target_camera, "mode": f"live-{display_mode}"},
        )
        artifacts = save_registration_artifacts(
            args.output_dir,
            reference,
            last_target,
            mask,
            last_details,
            reference_source,
            source,
        )
        _print_summary(last_details, artifacts, reference_source, source)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a global camera view against a reference LeRobot dataset frame. "
            "The tool estimates a target-to-reference similarity transform from static background."
        )
    )
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-dataset-root")
    reference.add_argument("--reference-image")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-dataset-root")
    target.add_argument("--target-image")
    target.add_argument("--target-camera")
    parser.add_argument("--camera-key", default="top")
    parser.add_argument("--reference-episode", type=int, default=0)
    parser.add_argument("--reference-frame-offset", type=int, default=0)
    parser.add_argument("--target-episode", type=int, default=0)
    parser.add_argument("--target-frame-offset", type=int, default=0)
    parser.add_argument(
        "--mask-profile",
        choices=["task1_top", "none"],
        default="task1_top",
        help="Mask movable task objects and the robot. Use none for another camera layout.",
    )
    parser.add_argument(
        "--exclude-rect",
        action="append",
        type=_parse_rect,
        default=[],
        metavar="X,Y,W,H",
        help="Additional target/reference region to exclude; may be repeated.",
    )
    parser.add_argument("--ratio-threshold", type=float, default=0.78)
    parser.add_argument("--ransac-threshold-px", type=float, default=3.0)
    parser.add_argument("--output-dir", default="camera_pose_compare")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Continuously compare a target camera. Requires --target-camera.",
    )
    parser.add_argument(
        "--live-display",
        choices=["auto", "window", "browser", "terminal"],
        default="auto",
        help=(
            "Live display backend. auto uses an OpenCV window when available and otherwise "
            "serves a browser preview."
        ),
    )
    parser.add_argument("--live-host", default="127.0.0.1")
    parser.add_argument("--live-port", type=int, default=8765)
    parser.add_argument("--live-jpeg-quality", type=int, default=85)
    parser.add_argument("--live-update-interval-s", type=float, default=0.5)
    parser.add_argument("--live-print-interval-s", type=float, default=1.0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--camera-warmup-frames", type=int, default=30)
    args = parser.parse_args(argv)
    if args.live and not args.target_camera:
        parser.error("--live requires --target-camera")
    if not 0.0 < args.ratio_threshold < 1.0:
        parser.error("--ratio-threshold must be within (0, 1)")
    if args.ransac_threshold_px <= 0:
        parser.error("--ransac-threshold-px must be > 0")
    if not 0 <= args.live_port <= 65535:
        parser.error("--live-port must be within [0, 65535]")
    if not 1 <= args.live_jpeg_quality <= 100:
        parser.error("--live-jpeg-quality must be within [1, 100]")
    if args.live_update_interval_s <= 0 or args.live_print_interval_s <= 0:
        parser.error("live update/print intervals must be > 0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    reference, reference_source = _load_reference(args)
    if args.live:
        _live_compare(args, reference, reference_source)
        return
    target, target_source = _load_target(args)
    details, artifacts = _compare_once(
        args,
        reference,
        reference_source,
        target,
        target_source,
    )
    _print_summary(details, artifacts, reference_source, target_source)


if __name__ == "__main__":
    main()
