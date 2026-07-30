from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest

from evo_rlt.cli import compare_camera_pose


def _feature_image():
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for index, (x, y) in enumerate(rng.integers([20, 20], [620, 460], size=(200, 2))):
        color = (
            int(80 + index % 175),
            int(60 + index * 3 % 195),
            int(100 + index * 7 % 155),
        )
        cv2.circle(image, (int(x), int(y)), 3 + index % 5, color, -1)
    cv2.putText(
        image,
        "STATIC CAMERA REFERENCE",
        (120, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def test_camera_registration_recovers_target_translation():
    cv2 = pytest.importorskip("cv2")
    reference = _feature_image()
    reference_to_target = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -8.0]])
    target = cv2.warpAffine(reference, reference_to_target, (640, 480))
    mask = compare_camera_pose.build_static_mask(
        480,
        640,
        profile="none",
        excluded_rectangles=[],
    )

    details = compare_camera_pose.register_camera_frames(reference, target, mask)

    assert details.result.rotation_deg == pytest.approx(0.0, abs=0.1)
    assert details.result.scale == pytest.approx(1.0, abs=0.002)
    assert details.result.center_shift_x_px == pytest.approx(-12.0, abs=0.5)
    assert details.result.center_shift_y_px == pytest.approx(8.0, abs=0.5)
    assert details.result.median_reprojection_error_px < 0.5
    assert details.result.confidence == "high"


def test_task1_mask_excludes_workspace_and_robot():
    mask = compare_camera_pose.build_static_mask(
        480,
        640,
        profile="task1_top",
        excluded_rectangles=[(0, 0, 10, 10)],
    )

    assert mask[5, 5] == 0
    assert mask[100, 350] == 0
    assert mask[400, 320] == 0
    assert mask[100, 100] == 255


def test_dataset_wrapper_resolves_single_nested_dataset(tmp_path):
    dataset_root = tmp_path / "wrapper" / "eval_vla_full_123456"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta/info.json").write_text("{}")

    assert compare_camera_pose._resolve_dataset_root(tmp_path / "wrapper") == dataset_root


def test_live_mode_requires_target_camera():
    with pytest.raises(SystemExit):
        compare_camera_pose.parse_args(
            [
                "--reference-image",
                "/tmp/reference.png",
                "--target-image",
                "/tmp/target.png",
                "--live",
            ]
        )


def test_auto_live_display_uses_browser_for_headless_opencv():
    class FakeCV2:
        @staticmethod
        def getBuildInformation():
            return "OpenCV\n  GUI:                           NONE\n"

    args = compare_camera_pose.parse_args(
        [
            "--reference-image",
            "/tmp/reference.png",
            "--target-camera",
            "/dev/video0",
            "--live",
        ]
    )

    assert compare_camera_pose._resolve_live_display(args, FakeCV2) == "browser"


def test_safe_destroy_windows_ignores_headless_error():
    class FakeError(Exception):
        pass

    class FakeCV2:
        error = FakeError

        @staticmethod
        def destroyAllWindows():
            raise FakeError("not implemented")

    compare_camera_pose._safe_destroy_windows(FakeCV2)


def test_browser_preview_serves_html_and_report():
    state = compare_camera_pose._MjpegState()
    try:
        server, thread = compare_camera_pose._start_mjpeg_server(state, "127.0.0.1", 0)
    except PermissionError:
        pytest.skip("test sandbox does not permit binding a loopback socket")
    port = server.server_address[1]
    try:
        state.publish(b"jpeg", {"status": "aligned"})
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
            html = response.read().decode()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/report.json",
            timeout=2,
        ) as response:
            report = json.load(response)

        assert "Evo-RLT camera pose alignment" in html
        assert report == {"status": "aligned"}
    finally:
        state.stop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
