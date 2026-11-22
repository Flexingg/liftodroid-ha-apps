"""End-to-end HTTP tests for the built-in Home Assistant Ingress Web UI server."""

import json
import urllib.request
from dataclasses import replace

import cv2

from barbell.config import Plate, load_with_options, to_dict
from barbell.web import WebContext, start_server
from tests import synth
from tests.test_colors import bumper_plates_with_tape, tape_scene


def _req(url: str, method: str = "GET", payload=None, raw_bytes: bytes | None = None):
    headers = {}
    data = None
    if raw_bytes is not None:
        data = raw_bytes
    elif payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def test_web_ui_full_setup_test_calibrate_and_save(tmp_path):
    cfg = replace(
        synth.config(),
        plates=bumper_plates_with_tape(),
        reference_image=str(tmp_path / "reference.png"),
        use_shape=True,
        use_color=True,
    )
    current = {"frame": tape_scene(cfg, [])}
    saved_events = []
    published = []
    opts = {"use_shape": False, "plates": "45:450/76", "camera_entity": "camera.gym"}

    ctx = WebContext(
        cfg=cfg,
        cfg_path=tmp_path / "barbell.yaml",
        opts=opts,
        grab_frame=lambda: current["frame"].copy(),
        on_config_saved=lambda new_cfg, new_ref: saved_events.append((new_cfg, new_ref is not None)),
        on_publish_reading=lambda a: published.append(a.reading.detected_weight),
    )
    server = start_server(ctx, host="127.0.0.1", port=0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        # 1. HTML UI loads cleanly with relative API paths (required for HA Ingress).
        code, ct, html = _req(f"{base}/")
        assert code == 200 and "text/html" in ct
        text = html.decode("utf-8")
        assert "LiftoDroid Barbell Camera" in text
        assert "fetch('/api" not in text

        # 2. Save empty-bar reference via Web UI.
        code, _, body = _req(f"{base}/api/reference", "POST", {})
        assert code == 200 and json.loads(body)["ok"]
        assert (tmp_path / "reference.png").exists()

        # 3. Grab a loaded frame (45 + 25 = 185 lb), sample tape color via eyedropper, and test analyze.
        current["frame"] = tape_scene(cfg, [45, 25], tape_width_mm=19.0)
        code, _, body = _req(f"{base}/api/grab", "POST")
        assert code == 200 and json.loads(body)["width"] == synth.W

        red_x = (synth.COLLAR[0] + 38 * synth.PX_PER_MM) / synth.W
        red_y = synth.COLLAR[1] / synth.H
        code, _, body = _req(f"{base}/api/sample_color", "POST", {"x": red_x, "y": red_y})
        assert code == 200 and json.loads(body)["nearest_name"] == "red"

        code, _, body = _req(f"{base}/api/analyze", "POST", {"config": to_dict(cfg), "publish": True})
        res = json.loads(body)
        assert code == 200 and res["reading"]["detected_weight"] == 185
        assert res["reading"]["per_side_breakdown"] == [45, 25]
        assert res["color_counts"]["red"] == 1 and res["color_counts"]["blue"] == 1
        assert published == [185.0]

        # 4. Frame endpoints return valid JPEGs (undistorted, annotated, diagnostic strip, reference).
        for kind in ("undistorted", "annotated", "diagnostic", "reference"):
            code, ct, jpg = _req(f"{base}/api/frame?kind={kind}")
            assert code == 200 and "image/jpeg" in ct and len(jpg) > 500

        # 5. Upload a custom photo directly.
        ok, buf = cv2.imencode(".jpg", tape_scene(cfg, [45, 45]))
        assert ok
        code, _, body = _req(f"{base}/api/upload_frame", "POST", raw_bytes=buf.tobytes())
        assert code == 200 and json.loads(body)["ok"]

        # 6. Save config from Web UI -> writes barbell.yaml, hot-reloads worker, and survives restart
        #    even when old HA Configuration tab options still exist in options.json.
        draft = to_dict(cfg)
        draft["use_shape"] = True
        code, _, body = _req(f"{base}/api/config", "POST", {"config": draft})
        assert code == 200 and json.loads(body)["ok"]
        assert saved_events[-1][0].use_shape is True
        reloaded = load_with_options(tmp_path / "barbell.yaml", opts)
        assert reloaded.use_shape is True
        # If user later explicitly changes an HA option, the new HA option applies.
        reloaded_changed = load_with_options(tmp_path / "barbell.yaml", {**opts, "use_shape": True, "unit": "kg"})
        assert reloaded_changed.unit == "kg"
    finally:
        server.shutdown()
        server.server_close()


def test_two_point_angled_thickness_calibration(tmp_path):
    """2-point calibration from 135 lb and 225 lb on an angled camera solves both sleeve_length_mm and outer_face_mm."""
    plates = (Plate(45, 4, 450, 76), Plate(25, 2, 450, 43))
    true_face = 52.0
    # Start with a deliberately wrong sleeve_length_mm (300 instead of 415) and outer_face_mm=0.
    wrong_cfg = replace(
        synth.config(),
        plates=plates,
        sleeve=replace(synth.sleeve(), sleeve_length_mm=300.0, outer_face_mm=0.0),
    )
    ref = tape_scene(wrong_cfg, [], outer_face_mm=true_face)
    ctx = WebContext(
        cfg=wrong_cfg,
        cfg_path=tmp_path / "barbell.yaml",
        opts={},
        grab_frame=lambda: ref.copy(),
        reference=ref,
    )
    # Sample 1: 135 lb (1 x 45 = 76 mm + 52 mm outer face)
    ctx.set_live_frame(tape_scene(wrong_cfg, [45], outer_face_mm=true_face))
    s1 = ctx.calibrate_thickness(to_dict(wrong_cfg), known_weight=135)
    # Sample 2: 225 lb (2 x 45 = 152 mm + 52 mm outer face)
    ctx.set_live_frame(tape_scene(wrong_cfg, [45, 45], outer_face_mm=true_face))
    s2 = ctx.calibrate_thickness(to_dict(wrong_cfg), known_weight=225)
    solved = ctx.calibrate_thickness(to_dict(wrong_cfg), samples=[s1, s2])

    assert solved["mode"] == "two_point"
    assert abs(solved["sleeve_length_mm"] - synth.SLEEVE_MM) < 12.0
    assert abs(solved["outer_face_mm"] - true_face) < 6.0


def test_overhead_roi_behind_collar_keeps_colors_and_grab_fresh_two_point_calibration(tmp_path):
    """When C is placed ahead of the plates (on the visible sleeve) while the yellow ROI covers the plates behind C,
    the strip pads backward so both rim colors and 2-point calibration (with grab_fresh) work together."""
    plates = bumper_plates_with_tape()
    # Place C 160 px to the right of the real plate collar, while ROI still extends back past the plates.
    x_c = synth.COLLAR[0] + 160
    x_t = synth.COLLAR[0] + int(synth.SLEEVE_MM * synth.PX_PER_MM)
    y0 = synth.COLLAR[1]
    pad = 360
    roi = (
        ((synth.COLLAR[0] - 20) / synth.W, (y0 - pad) / synth.H),
        ((x_t + 20) / synth.W, (y0 - pad) / synth.H),
        ((x_t + 20) / synth.W, (y0 + pad) / synth.H),
        ((synth.COLLAR[0] - 20) / synth.W, (y0 + pad) / synth.H),
    )
    from barbell.config import Sleeve
    sl = Sleeve(axis=((x_c / synth.W, y0 / synth.H), (x_t / synth.W, y0 / synth.H)), roi=roi, sleeve_length_mm=300.0)
    cfg = replace(synth.config(), plates=plates, sleeve=sl, use_shape=True, use_color=True)
    ref = tape_scene(cfg, [], outer_face_mm=40.0)
    live = {"frame": tape_scene(cfg, [45], tape_width_mm=19.0, outer_face_mm=40.0)}

    ctx = WebContext(
        cfg=cfg,
        cfg_path=tmp_path / "barbell.yaml",
        opts={},
        grab_frame=lambda: live["frame"].copy(),
        reference=ref,
    )
    # 1. Analyze sees red tape even though the 45 lb plate sits completely behind C!
    res = ctx.analyze(to_dict(cfg), grab_fresh=True)
    assert res["color_counts"]["red"] == 1
    assert res["measured_stack_mm"] > 50.0

    # 2. 2-point calibration automatically grabs fresh frames as weights change from 135 -> 225.
    s1 = ctx.calibrate_thickness(to_dict(cfg), known_weight=135, grab_fresh=True)
    live["frame"] = tape_scene(cfg, [45, 45], tape_width_mm=19.0, outer_face_mm=40.0)
    s2 = ctx.calibrate_thickness(to_dict(cfg), known_weight=225, grab_fresh=True)
    solved = ctx.calibrate_thickness(to_dict(cfg), samples=[s1, s2])
    assert solved["mode"] == "two_point"
    assert abs(solved["outer_face_mm"] - 40.0) < 8.0
    # After applying calibrated sleeve_length_mm & outer_face_mm, both thickness and color read 225 lb cleanly.
    cal_cfg = replace(cfg, sleeve=replace(sl, sleeve_length_mm=solved["sleeve_length_mm"], outer_face_mm=solved["outer_face_mm"]))
    res2 = ctx.analyze(to_dict(cal_cfg))
    assert res2["reading"]["detected_weight"] == 225
    assert res2["color_counts"]["red"] == 2
