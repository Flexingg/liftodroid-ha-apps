"""Built-in Home Assistant Ingress Web UI for visual setup, calibration, and live testing.

Uses only Python's standard library HTTP server so the add-on image needs no extra dependencies.
All frontend paths are relative so the UI works identically behind Home Assistant Ingress
(`/api/hassio_ingress/<token>/...`) and when opened directly.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from . import __version__
from .colors import NAMED_COLORS, sample_hsv_patch
from .config import Config, from_dict, save, to_dict
from .fusion import predicted_stack_mm
from .geometry import Undistorter
from .inventory import Loading, planned_loading
from .ocr import TextReader
from .pipeline import Analysis, Analyzer, diagnostic_strip

LOG = logging.getLogger("barbell.web")
UI_HTML_PATH = Path(__file__).resolve().parent / "web_ui.html"


def _encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG")
    return buf.tobytes()


class WebContext:
    """Thread-safe shared state between the HTTP UI server and the running barbell worker."""

    def __init__(
        self,
        cfg: Config,
        cfg_path: str | Path,
        opts: dict[str, Any],
        grab_frame: Callable[[], np.ndarray | None],
        reader: TextReader | None = None,
        reference: np.ndarray | None = None,
        on_config_saved: Callable[[Config, np.ndarray | None], None] | None = None,
        on_publish_reading: Callable[[Analysis], None] | None = None,
        status_provider: Callable[[], dict[str, Any]] | None = None,
    ):
        self.lock = threading.RLock()
        self.cfg = cfg
        self.cfg_path = Path(cfg_path)
        self.opts = dict(opts)
        self._grab_frame = grab_frame
        self.reader = reader
        self.reference = reference
        self._on_config_saved = on_config_saved
        self._on_publish_reading = on_publish_reading
        self._status_provider = status_provider
        self.last_frame: np.ndarray | None = None
        self.last_frame_ts: float | None = None
        self.last_analysis: Analysis | None = None
        self.last_analysis_cfg: Config | None = None

    def set_live_frame(self, frame: np.ndarray, ts: float | None = None) -> None:
        with self.lock:
            self.last_frame = frame.copy()
            self.last_frame_ts = ts or time.time()

    def set_reference(self, ref: np.ndarray | None) -> None:
        with self.lock:
            self.reference = ref.copy() if ref is not None else None

    def set_config(self, cfg: Config) -> None:
        with self.lock:
            self.cfg = cfg

    def grab(self) -> dict[str, Any]:
        frame = self._grab_frame()
        if frame is None:
            raise RuntimeError(
                "Couldn't grab a frame from the camera. Check camera_url or camera_entity in the Configuration tab."
            )
        ts = time.time()
        with self.lock:
            self.last_frame = frame.copy()
            self.last_frame_ts = ts
            h, w = frame.shape[:2]
        return {"ok": True, "width": w, "height": h, "ts": round(ts, 2)}

    def upload_frame(self, raw_bytes: bytes) -> dict[str, Any]:
        arr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Uploaded file could not be decoded as an image")
        ts = time.time()
        with self.lock:
            self.last_frame = frame
            self.last_frame_ts = ts
            h, w = frame.shape[:2]
        return {"ok": True, "width": w, "height": h, "ts": round(ts, 2)}

    def _merge_draft_config(self, draft: dict[str, Any] | None) -> Config:
        with self.lock:
            base = self.cfg
        if not draft:
            return base
        merged = to_dict(base)
        for key, val in draft.items():
            if val is None and key in ("sleeve_b", "storage_roi", "bench_roi"):
                merged.pop(key, None)
            else:
                merged[key] = val
        cfg = from_dict(merged)
        # Runtime secrets & connection settings always stay from the active app options.
        return replace(
            cfg,
            camera_url=base.camera_url,
            person_entity=base.person_entity,
            reference_image=cfg.reference_image or base.reference_image or "/config/reference.png",
        )

    def state_payload(self) -> dict[str, Any]:
        with self.lock:
            cfg = self.cfg
            frame = self.last_frame
            frame_ts = self.last_frame_ts
            has_ref = self.reference is not None
            last_a = self.last_analysis
            opts = self.opts
        cam_mode = "rtsp" if cfg.camera_url else ("entity" if opts.get("camera_entity") else "none")
        worker_status = self._status_provider() if self._status_provider else {}
        return {
            "version": __version__,
            "config": to_dict(cfg),
            "has_frame": frame is not None,
            "frame_size": [frame.shape[1], frame.shape[0]] if frame is not None else None,
            "frame_ts": round(frame_ts, 2) if frame_ts is not None else None,
            "has_reference": has_ref,
            "reference_path": cfg.reference_image or "/config/reference.png",
            "ocr_available": self.reader is not None,
            "camera_configured": cam_mode != "none",
            "camera_mode": cam_mode,
            "camera_entity": opts.get("camera_entity") or "",
            "person_entity": cfg.person_entity or "",
            "named_colors": {name: info[3] for name, info in NAMED_COLORS.items()},
            "last_reading": last_a.reading.to_payload() if last_a is not None else None,
            "worker_status": worker_status,
        }

    def render_image(self, kind: str) -> bytes | None:
        with self.lock:
            cfg = self.cfg
            frame = self.last_frame
            ref = self.reference
            last_a = self.last_analysis
            last_cfg = self.last_analysis_cfg or cfg
        if kind == "raw":
            return _encode_jpeg(frame) if frame is not None else None
        if kind == "undistorted":
            if frame is None:
                return None
            return _encode_jpeg(Undistorter(cfg.lens)(frame))
        if kind == "reference":
            return _encode_jpeg(ref) if ref is not None else None
        if kind == "annotated":
            return _encode_jpeg(last_a.annotated) if last_a is not None else None
        if kind == "diagnostic":
            if last_a is None:
                return None
            diag = diagnostic_strip(last_a, last_cfg)
            return _encode_jpeg(diag, quality=90) if diag is not None else None
        raise ValueError(f"Unknown frame kind: {kind}")

    def analyze(self, draft: dict[str, Any] | None = None, expected_total: float | None = None,
                publish: bool = False, grab_fresh: bool = False) -> dict[str, Any]:
        if grab_fresh:
            self.grab()
        with self.lock:
            frame = self.last_frame
            ref = self.reference
        if frame is None:
            self.grab()
            with self.lock:
                frame = self.last_frame
                ref = self.reference
        cfg = self._merge_draft_config(draft)
        if cfg.sleeve is None:
            raise ValueError("No sleeve region configured yet. Draw the sleeve ROI and axis first.")
        res = Analyzer(cfg, self.reader, ref).analyze(frame, expected_total=expected_total)
        with self.lock:
            self.last_analysis = res
            self.last_analysis_cfg = cfg
        if publish and self._on_publish_reading is not None:
            self._on_publish_reading(res)
        raw_prof = res.raw_profile_a
        face_mm = cfg.sleeve.outer_face_mm if cfg.sleeve is not None else 0.0
        det_loading = Loading(tuple(res.reading.per_side_breakdown))
        pred_mm = round(predicted_stack_mm(cfg, det_loading, face_mm), 1) if res.reading.detected_weight is not None else None
        return {
            "ok": True,
            "reading": res.reading.to_payload(),
            "measured_stack_mm": round(raw_prof.stack_length_mm, 1) if raw_prof is not None else None,
            "measured_stack_px": round(raw_prof.stack_length_mm * res.strip_a.px_per_mm, 1) if (raw_prof is not None and res.strip_a is not None) else None,
            "strip_length_px": res.strip_a.length if res.strip_a is not None else None,
            "predicted_stack_mm": pred_mm,
            "glare_fraction": round(raw_prof.glare_fraction, 3) if raw_prof is not None else 0.0,
            "slabs": [
                {"start_mm": round(r.start_mm, 1), "end_mm": round(r.end_mm, 1),
                 "width_mm": round(r.width_mm, 1), "height_mm": round(r.height_mm, 1)}
                for r in (raw_prof.runs if raw_prof is not None else ())
            ],
            "color_bands": [b.to_dict() for b in res.colors_a.bands] if res.colors_a is not None else [],
            "color_counts": dict(res.colors_a.count_by_color) if res.colors_a is not None else {},
            "has_diagnostic": res.strip_a is not None,
        }

    def sample_color(self, x: float, y: float) -> dict[str, Any]:
        with self.lock:
            frame = self.last_frame
            cfg = self.cfg
        if frame is None:
            raise ValueError("Grab a frame first before picking a tape color.")
        und = Undistorter(cfg.lens)(frame)
        return {"ok": True, **sample_hsv_patch(und, float(x), float(y))}

    def calibrate_thickness(self, draft: dict[str, Any] | None, known_weight: float | None = None,
                            known_thickness_mm: float | None = None,
                            samples: list[dict[str, float]] | None = None,
                            grab_fresh: bool = False) -> dict[str, Any]:
        """Measure the stack in pixels on the active frame and compute calibrated sleeve_length_mm (and outer_face_mm
        when two different known weights are provided)."""
        cfg = self._merge_draft_config(draft)
        if cfg.sleeve is None:
            raise ValueError("Draw the sleeve ROI and axis before calibrating thickness.")

        # Two-point solve mode: exact scale + angled-camera outer_face_mm from two recorded weights.
        if samples and len(samples) >= 2:
            s1, s2 = samples[0], samples[1]
            t1, p1 = float(s1["true_mm"]), float(s1["measured_px"])
            t2, p2 = float(s2["true_mm"]), float(s2["measured_px"])
            axis_px = float(
                s2.get("axis_length_px") or s2.get("strip_length_px")
                or s1.get("axis_length_px") or s1.get("strip_length_px") or 0.0
            )
            if axis_px <= 0:
                raise ValueError("Missing sleeve axis length in recorded samples. Record both samples again.")
            if abs(t2 - t1) < 5.0:
                raise ValueError(
                    f"Both samples have nearly the same plate thickness ({t1:.0f} mm vs {t2:.0f} mm). "
                    "Use two different weights (e.g. 135 and 225)."
                )
            if abs(p2 - p1) < 2.0:
                raise ValueError(
                    f"Both frames measured almost the same stack width ({p1:.1f} px vs {p2:.1f} px). "
                    "Make sure plates changed between Sample 1 and Sample 2 and the green Plate mask covers the full stack."
                )
            px_per_mm = (p2 - p1) / (t2 - t1)
            if px_per_mm <= 0:
                raise ValueError(
                    f"Heavier weight ({t2:.0f} mm -> {p2:.1f} px) measured shorter than lighter weight "
                    f"({t1:.0f} mm -> {p1:.1f} px). Check collar -> tip direction and that the green Plate mask covers the full stack."
                )
            new_len = round(axis_px / px_per_mm, 1)
            outer_face = round(max(0.0, p1 / px_per_mm - t1), 1)
            return {
                "ok": True,
                "mode": "two_point",
                "sleeve_length_mm": new_len,
                "outer_face_mm": outer_face,
                "px_per_mm": round(px_per_mm, 3),
            }

        if grab_fresh:
            try:
                self.grab()
            except RuntimeError:
                if self.last_frame is None:
                    raise
        with self.lock:
            frame = self.last_frame
            ref = self.reference
        if frame is None:
            raise ValueError("Grab a frame with plates on the bar first.")
        res = Analyzer(cfg, None, ref).analyze(frame)
        with self.lock:
            self.last_analysis = res
            self.last_analysis_cfg = cfg
        raw_prof = res.raw_profile_a
        strip = res.strip_a
        if raw_prof is None or strip is None or raw_prof.stack_length_mm <= 0:
            raise ValueError("No plates were detected in the sleeve mask. Save an empty-bar reference or lower plate_pixel_delta.")
        measured_px = raw_prof.stack_length_mm * strip.px_per_mm

        if known_thickness_mm is None:
            if known_weight is None:
                raise ValueError("Provide either known_weight or known_thickness_mm.")
            plan = planned_loading(cfg, float(known_weight))
            if plan is None or not plan.plates:
                raise ValueError(f"Your plate inventory can't make {known_weight:g} {cfg.unit} (bar is {cfg.bar_weight:g}).")
            known_thickness_mm = sum((cfg.plate_by_weight(w).thickness_mm if cfg.plate_by_weight(w) else 0.0) for w in plan.plates)
            per_side = list(plan.plates)
        else:
            known_thickness_mm = float(known_thickness_mm)
            per_side = []
        if known_thickness_mm <= 0:
            raise ValueError("Known plate thickness must be > 0 mm.")

        face_mm = cfg.sleeve.outer_face_mm
        axis_px = strip.axis_px
        new_len = round((known_thickness_mm + face_mm) * axis_px / measured_px, 1)
        return {
            "ok": True,
            "mode": "single_point",
            "true_mm": round(known_thickness_mm, 1),
            "measured_px": round(measured_px, 1),
            "axis_length_px": round(axis_px, 1),
            "strip_length_px": round(axis_px, 1),
            "sleeve_length_mm": new_len,
            "outer_face_mm": round(face_mm, 1),
            "per_side": per_side,
        }

    def save_reference(self, use_current: bool = False) -> dict[str, Any]:
        if not use_current or self.last_frame is None:
            self.grab()
        with self.lock:
            frame = self.last_frame
            cfg = self.cfg
        if frame is None:
            raise RuntimeError("No frame available to save as reference.")
        und = Undistorter(cfg.lens)(frame)
        path = cfg.reference_image or "/config/reference.png"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, und)
        with self.lock:
            self.reference = und
            if not self.cfg.reference_image:
                self.cfg = replace(self.cfg, reference_image=path)
            updated_cfg = self.cfg
        if self._on_config_saved is not None:
            self._on_config_saved(updated_cfg, und)
        LOG.info("Saved empty-bar reference from Web UI (%s)", path)
        return {"ok": True, "reference_path": path}

    def clear_reference(self) -> dict[str, Any]:
        with self.lock:
            path = self.cfg.reference_image or "/config/reference.png"
            self.reference = None
            updated_cfg = self.cfg
        try:
            if path and Path(path).exists():
                Path(path).unlink()
        except OSError as e:
            LOG.warning("Couldn't remove reference file %s: %s", path, e)
        if self._on_config_saved is not None:
            self._on_config_saved(updated_cfg, None)
        return {"ok": True}

    def save_config(self, draft: dict[str, Any]) -> dict[str, Any]:
        new_cfg = self._merge_draft_config(draft)
        save(new_cfg, self.cfg_path, options_snapshot=self.opts)
        with self.lock:
            self.cfg = new_cfg
            ref = self.reference
        if self._on_config_saved is not None:
            self._on_config_saved(new_cfg, ref)
        LOG.info("Saved config from Web UI to %s (use_shape=%s, use_color=%s, use_ocr=%s)",
                 self.cfg_path, new_cfg.use_shape, new_cfg.use_color, new_cfg.use_ocr)
        return {"ok": True, "config": to_dict(new_cfg)}


def make_handler(ctx: WebContext) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            LOG.debug("HTTP %s - " + fmt, self.address_string(), *args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, data: dict[str, Any]) -> None:
            self._send(code, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

        def _read_body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or "0")
            if n > 20 * 1024 * 1024:
                raise ValueError("Request body too large")
            return self.rfile.read(n) if n > 0 else b""

        def _read_json(self) -> dict[str, Any]:
            raw = self._read_body()
            return json.loads(raw.decode("utf-8")) if raw else {}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path in ("/", "/index.html"):
                    html = UI_HTML_PATH.read_bytes()
                    self._send(HTTPStatus.OK, html, "text/html; charset=utf-8")
                    return
                if path == "/api/state":
                    self._send_json(HTTPStatus.OK, ctx.state_payload())
                    return
                if path == "/api/frame":
                    qs = parse_qs(parsed.query)
                    kind = qs.get("kind", ["undistorted"])[0]
                    jpg = ctx.render_image(kind)
                    if jpg is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"No {kind} image available yet."})
                        return
                    self._send(HTTPStatus.OK, jpg, "image/jpeg")
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            except ValueError as e:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})
            except Exception as e:  # noqa: BLE001
                LOG.exception("GET %s failed", self.path)
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(e)})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/")
            try:
                if path == "/api/grab":
                    self._send_json(HTTPStatus.OK, ctx.grab())
                    return
                if path == "/api/upload_frame":
                    self._send_json(HTTPStatus.OK, ctx.upload_frame(self._read_body()))
                    return
                if path == "/api/analyze":
                    body = self._read_json()
                    exp = body.get("expected_total")
                    res = ctx.analyze(
                        draft=body.get("config"),
                        expected_total=float(exp) if exp not in (None, "") else None,
                        publish=bool(body.get("publish", False)),
                        grab_fresh=bool(body.get("grab_fresh", False)),
                    )
                    self._send_json(HTTPStatus.OK, res)
                    return
                if path == "/api/sample_color":
                    body = self._read_json()
                    self._send_json(HTTPStatus.OK, ctx.sample_color(body["x"], body["y"]))
                    return
                if path == "/api/calibrate_thickness":
                    body = self._read_json()
                    kw = body.get("known_weight")
                    kt = body.get("known_thickness_mm")
                    res = ctx.calibrate_thickness(
                        draft=body.get("config"),
                        known_weight=float(kw) if kw not in (None, "") else None,
                        known_thickness_mm=float(kt) if kt not in (None, "") else None,
                        samples=body.get("samples"),
                        grab_fresh=bool(body.get("grab_fresh", False)),
                    )
                    self._send_json(HTTPStatus.OK, res)
                    return
                if path == "/api/reference":
                    body = self._read_json()
                    self._send_json(HTTPStatus.OK, ctx.save_reference(use_current=bool(body.get("use_current", False))))
                    return
                if path == "/api/config":
                    body = self._read_json()
                    self._send_json(HTTPStatus.OK, ctx.save_config(body.get("config", body)))
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            except ValueError as e:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})
            except RuntimeError as e:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": str(e)})
            except Exception as e:  # noqa: BLE001
                LOG.exception("POST %s failed", self.path)
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(e)})

        def do_DELETE(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/")
            try:
                if path == "/api/reference":
                    self._send_json(HTTPStatus.OK, ctx.clear_reference())
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            except Exception as e:  # noqa: BLE001
                LOG.exception("DELETE %s failed", self.path)
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(e)})

    return Handler


def start_server(ctx: WebContext, host: str = "0.0.0.0", port: int = 8099) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(ctx))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="barbell-web", daemon=True)
    thread.start()
    actual_port = server.server_address[1]
    LOG.info("Web UI listening on http://%s:%d", host, actual_port)
    return server
