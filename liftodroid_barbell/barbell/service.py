"""The always-on worker. Idle (camera closed) until the phone arms it for a workout; then reads the sleeve every
time it settles and publishes the result to Home Assistant over MQTT discovery.

MQTT (base topic liftodroid/barbell):
  status            online / offline (last will)
  state             retained JSON Reading: detected_weight, confidence, per_side_breakdown, raw_ocr_matches,
                    frame_timestamp, needs_confirmation, status, reasons, alternatives, last_motion_ts, context
  frame             retained JPEG: the annotated frame of the last reading
  motion, armed     ON / OFF
  enabled           ON / OFF (state);  cmd/enabled  ON / OFF (set)
  cmd/arm           JSON {"ttl_s", "set_id", "exercise", "expected", "unit", "bar", "plates": [{"weight","count"}]}
  cmd/disarm        anything
  cmd/detect        anything: read the sleeve on the next quiet frame
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import __version__
from .config import Config, load, with_context
from .geometry import polygon_mask
from .motion import GateEvent, MotionGate
from .pipeline import Analyzer

LOG = logging.getLogger("barbell")
BASE = "liftodroid/barbell"
NODE = "liftodroid_barbell"


# ------------------------------------------------------------------------------------------------ arming (pure)

@dataclass
class ArmState:
    """Armed until [until] (wall seconds). The phone re-arms on every set, so a phone that goes away disarms us."""

    until: float = 0.0
    context: dict = field(default_factory=dict)

    def arm(self, payload: dict, now: float, default_ttl: float = 1200.0) -> None:
        ttl = float(payload.get("ttl_s", default_ttl))
        self.until = now + max(10.0, min(ttl, 4 * 3600.0))
        ctx = {k: payload[k] for k in ("set_id", "exercise", "expected", "unit", "bar", "plates") if k in payload}
        # A new set: what the last set read is no longer this set's reading.
        self.context = ctx

    def disarm(self) -> None:
        self.until = 0.0

    def armed(self, now: float) -> bool:
        return now < self.until


def discovery(prefix: str, unit: str = "lb") -> list[tuple[str, dict]]:
    dev = {"identifiers": [NODE], "name": "Barbell Camera", "manufacturer": "LiftoDroid", "model": "Barbell weight reader", "sw_version": __version__}
    avail = {"availability_topic": f"{BASE}/status"}
    return [
        (f"{prefix}/sensor/{NODE}/weight/config", {
            "name": "Weight", "unique_id": f"{NODE}_weight", "state_topic": f"{BASE}/state",
            "value_template": "{{ value_json.detected_weight if value_json.detected_weight is not none else 'unknown' }}",
            "json_attributes_topic": f"{BASE}/state", "unit_of_measurement": unit, "device_class": "weight",
            "icon": "mdi:weight-lifter", "device": dev, **avail,
        }),
        (f"{prefix}/sensor/{NODE}/confidence/config", {
            "name": "Confidence", "unique_id": f"{NODE}_confidence", "state_topic": f"{BASE}/state",
            "value_template": "{{ (value_json.confidence * 100) | round(0) }}", "unit_of_measurement": "%",
            "entity_category": "diagnostic", "device": dev, **avail,
        }),
        (f"{prefix}/binary_sensor/{NODE}/motion/config", {
            "name": "Sleeve motion", "unique_id": f"{NODE}_motion", "state_topic": f"{BASE}/motion",
            "device_class": "motion", "device": dev, **avail,
        }),
        (f"{prefix}/binary_sensor/{NODE}/armed/config", {
            "name": "Watching", "unique_id": f"{NODE}_armed", "state_topic": f"{BASE}/armed",
            "device_class": "running", "entity_category": "diagnostic", "device": dev, **avail,
        }),
        (f"{prefix}/image/{NODE}/frame/config", {
            "name": "Last reading", "unique_id": f"{NODE}_frame", "image_topic": f"{BASE}/frame",
            "content_type": "image/jpeg", "device": dev, **avail,
        }),
        (f"{prefix}/button/{NODE}/detect/config", {
            "name": "Read now", "unique_id": f"{NODE}_detect", "command_topic": f"{BASE}/cmd/detect",
            "icon": "mdi:camera-iris", "device": dev, **avail,
        }),
        (f"{prefix}/switch/{NODE}/enabled/config", {
            "name": "Enabled", "unique_id": f"{NODE}_enabled", "command_topic": f"{BASE}/cmd/enabled",
            "state_topic": f"{BASE}/enabled", "entity_category": "config", "device": dev, **avail,
        }),
    ]


# ------------------------------------------------------------------------------------------------ frames

class RtspCapture:
    """Keeps the stream drained in a thread so the frame we analyse is the newest, not one from seconds ago."""

    def __init__(self, url: str):
        self.url = url
        self._cap = None
        self._latest: tuple[float, np.ndarray] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                LOG.warning("Camera stream didn't open; retrying in 5 s")
                self._stop.wait(5)
                continue
            self._cap = cap
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    LOG.warning("Camera stream dropped; reconnecting")
                    break
                with self._lock:
                    self._latest = (time.time(), frame)
            cap.release()
            self._cap = None

    def latest(self) -> tuple[float, np.ndarray] | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._latest = None


class SnapshotCapture:
    """Fallback: Home Assistant's camera proxy for a camera entity (slower; fine at 1-2 fps)."""

    def __init__(self, entity_id: str, core: "CoreApi"):
        self.entity_id = entity_id
        self.core = core

    def open(self) -> None:
        pass

    def latest(self) -> tuple[float, np.ndarray] | None:
        data = self.core.get_bytes(f"/camera_proxy/{self.entity_id}")
        if not data:
            return None
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return (time.time(), img) if img is not None else None

    def close(self) -> None:
        pass


class CoreApi:
    """Home Assistant's REST API through the Supervisor (the app token needs no user setup)."""

    def __init__(self, token: str | None = None, base: str = "http://supervisor/core/api"):
        self.token = token or os.environ.get("SUPERVISOR_TOKEN", "")
        self.base = base

    def _req(self, path: str) -> bytes | None:
        if not self.token:
            return None
        req = urllib.request.Request(self.base + path, headers={"Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - a missing entity or a blip must never stop the worker
            LOG.debug("HA API %s failed: %s", path, e)
            return None

    def get_bytes(self, path: str) -> bytes | None:
        return self._req(path)

    def state(self, entity_id: str) -> str | None:
        raw = self._req(f"/states/{entity_id}")
        return json.loads(raw).get("state") if raw else None


def supervisor_mqtt() -> dict[str, Any] | None:
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        return None
    req = urllib.request.Request("http://supervisor/services/mqtt", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read()).get("data")
    except Exception as e:  # noqa: BLE001
        LOG.warning("Couldn't read MQTT settings from the Supervisor: %s", e)
        return None


# ------------------------------------------------------------------------------------------------ worker

class Worker:
    def __init__(self, cfg: Config, analyzer: Analyzer, capture_factory: Callable[[], Any], core: CoreApi,
                 publish: Callable[[str, Any, bool], None], clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.analyzer = analyzer
        self.capture_factory = capture_factory
        self.capture = None
        self.core = core
        self.publish = publish
        self.clock = clock
        self.arm = ArmState()
        self.enabled = True
        self.gate = MotionGate(cfg.thresholds.motion_pixel_delta, cfg.thresholds.motion_fraction, cfg.thresholds.settle_s)
        self._motion = None
        self._armed_published = None
        self._pending = False
        self._last_analysis = 0.0
        self._mask = None

    # Commands arrive on the MQTT thread; the loop reads these flags.
    def on_command(self, topic: str, payload: bytes) -> None:
        now = self.clock()
        name = topic.rsplit("/", 1)[-1]
        if name == "arm":
            try:
                body = json.loads(payload or b"{}")
            except ValueError:
                body = {}
            new_set = body.get("set_id") != self.arm.context.get("set_id")
            self.arm.arm(body, now)
            if new_set:
                self.gate.force()  # read the bar as it is for this set, even if nothing moves
        elif name == "disarm":
            self.arm.disarm()
        elif name == "detect":
            self.gate.force()
            if not self.arm.armed(now):
                self.arm.arm({"ttl_s": 60, **self.arm.context}, now)
        elif name == "enabled":
            self.enabled = payload.strip().upper() in (b"ON", b"1", b"TRUE")
            self.publish(f"{BASE}/enabled", "ON" if self.enabled else "OFF", True)

    def step(self) -> None:
        now = self.clock()
        armed = self.enabled and self.arm.armed(now)
        if armed != self._armed_published:
            self._armed_published = armed
            self.publish(f"{BASE}/armed", "ON" if armed else "OFF", True)
        if not armed:
            if self.capture is not None:
                self.capture.close()
                self.capture = None
                self.gate.reset()
            return
        if self.capture is None:
            self.capture = self.capture_factory()
            self.capture.open()
        got = self.capture.latest()
        if got is None:
            return
        t, frame = got
        if t - self._last_analysis < 1.0 / max(self.cfg.analysis_fps, 0.1):
            return
        self._last_analysis = t
        und = self.analyzer.undistort(frame)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        if self._mask is None or self._mask.shape != gray.shape:
            self._mask = polygon_mask(gray.shape, self.cfg.sleeve.roi)
        ev = self.gate.feed(t, gray, self._mask)
        moving = ev == GateEvent.MOVING
        if moving != self._motion:
            self._motion = moving
            self.publish(f"{BASE}/motion", "ON" if moving else "OFF", True)
        if ev == GateEvent.SETTLED:
            self._pending = True
        if self._pending and not moving:
            if self.cfg.person_entity and self.core.state(self.cfg.person_entity) == "on":
                return  # someone is in the rack: wait until they step out
            self._pending = False
            self.read(frame, t)

    def read(self, frame: np.ndarray, t: float) -> None:
        ctx = self.arm.context
        cfg = with_context(self.cfg, ctx.get("unit"), ctx.get("bar"), ctx.get("plates"))
        analyzer = self.analyzer if cfg is self.cfg else self.analyzer.with_config(cfg)
        a = analyzer.analyze(frame, expected_total=ctx.get("expected"))
        r = replace(a.reading, frame_timestamp=t, last_motion_ts=self.gate.last_motion_ts, context=ctx or None)
        ok, jpg = cv2.imencode(".jpg", a.annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.publish(f"{BASE}/frame", jpg.tobytes(), True)
        self.publish(f"{BASE}/state", json.dumps(r.to_payload()), True)
        LOG.info("Read %s %s (%.0f%%) %s", r.detected_weight, r.unit, r.confidence * 100, r.status)


def ensure_config_file(cfg_path: str | Path, example_path: Path | None = None) -> bool:
    """Create [cfg_path] from the bundled barbell.example.yaml if it doesn't exist yet. Returns True if created."""
    dest = Path(cfg_path)
    if dest.exists():
        return False
    src = example_path or (Path(__file__).resolve().parents[1] / "barbell.example.yaml")
    if not src.exists():
        raise FileNotFoundError(f"Default config template not found at {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return True


def main() -> None:
    import paho.mqtt.client as mqtt

    opts = json.load(open("/data/options.json")) if os.path.exists("/data/options.json") else {}
    logging.basicConfig(level=getattr(logging, str(opts.get("log_level", "info")).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg_path = opts.get("config_file", "/config/barbell.yaml")
    if ensure_config_file(cfg_path):
        LOG.info("Created default %s from barbell.example.yaml; edit it to calibrate your camera, sleeve, and plates (see the Documentation tab).", cfg_path)
    cfg = load(cfg_path)
    if opts.get("camera_url"):
        cfg = replace(cfg, camera_url=opts["camera_url"])
    if opts.get("person_entity"):
        cfg = replace(cfg, person_entity=opts["person_entity"])
    reference = cv2.imread(cfg.reference_image) if cfg.reference_image and os.path.exists(cfg.reference_image) else None
    reader = None
    try:
        from .ocr import RapidOcrReader

        reader = RapidOcrReader()
    except Exception as e:  # noqa: BLE001
        LOG.warning("OCR unavailable (%s): reading plate sizes only", e)
    analyzer = Analyzer(cfg, reader, reference)
    core = CoreApi()
    camera_entity = opts.get("camera_entity")
    if not cfg.camera_url and not camera_entity:
        LOG.warning("No camera_url or camera_entity configured yet; set one in the app's Configuration tab.")
    factory = (lambda: RtspCapture(cfg.camera_url)) if cfg.camera_url else (lambda: SnapshotCapture(camera_entity, core))

    m = opts.get("mqtt_host") and {"host": opts["mqtt_host"], "port": opts.get("mqtt_port", 1883),
                                   "username": opts.get("mqtt_username"), "password": opts.get("mqtt_password")} or supervisor_mqtt()
    if not m:
        raise SystemExit("No MQTT broker: install the Mosquitto broker app or set mqtt_host")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=NODE)
    if m.get("username"):
        client.username_pw_set(m["username"], m.get("password"))
    client.will_set(f"{BASE}/status", "offline", retain=True)

    def publish(topic: str, payload: Any, retain: bool) -> None:
        client.publish(topic, payload, qos=1, retain=retain)

    worker = Worker(cfg, analyzer, factory, core, publish)
    prefix = opts.get("discovery_prefix", "homeassistant")

    def on_connect(c, _u, _f, rc, _p=None):
        LOG.info("MQTT connected (%s)", rc)
        for topic, payload in discovery(prefix, cfg.unit):
            c.publish(topic, json.dumps(payload), qos=1, retain=True)
        c.publish(f"{BASE}/status", "online", qos=1, retain=True)
        c.publish(f"{BASE}/enabled", "ON" if worker.enabled else "OFF", qos=1, retain=True)
        c.subscribe(f"{BASE}/cmd/#", qos=1)

    client.on_connect = on_connect
    client.on_message = lambda _c, _u, msg: worker.on_command(msg.topic, msg.payload)
    client.connect(m["host"], int(m.get("port", 1883)))
    client.loop_start()
    LOG.info("Barbell camera ready (v%s); idle until a workout arms it", __version__)
    while True:
        try:
            worker.step()
        except Exception:  # noqa: BLE001 - log, back off, keep serving
            LOG.exception("Reading failed")
            time.sleep(2)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
