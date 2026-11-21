import json

from barbell.pipeline import Analyzer
from barbell.service import BASE, ArmState, Worker, discovery
from tests import synth


class FakeCapture:
    def __init__(self, frames):
        self.frames = frames  # callable: t -> frame
        self.opened = False
        self.closed = False
        self.t = 0.0

    def open(self):
        self.opened = True

    def latest(self):
        return (self.t, self.frames(self.t))

    def close(self):
        self.closed = True


class FakeCore:
    def __init__(self):
        self.person = "off"

    def state(self, entity_id):
        return self.person


def make(frames, person_entity=None):
    from dataclasses import replace

    cfg = replace(synth.config(), person_entity=person_entity, analysis_fps=2.0)
    cap = FakeCapture(frames)
    clock = {"t": 0.0}
    published = []
    w = Worker(cfg, Analyzer(cfg, reference=synth.scene(cfg, [])), lambda: cap, FakeCore(),
               lambda topic, payload, retain: published.append((topic, payload)), clock=lambda: clock["t"])
    return w, cap, clock, published


def run(w, cap, clock, seconds, dt=0.25):
    for _ in range(int(seconds / dt)):
        clock["t"] += dt
        cap.t = clock["t"]
        w.step()


def states(published):
    return [json.loads(p) for t, p in published if t == f"{BASE}/state"]


def test_arm_state_ttl_and_context():
    a = ArmState()
    assert not a.armed(0)
    a.arm({"ttl_s": 60, "set_id": "s1", "expected": 225, "junk": 1}, now=100)
    assert a.armed(159) and not a.armed(161)
    assert a.context == {"set_id": "s1", "expected": 225}
    a.arm({"ttl_s": 1}, now=0)
    assert a.armed(9.9)  # never shorter than 10 s
    a.disarm()
    assert not a.armed(0)


def test_discovery_has_every_entity_and_the_gym_unit():
    d = dict(discovery("homeassistant", "kg"))
    assert d["homeassistant/sensor/liftodroid_barbell/weight/config"]["unit_of_measurement"] == "kg"
    kinds = {t.split("/")[1] for t in d}
    assert kinds == {"sensor", "binary_sensor", "image", "button", "switch"}
    assert all(p["availability_topic"] == f"{BASE}/status" for p in d.values())


def test_idle_until_armed_then_reads_the_settled_bar_once():
    cfg = synth.config()
    loaded = synth.scene(cfg, [45, 45])
    w, cap, clock, pub = make(lambda t: loaded)
    run(w, cap, clock, 3)
    assert not cap.opened and states(pub) == []
    w.on_command(f"{BASE}/cmd/arm", json.dumps({"set_id": "s1", "expected": 225, "ttl_s": 600}).encode())
    run(w, cap, clock, 5)
    s = states(pub)
    assert len(s) == 1, "one reading per settle, not one per frame"
    assert s[0]["detected_weight"] == 225
    assert s[0]["context"]["set_id"] == "s1"
    assert s[0]["frame_timestamp"] is not None
    assert any(t == f"{BASE}/frame" for t, _ in pub)


def test_loading_plates_then_stepping_away_triggers_a_new_reading():
    cfg = synth.config()
    scenes = {"a": synth.scene(cfg, [45]), "b": synth.scene(cfg, [45, 25])}
    state = {"now": "a"}
    w, cap, clock, pub = make(lambda t: scenes[state["now"]])
    w.on_command(f"{BASE}/cmd/arm", b'{"set_id": "s1"}')
    run(w, cap, clock, 4)
    state["now"] = "b"  # plates change between sets
    run(w, cap, clock, 4)
    s = states(pub)
    assert [x["detected_weight"] for x in s] == [135, 185]
    assert s[-1]["last_motion_ts"] is not None
    assert (f"{BASE}/motion", "ON") in pub


def test_waits_while_someone_is_in_the_rack():
    cfg = synth.config()
    w, cap, clock, pub = make(lambda t: synth.scene(cfg, [45]), person_entity="binary_sensor.in_squat_rack_person")
    w.core.person = "on"
    w.on_command(f"{BASE}/cmd/arm", b'{"set_id": "s1"}')
    run(w, cap, clock, 5)
    assert states(pub) == []
    w.core.person = "off"
    run(w, cap, clock, 1)
    assert [x["detected_weight"] for x in states(pub)] == [135]


def test_expiry_and_disable_close_the_camera():
    cfg = synth.config()
    w, cap, clock, pub = make(lambda t: synth.scene(cfg, []))
    w.on_command(f"{BASE}/cmd/arm", b'{"ttl_s": 20}')
    run(w, cap, clock, 2)
    assert cap.opened
    run(w, cap, clock, 25)
    assert cap.closed
    assert (f"{BASE}/armed", "OFF") in pub
    w.on_command(f"{BASE}/cmd/enabled", b"OFF")
    assert (f"{BASE}/enabled", "OFF") in pub


def test_calibration_cli_analyze_and_evaluate(tmp_path, capsys):
    import cv2
    import yaml

    from barbell.calibrate import main

    cfg = synth.config()
    s = cfg.sleeve
    conf = {
        "unit": "lb", "bar_weight": 45,
        "sleeve": {"axis": [list(p) for p in s.axis], "roi": [list(p) for p in s.roi], "sleeve_length_mm": s.sleeve_length_mm},
        "reference_image": str(tmp_path / "ref.png"),
    }
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(conf))
    cv2.imwrite(str(tmp_path / "empty.png"), synth.scene(cfg, []))
    main(["reference", "--config", str(tmp_path / "c.yaml"), "--frame", str(tmp_path / "empty.png"), "--out", str(tmp_path / "ref.png")])
    labelled = tmp_path / "labelled"
    labelled.mkdir()
    for side, total in (([45], 135), ([45, 25], 185), ([], 45)):
        cv2.imwrite(str(labelled / f"{total}_a.png"), synth.scene(cfg, side))
    main(["evaluate", "--config", str(tmp_path / "c.yaml"), "--dir", str(labelled)])
    out = capsys.readouterr().out
    assert "3/3 right" in out
    main(["analyze", "--config", str(tmp_path / "c.yaml"), "--frame", str(labelled / "185_a.png"), "--out", str(tmp_path / "ann.jpg")])
    assert '"detected_weight": 185.0' in capsys.readouterr().out
    assert (tmp_path / "ann.jpg").exists()


def test_ensure_config_file_creates_default_once(tmp_path):
    from barbell.config import load
    from barbell.service import ensure_config_file

    dest = tmp_path / "nested" / "barbell.yaml"
    assert ensure_config_file(dest) is True
    assert dest.exists()
    cfg = load(dest)
    assert cfg.unit == "lb" and cfg.sleeve is not None

    dest.write_text("unit: kg\n")
    assert ensure_config_file(dest) is False
    assert dest.read_text() == "unit: kg\n"


def test_token_falls_back_to_the_s6_environment(tmp_path, monkeypatch):
    from barbell.service import supervisor_token

    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    assert supervisor_token(str(tmp_path)) is None
    (tmp_path / "SUPERVISOR_TOKEN").write_text("abc123\n")
    assert supervisor_token(str(tmp_path)) == "abc123"
    monkeypatch.setenv("SUPERVISOR_TOKEN", "from-env")
    assert supervisor_token(str(tmp_path)) == "from-env"


def test_save_empty_bar_makes_the_next_frame_the_reference(tmp_path):
    from dataclasses import replace

    cfg = replace(synth.config(), reference_image=str(tmp_path / "ref.png"))
    empty = synth.scene(cfg, [])
    cap = FakeCapture(lambda t: empty)
    clock = {"t": 0.0}
    pub = []
    w = Worker(cfg, Analyzer(cfg), lambda: cap, FakeCore(), lambda t, p, r: pub.append((t, p)), clock=lambda: clock["t"])
    w.on_command(f"{BASE}/cmd/reference", b"")
    run(w, cap, clock, 1)
    assert (tmp_path / "ref.png").exists()
    assert w.analyzer.reference is not None
    # With the reference in place, a loaded bar reads (and the empty frame itself isn't published as a reading).
    assert states(pub) == []
    assert "homeassistant/button/liftodroid_barbell/reference/config" in dict(discovery("homeassistant"))


def test_snapshot_mode_asks_home_assistant_at_most_once_per_interval():
    import cv2

    from barbell.service import SnapshotCapture

    ok, jpg = cv2.imencode(".jpg", synth.scene(synth.config(), [45]))
    calls = []

    class Core:
        def get_bytes(self, path):
            calls.append(path)
            return jpg.tobytes()

    now = {"t": 0.0}
    cap = SnapshotCapture("camera.gym", Core(), min_interval_s=0.5, clock=lambda: now["t"])
    for _ in range(20):  # one second of 50 ms loop ticks
        assert cap.latest() is not None
        now["t"] += 0.05
    assert calls == ["/camera_proxy/camera.gym"] * 2
