from dataclasses import replace

import cv2
import numpy as np
import pytest

from barbell.config import Plate
from barbell.fusion import SideEvidence, fuse
from barbell.motion import GateEvent, MotionGate
from barbell.ocr import TextHit
from barbell.pipeline import Analyzer
from tests import synth


@pytest.fixture
def cfg():
    return synth.config()


def empty(cfg):
    return synth.scene(cfg, [])


@pytest.mark.parametrize("side,total", [
    ([], 45), ([45], 135), ([45, 45], 225), ([45, 25], 185), ([45, 10], 155), ([25], 95),
    ([45, 45, 45], 315), ([45, 35, 10, 5], 235), ([10, 5, 2.5], 80),
])
def test_reads_loading_from_plate_slabs(cfg, side, total):
    a = Analyzer(cfg, reference=empty(cfg)).analyze(synth.scene(cfg, side))
    assert a.reading.detected_weight == total, (a.reading, [r for r in a.profile_a.runs])
    assert a.reading.per_side_breakdown == side
    assert a.reading.confidence >= 0.8


def test_works_without_a_reference_on_a_plain_floor(cfg):
    r = Analyzer(cfg).analyze(synth.scene(cfg, [45, 25])).reading
    assert r.detected_weight == 185


def test_sleeve_pointing_left(cfg):
    from barbell.config import Sleeve

    c = replace(cfg, sleeve=synth.sleeve(collar=(980, 360), direction=-1))
    frame = synth.scene(c, [45, 10], collar=(980, 360), direction=-1)
    base = synth.scene(c, [], collar=(980, 360), direction=-1)
    assert Analyzer(c, reference=base).analyze(frame).reading.detected_weight == 155


def bumpers():
    # Bumpers: every plate is 450 mm across, so only thickness and lettering tell them apart.
    return (Plate(45, 4, 450, 60), Plate(35, 2, 450, 60), Plate(25, 2, 450, 60), Plate(10, 2, 450, 25))


def test_equal_size_plates_are_ambiguous_without_lettering(cfg):
    c = replace(cfg, plates=bumpers())
    r = Analyzer(c, reference=empty(c)).analyze(synth.scene(c, [35])).reading
    assert r.needs_confirmation
    assert {round(x["total"]) for x in r.alternatives[:3]} >= {115, 95}


def test_lettering_settles_what_shape_cannot(cfg):
    c = replace(cfg, plates=bumpers())
    frame = synth.scene(c, [35])
    collar_px = (synth.COLLAR[0] + 30, synth.COLLAR[1])
    reader = synth.FakeReader([TextHit("35LB", 0.93, (collar_px[0] - c.sleeve.roi[0][0] * synth.W, collar_px[1] - c.sleeve.roi[0][1] * synth.H))])
    r = Analyzer(c, reader=reader, reference=empty(c)).analyze(frame).reading
    assert r.detected_weight == 115
    assert not r.needs_confirmation
    assert r.raw_ocr_matches[0]["weight"] == 35


def test_glare_over_part_of_the_stack_is_tolerated(cfg):
    frame = synth.scene(cfg, [45, 45, 10])
    cv2.circle(frame, (int(synth.COLLAR[0] + 40), synth.COLLAR[1] - 80), 30, (255, 255, 255), -1)
    r = Analyzer(cfg, reference=empty(cfg)).analyze(frame).reading
    assert r.detected_weight == 245


def test_the_plan_never_overrides_the_camera(cfg):
    r = Analyzer(cfg, reference=empty(cfg)).analyze(synth.scene(cfg, [45]), expected_total=225).reading
    assert r.detected_weight == 135
    assert any("planned 225" in x for x in r.reasons)


def test_asymmetric_loading_is_flagged(cfg):
    from barbell.config import Sleeve

    b = synth.sleeve(collar=(980, 600), direction=-1)
    c = replace(cfg, sleeve_b=b)
    base = synth.scene(c, [], collar=(980, 600), direction=-1, base=empty(c))
    frame = synth.scene(c, [45, 10], collar=(980, 600), direction=-1, base=synth.scene(c, [45, 25]))
    r = Analyzer(c, reference=base).analyze(frame).reading
    assert r.symmetric is False
    assert r.needs_confirmation and r.status == "asymmetric"
    assert r.detected_weight == 45 + 70 + 55


def test_nothing_to_go_on_is_reported_not_guessed(cfg):
    c = replace(cfg, plates=(Plate(45, 4, 0, 0), Plate(25, 2, 0, 0)))  # sizes unknown, no reader
    r = fuse(c, SideEvidence(None))
    assert r.detected_weight is None and r.status == "no_evidence" and r.needs_confirmation


def test_motion_gate_waits_for_quiet_then_fires_once():
    g = MotionGate(settle_s=1.5)
    still = np.full((100, 100), 100, np.uint8)
    moving = still.copy()
    events = []
    t = 0.0
    for i in range(6):  # someone loading plates
        f = still.copy() if i % 2 else moving
        f = f.copy()
        cv2.rectangle(f, (10 + i * 10, 10), (40 + i * 10, 90), 20, -1)
        events.append(g.feed(t, f)); t += 0.5
    for _ in range(6):  # they step away
        events.append(g.feed(t, still)); t += 0.5
    assert GateEvent.MOVING in events
    assert events.count(GateEvent.SETTLED) == 1
    first_settled = events.index(GateEvent.SETTLED)
    quiet_start = 6
    assert (first_settled - quiet_start) * 0.5 >= 1.5 - 0.5  # waited out the settle time
    assert g.last_motion_ts is not None
    g.force()
    assert g.feed(t, still) == GateEvent.SETTLED  # "detect now"
