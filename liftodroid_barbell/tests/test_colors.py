"""Tests for colored electrical tape detection along the sleeve rim and angled-camera thickness detection."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from barbell.colors import NAMED_COLORS, color_bgr, sample_hsv_patch
from barbell.config import Plate, with_context
from barbell.pipeline import Analyzer, diagnostic_strip
from tests import synth


def bumper_plates_with_tape():
    # Equal-diameter 450 mm bumper plates with distinct electrical tape colors on the outer rim.
    return (
        Plate(45, 6, 450, 76, color="red"),
        Plate(35, 2, 450, 58, color="yellow"),
        Plate(25, 2, 450, 43, color="blue"),
        Plate(10, 4, 450, 25, color="green"),
    )


def tape_scene(cfg, side, tape_width_mm=19.0, full_rim=False, bg_red_rack=False, outer_face_mm=0.0):
    """Draw a synthetic bumper-plate sleeve with either 19mm electrical tape stripes or full-width colored rims."""
    img = synth.scene(cfg, [], floor=130)
    if bg_red_rack:
        # Bright red squat-rack upright behind the sleeve (must be ignored when empty-bar reference is present).
        cv2.rectangle(img, (synth.COLLAR[0] + 15, synth.COLLAR[1] - 320), (synth.COLLAR[0] + 55, synth.COLLAR[1] + 320), (20, 20, 210), -1)
    if not side:
        return img
    x = float(synth.COLLAR[0])
    y0 = synth.COLLAR[1]
    for i, w in enumerate(side):
        p = cfg.plate_by_weight(w)
        t_px = p.thickness_mm * synth.PX_PER_MM
        face_px = outer_face_mm * synth.PX_PER_MM if i == len(side) - 1 else 0.0
        r = p.diameter_mm * synth.PX_PER_MM / 2
        cv2.rectangle(img, (int(round(x)), int(y0 - r)), (int(round(x + t_px + face_px)) - 1, int(y0 + r)), (28, 28, 28), -1)
        if p.color_key:
            bgr = color_bgr(p.color, p.color_hsv)
            if full_rim:
                ta, tb = x + 1, x + t_px - 1
            else:
                tw = min(tape_width_mm, p.thickness_mm - 4) * synth.PX_PER_MM
                mid = x + t_px / 2
                ta, tb = mid - tw / 2, mid + tw / 2
            cv2.rectangle(img, (int(round(ta)), int(y0 - r + 4)), (int(round(tb)), int(y0 + r - 4)), bgr, -1)
        x += t_px
    return img


@pytest.mark.parametrize("side,total", [
    ([45], 135),
    ([45, 45], 225),
    ([45, 25], 185),
    ([45, 35, 10], 225),
    ([45, 45, 25, 10], 295),
    ([25, 10], 115),
])
def test_narrow_electrical_tape_reads_hidden_plates_without_shape(side, total):
    cfg = replace(synth.config(), plates=bumper_plates_with_tape(), use_shape=False, use_color=True)
    ref = tape_scene(cfg, [])
    frame = tape_scene(cfg, side, tape_width_mm=19.0)
    a = Analyzer(cfg, reference=ref).analyze(frame)
    assert a.reading.detected_weight == total, (a.reading, a.colors_a)
    assert a.reading.per_side_breakdown == side
    assert a.reading.confidence >= 0.8
    assert not a.reading.needs_confirmation
    assert diagnostic_strip(a, cfg) is not None


def test_full_width_colored_rims_split_touching_same_color_plates():
    cfg = replace(synth.config(), plates=bumper_plates_with_tape(), use_shape=False, use_color=True)
    ref = tape_scene(cfg, [])
    frame = tape_scene(cfg, [45, 45, 25], full_rim=True)
    a = Analyzer(cfg, reference=ref).analyze(frame)
    assert a.reading.detected_weight == 275
    assert a.reading.per_side_breakdown == [45, 45, 25]
    assert a.colors_a.count_by_color["red"] == 2
    assert a.colors_a.count_by_color["blue"] == 1


def test_empty_reference_suppresses_red_rack_background():
    cfg = replace(synth.config(), plates=bumper_plates_with_tape(), use_shape=False, use_color=True)
    ref = tape_scene(cfg, [], bg_red_rack=True)
    # Only a blue 25 lb plate is on the bar; the red rack behind it must NOT count as a red 45.
    frame = tape_scene(cfg, [25], bg_red_rack=True)
    a = Analyzer(cfg, reference=ref).analyze(frame)
    assert a.reading.detected_weight == 95
    assert a.reading.per_side_breakdown == [25]
    assert a.colors_a.count_by_color["red"] == 0
    assert a.colors_a.count_by_color["blue"] == 1


def test_eyedropper_sampling_and_custom_hsv_matching():
    cfg = replace(synth.config(), plates=bumper_plates_with_tape(), use_shape=False, use_color=True)
    ref = tape_scene(cfg, [])
    frame = tape_scene(cfg, [45, 25], tape_width_mm=19.0)
    # Click on the middle of the red tape stripe on the first 45 lb plate.
    red_x = (synth.COLLAR[0] + 38 * synth.PX_PER_MM) / synth.W
    red_y = synth.COLLAR[1] / synth.H
    sampled = sample_hsv_patch(frame, red_x, red_y)
    assert sampled["nearest_name"] == "red"
    assert len(sampled["hsv"]) == 3 and sampled["hex"].startswith("#")

    # Save sampled HSV onto the 45 lb plate and verify it still detects cleanly.
    plates = tuple(
        replace(p, color_hsv=tuple(sampled["hsv"])) if p.weight == 45 else p
        for p in cfg.plates
    )
    cfg2 = replace(cfg, plates=plates)
    r = Analyzer(cfg2, reference=ref).analyze(frame).reading
    assert r.detected_weight == 185 and r.per_side_breakdown == [45, 25]


def test_with_context_preserves_plate_tape_colors():
    cfg = replace(synth.config(), plates=bumper_plates_with_tape())
    merged = with_context(cfg, "lb", 45, [{"weight": 45, "count": 4}, {"weight": 25, "count": 2}])
    assert merged.plate_by_weight(45).color == "red"
    assert merged.plate_by_weight(25).color == "blue"


@pytest.mark.parametrize("side,total", [
    ([], 45),
    ([45], 135),
    ([45, 25], 185),
    ([45, 45], 225),
    ([45, 10], 155),
])
def test_angled_camera_thickness_with_outer_face_overhang(side, total):
    """Equal-diameter 450mm bumper plates with a 50mm outer-face overhang (angled 3/4 view): thickness alone reads
    the hidden plates behind the outer one."""
    plates = (
        Plate(45, 4, 450, 76),
        Plate(25, 2, 450, 43),
        Plate(10, 2, 450, 25),
    )
    sl = replace(synth.sleeve(), outer_face_mm=50.0)
    cfg = replace(synth.config(), plates=plates, sleeve=sl, use_shape=True, use_color=False)
    ref = tape_scene(cfg, [], outer_face_mm=50.0)
    frame = tape_scene(cfg, side, outer_face_mm=50.0)
    a = Analyzer(cfg, reference=ref).analyze(frame)
    assert a.reading.detected_weight == total, (a.reading, a.raw_profile_a.stack_length_mm)
    assert a.reading.per_side_breakdown == side
    assert a.reading.confidence >= 0.8
