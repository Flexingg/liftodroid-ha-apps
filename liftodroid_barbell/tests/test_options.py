"""Everything a setup needs can come from the app's Configuration tab."""

import pytest

from barbell.config import apply_options, from_dict
from barbell.fusion import SideEvidence, fuse
from barbell.ocr import Denomination

BASE = from_dict({})


def test_empty_options_change_nothing():
    assert apply_options(BASE, {"log_level": "info", "sleeve_roi": "", "plates": ""}) == BASE


def test_full_remote_setup():
    cfg = apply_options(BASE, {
        "unit": "lb", "bar_weight": 45,
        "plates": "45x6:450/76, 25:450/43, 10:450/25",
        "sleeve_roi": "0.19,0.10 0.41,0.10 0.41,0.46 0.19,0.46",
        "use_shape": False,
        "person_entity": "binary_sensor.in_squat_rack_person",
    })
    assert [(p.weight, p.count, p.diameter_mm, p.thickness_mm) for p in cfg.plates] == [(45, 6, 450, 76), (25, 2, 450, 43), (10, 2, 450, 25)]
    assert cfg.sleeve.roi[0] == (0.19, 0.10) and len(cfg.sleeve.roi) == 4
    assert cfg.sleeve.axis[0][0] == 0.19 and cfg.sleeve.axis[1][0] == 0.41  # derived across the region
    assert cfg.use_shape is False
    assert cfg.person_entity == "binary_sensor.in_squat_rack_person"


def test_axis_and_length():
    cfg = apply_options(BASE, {
        "sleeve_roi": "0.1,0.1;0.5,0.1;0.5,0.5",
        "sleeve_axis": "0.2,0.3 0.45,0.3",
        "sleeve_length_mm": 420,
        "outer_face_mm": 45,
        "plates": "45x4:450/76:red, 25:450/43:blue",
        "use_color": True,
    })
    assert cfg.sleeve.axis == ((0.2, 0.3), (0.45, 0.3))
    assert cfg.sleeve.sleeve_length_mm == 420
    assert cfg.sleeve.outer_face_mm == 45
    assert [(p.weight, p.color) for p in cfg.plates] == [(45, "red"), (25, "blue")]
    assert cfg.use_color is True


@pytest.mark.parametrize("bad", [{"sleeve_roi": "0.1,0.1 1.5,0.2 0.3,0.3"}, {"sleeve_roi": "0.1,0.1 0.2,0.2"},
                                 {"sleeve_roi": "0.1,0.1 0.2,0.1 0.2,0.2", "sleeve_axis": "0.1,0.1"}, {"plates": " , "}])
def test_bad_options_are_refused_clearly(bad):
    with pytest.raises(ValueError):
        apply_options(BASE, bad)


def _seen(w, score=0.9):
    return SideEvidence(None, ((Denomination(w, score, f"{w:g}LB", (0, 0)), 0.0),))


FACE_ON = apply_options(BASE, {"plates": "45x4:450/76, 35:450/58, 25:450/43, 10:450/25", "use_shape": False})


def test_face_on_outer_plate_matching_the_plan_reports_the_plan():
    r = fuse(FACE_ON, _seen(10), expected_total=155)  # 45 + 10 a side
    assert r.detected_weight == 155 and not r.needs_confirmation and r.status == "ok"
    assert r.per_side_breakdown == [45, 10]
    assert "can't be seen" in r.reasons[0]


def test_face_on_outer_plate_against_the_plan_reports_no_weight():
    r = fuse(FACE_ON, _seen(10), expected_total=225)  # plan: 45 + 45, outer 45
    assert r.detected_weight is None and r.needs_confirmation and r.status == "outer_plate_mismatch"
    assert r.reasons == ["Outer plate reads 10 lb, but the plan's outer plate is 45 lb."]


def test_face_on_without_a_plan_never_invents_a_total():
    r = fuse(FACE_ON, _seen(10))
    assert r.detected_weight is None and r.status == "outer_plate_only"
