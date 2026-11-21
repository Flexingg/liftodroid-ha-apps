"""Typed configuration. Everything a camera, a lens or a gym differs in lives here, never in code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

Point = tuple[float, float]


@dataclass(frozen=True)
class Plate:
    """One plate denomination you own. [count] is the total (both sides). Sizes in mm; measure your own plates."""

    weight: float
    count: int
    diameter_mm: float
    thickness_mm: float


@dataclass(frozen=True)
class Lens:
    """Camera intrinsics. model: "none" (already rectilinear), "pinhole" (k1..k3, p1, p2) or "fisheye" (k1..k4)."""

    model: str = "none"
    camera_matrix: tuple[tuple[float, float, float], ...] | None = None
    dist_coeffs: tuple[float, ...] = ()
    # Fisheye only: 0 crops to valid pixels, 1 keeps the whole field of view.
    balance: float = 0.0
    # The resolution the intrinsics were calibrated at; frames of another size are scaled to match.
    calibrated_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class Sleeve:
    """The loaded sleeve, in the UNDISTORTED frame, normalised 0..1.

    axis: from the collar (where the first plate sits) to the sleeve tip.
    roi:  polygon around the sleeve and its plates (used for motion gating and OCR).
    """

    axis: tuple[Point, Point]
    roi: tuple[Point, ...]
    sleeve_length_mm: float = 415.0


@dataclass(frozen=True)
class Thresholds:
    # Motion gate: a pixel moved if its grey level changed by more than this...
    motion_pixel_delta: int = 18
    # ...and the sleeve moved if more than this fraction of its pixels did.
    motion_fraction: float = 0.015
    # Quiet this long after movement before reading (the lifter steps away, blur is gone).
    settle_s: float = 1.5
    # Readings older than this are stale; the phone ignores them.
    ttl_s: float = 180.0
    # At or above this confidence (with no warnings) the phone may apply the weight by itself.
    auto_apply_confidence: float = 0.8
    # OCR hits below this score are ignored.
    ocr_min_score: float = 0.55
    # Difference against the empty-sleeve reference that counts as "plate here".
    plate_pixel_delta: int = 35
    # How sharply plate-shape mismatches count against a loading (higher = more decisive). Tune on real frames.
    shape_weight: float = 6.0
    # Slop allowed between the measured and predicted stack length, mm (plus 5% of the stack).
    length_tolerance_mm: float = 6.0


@dataclass(frozen=True)
class Config:
    unit: str = "lb"
    bar_weight: float = 45.0
    plates: tuple[Plate, ...] = ()
    lens: Lens = Lens()
    sleeve: Sleeve | None = None
    # Optional second sleeve, for symmetric-loading checks.
    sleeve_b: Sleeve | None = None
    # Optional context zones (normalised polygons in the undistorted frame).
    storage_roi: tuple[Point, ...] | None = None
    bench_roi: tuple[Point, ...] | None = None
    thresholds: Thresholds = Thresholds()
    camera_url: str = ""
    analysis_fps: float = 2.0
    # The empty-sleeve reference image (undistorted frame), captured during calibration.
    reference_image: str | None = None
    # Optional HA entity that is "on" while someone is in the rack (e.g. a VisionState person sensor):
    # readings wait until it turns off.
    person_entity: str | None = None
    # Measure plate slabs along the sleeve. Needs a side-on view of the sleeve; turn off when the camera looks along
    # the bar (then only lettering and the inventory decide, and the phone asks more often).
    use_shape: bool = True

    def plate_by_weight(self, w: float) -> Plate | None:
        return next((p for p in self.plates if abs(p.weight - w) < 1e-6), None)


DEFAULT_LB_PLATES = (
    Plate(45, 6, 450, 45),
    Plate(35, 2, 405, 38),
    Plate(25, 2, 345, 32),
    Plate(10, 2, 255, 25),
    Plate(5, 2, 205, 20),
    Plate(2.5, 2, 165, 16),
)

DEFAULT_KG_PLATES = (
    Plate(25, 2, 450, 45),
    Plate(20, 4, 450, 40),
    Plate(15, 2, 400, 35),
    Plate(10, 2, 330, 30),
    Plate(5, 2, 230, 25),
    Plate(2.5, 2, 190, 20),
    Plate(1.25, 2, 160, 15),
)


def _points(v: Any) -> tuple[Point, ...]:
    return tuple((float(p[0]), float(p[1])) for p in v)


def _sleeve(v: dict | None) -> Sleeve | None:
    if not v:
        return None
    axis = _points(v["axis"])
    if len(axis) != 2:
        raise ValueError("sleeve.axis needs exactly two points: collar, tip")
    return Sleeve(axis=(axis[0], axis[1]), roi=_points(v["roi"]), sleeve_length_mm=float(v.get("sleeve_length_mm", 415.0)))


def from_dict(d: dict[str, Any]) -> Config:
    unit = str(d.get("unit", "lb")).lower()
    if unit not in ("lb", "kg"):
        raise ValueError(f"unit must be lb or kg, not {unit}")
    plates = tuple(
        Plate(float(p["weight"]), int(p["count"]), float(p.get("diameter_mm", 0)), float(p.get("thickness_mm", 0)))
        for p in d.get("plates", [])
    ) or (DEFAULT_LB_PLATES if unit == "lb" else DEFAULT_KG_PLATES)
    for p in plates:
        if p.count < 0 or p.weight <= 0:
            raise ValueError(f"bad plate {p}")
    lens_d = d.get("lens") or {}
    lens = Lens(
        model=str(lens_d.get("model", "none")),
        camera_matrix=tuple(tuple(float(x) for x in row) for row in lens_d["camera_matrix"]) if lens_d.get("camera_matrix") else None,
        dist_coeffs=tuple(float(x) for x in lens_d.get("dist_coeffs", ())),
        balance=float(lens_d.get("balance", 0.0)),
        calibrated_size=tuple(lens_d["calibrated_size"]) if lens_d.get("calibrated_size") else None,
    )
    if lens.model not in ("none", "pinhole", "fisheye"):
        raise ValueError("lens.model must be none, pinhole or fisheye")
    if lens.model != "none" and lens.camera_matrix is None:
        raise ValueError("lens.camera_matrix is required unless lens.model is none")
    th = Thresholds(**{k: v for k, v in (d.get("thresholds") or {}).items() if k in Thresholds.__dataclass_fields__})
    return Config(
        unit=unit,
        bar_weight=float(d.get("bar_weight", 45.0 if unit == "lb" else 20.0)),
        plates=plates,
        lens=lens,
        sleeve=_sleeve(d.get("sleeve")),
        sleeve_b=_sleeve(d.get("sleeve_b")),
        storage_roi=_points(d["storage_roi"]) if d.get("storage_roi") else None,
        bench_roi=_points(d["bench_roi"]) if d.get("bench_roi") else None,
        thresholds=th,
        camera_url=str(d.get("camera_url", "")),
        analysis_fps=float(d.get("analysis_fps", 2.0)),
        reference_image=d.get("reference_image"),
        person_entity=d.get("person_entity"),
        use_shape=bool(d.get("use_shape", True)),
    )


def load(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return from_dict(yaml.safe_load(f) or {})


def with_context(cfg: Config, unit: str | None, bar: float | None, plates: list[dict] | None) -> Config:
    """The phone knows the gym's bar and plates from Liftosaur; its counts win, sizes stay from the file."""
    from dataclasses import replace

    out = cfg
    if unit in ("lb", "kg"):
        out = replace(out, unit=unit)
    if bar:
        out = replace(out, bar_weight=float(bar))
    if plates:
        merged = []
        for p in plates:
            w = float(p["weight"])
            known = cfg.plate_by_weight(w)
            merged.append(Plate(w, int(p["count"]), known.diameter_mm if known else 0.0, known.thickness_mm if known else 0.0))
        out = replace(out, plates=tuple(merged))
    return out


def _parse_points(text: str) -> tuple[Point, ...]:
    """"0.19,0.10 0.41,0.10 0.41,0.46" -> ((0.19, 0.10), ...). Spaces or semicolons between points."""
    pts = []
    for tok in text.replace(";", " ").split():
        x, y = tok.split(",")
        px, py = float(x), float(y)
        if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0):
            raise ValueError(f"point {tok} is outside the frame: use fractions 0..1")
        pts.append((px, py))
    return tuple(pts)


def _parse_plates(text: str, unit: str) -> tuple[Plate, ...]:
    """"45:450/76, 25:450/43, 10x4:450/25" -> weight[xcount]:diameter_mm/thickness_mm. Count defaults to 2 (one
    pair); the phone sends the real counts from Liftosaur for every set anyway."""
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        head, _, size = item.partition(":")
        weight, _, count = head.lower().partition("x")
        d, _, t = size.partition("/")
        out.append(Plate(float(weight), int(count) if count else 2, float(d) if d else 0.0, float(t) if t else 0.0))
    if not out:
        raise ValueError("plates is empty")
    return tuple(out)


def apply_options(cfg: Config, opts: dict[str, Any]) -> Config:
    """The app's Configuration tab overrides barbell.yaml, so everything a setup needs can be set from Home Assistant
    (and remotely) without editing files. Empty options leave the file's value alone."""
    from dataclasses import replace

    out = cfg
    if opts.get("unit") in ("lb", "kg"):
        out = replace(out, unit=opts["unit"])
    if opts.get("bar_weight"):
        out = replace(out, bar_weight=float(opts["bar_weight"]))
    if str(opts.get("plates", "")).strip():
        out = replace(out, plates=_parse_plates(str(opts["plates"]), out.unit))
    if opts.get("camera_url"):
        out = replace(out, camera_url=str(opts["camera_url"]))
    if opts.get("person_entity"):
        out = replace(out, person_entity=str(opts["person_entity"]))
    if "use_shape" in opts and opts["use_shape"] is not None:
        out = replace(out, use_shape=bool(opts["use_shape"]))
    roi_text = str(opts.get("sleeve_roi", "") or "").strip()
    axis_text = str(opts.get("sleeve_axis", "") or "").strip()
    if roi_text or axis_text:
        base = out.sleeve
        roi = _parse_points(roi_text) if roi_text else (base.roi if base else None)
        if roi is None or len(roi) < 3:
            raise ValueError("sleeve_roi needs at least three points")
        if axis_text:
            axis = _parse_points(axis_text)
            if len(axis) != 2:
                raise ValueError("sleeve_axis needs exactly two points: collar, tip")
        elif base is not None:
            axis = base.axis
        else:
            # No axis: across the middle of the region (only matters when use_shape is on).
            xs, ys = [p[0] for p in roi], [p[1] for p in roi]
            axis = ((min(xs), (min(ys) + max(ys)) / 2), (max(xs), (min(ys) + max(ys)) / 2))
        length = float(opts.get("sleeve_length_mm") or (base.sleeve_length_mm if base else 415.0))
        out = replace(out, sleeve=Sleeve(axis=(axis[0], axis[1]), roi=roi, sleeve_length_mm=length))
    elif opts.get("sleeve_length_mm") and out.sleeve is not None:
        out = replace(out, sleeve=replace(out.sleeve, sleeve_length_mm=float(opts["sleeve_length_mm"])))
    return out
