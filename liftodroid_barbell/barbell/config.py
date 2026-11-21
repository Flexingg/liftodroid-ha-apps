"""Typed configuration. Everything a camera, a lens or a gym differs in lives here, never in code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

Point = tuple[float, float]


@dataclass(frozen=True)
class Plate:
    """One plate denomination you own. [count] is the total (both sides). Sizes in mm; measure your own plates.

    Optional [color] is a named electrical tape color (e.g. "red", "blue", "yellow", "green", "orange", "purple",
    "pink", "white") wrapped around the outer circumference; [color_hsv] stores a sampled (H_deg 0..360, S_pct 0..100,
    V_pct 0..100) swatch from the camera for exact lighting calibration.
    """

    weight: float
    count: int
    diameter_mm: float
    thickness_mm: float
    color: str | None = None
    color_hsv: tuple[float, float, float] | None = None

    @property
    def color_key(self) -> str | None:
        """Canonical label identifying this plate's tape color, or None if uncolored."""
        if self.color:
            return self.color.strip().lower()
        if self.color_hsv is not None:
            return f"custom_{self.weight:g}"
        return None


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
    outer_face_mm: for angled / 3/4 cameras where the outermost plate's face projects past its rim along the axis.
    """

    axis: tuple[Point, Point]
    roi: tuple[Point, ...]
    sleeve_length_mm: float = 415.0
    outer_face_mm: float = 0.0


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
    # Colored rim electrical tape detection tuning.
    color_weight: float = 4.0
    color_hue_tolerance: float = 22.0
    color_min_saturation: int = 55
    color_min_value: int = 40
    color_min_fraction: float = 0.08


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
    # Measure plate slabs / thickness along the sleeve.
    use_shape: bool = True
    # Read colored electrical tape bands around the outer circumference of plates (when any plate has a color set).
    use_color: bool = True
    # Read white lettering stamped on plate faces via OCR.
    use_ocr: bool = True

    def plate_by_weight(self, w: float) -> Plate | None:
        return next((p for p in self.plates if abs(p.weight - w) < 1e-6), None)

    @property
    def has_tape_colors(self) -> bool:
        return any(p.color_key is not None for p in self.plates)


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

VISION_OPTION_KEYS = (
    "unit",
    "bar_weight",
    "plates",
    "sleeve_roi",
    "sleeve_axis",
    "sleeve_length_mm",
    "outer_face_mm",
    "use_shape",
    "use_color",
)


def _points(v: Any) -> tuple[Point, ...]:
    return tuple((float(p[0]), float(p[1])) for p in v)


def _sleeve(v: dict | None) -> Sleeve | None:
    if not v:
        return None
    axis = _points(v["axis"])
    if len(axis) != 2:
        raise ValueError("sleeve.axis needs exactly two points: collar, tip")
    return Sleeve(
        axis=(axis[0], axis[1]),
        roi=_points(v["roi"]),
        sleeve_length_mm=float(v.get("sleeve_length_mm", 415.0)),
        outer_face_mm=float(v.get("outer_face_mm", 0.0)),
    )


def _parse_hsv(v: Any) -> tuple[float, float, float] | None:
    if not v:
        return None
    if len(v) != 3:
        raise ValueError(f"color_hsv must have 3 values [H_deg, S_pct, V_pct], got {v}")
    return (float(v[0]) % 360.0, max(0.0, min(100.0, float(v[1]))), max(0.0, min(100.0, float(v[2]))))


def from_dict(d: dict[str, Any]) -> Config:
    unit = str(d.get("unit", "lb")).lower()
    if unit not in ("lb", "kg"):
        raise ValueError(f"unit must be lb or kg, not {unit}")
    plates = tuple(
        Plate(
            float(p["weight"]),
            int(p["count"]),
            float(p.get("diameter_mm", 0)),
            float(p.get("thickness_mm", 0)),
            str(p["color"]).strip().lower() if p.get("color") else None,
            _parse_hsv(p.get("color_hsv")),
        )
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
        calibrated_size=tuple(int(x) for x in lens_d["calibrated_size"]) if lens_d.get("calibrated_size") else None,
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
        use_color=bool(d.get("use_color", True)),
        use_ocr=bool(d.get("use_ocr", True)),
    )


def _sleeve_to_dict(s: Sleeve) -> dict[str, Any]:
    out: dict[str, Any] = {
        "axis": [[round(p[0], 5), round(p[1], 5)] for p in s.axis],
        "roi": [[round(p[0], 5), round(p[1], 5)] for p in s.roi],
        "sleeve_length_mm": round(float(s.sleeve_length_mm), 2),
    }
    if s.outer_face_mm:
        out["outer_face_mm"] = round(float(s.outer_face_mm), 2)
    return out


def to_dict(cfg: Config) -> dict[str, Any]:
    """Serialize [cfg] to a clean plain dict suitable for YAML or JSON (excluding camera_url secrets by default)."""
    plates_out = []
    for p in cfg.plates:
        item: dict[str, Any] = {
            "weight": p.weight,
            "count": p.count,
            "diameter_mm": p.diameter_mm,
            "thickness_mm": p.thickness_mm,
        }
        if p.color:
            item["color"] = p.color
        if p.color_hsv is not None:
            item["color_hsv"] = [round(v, 1) for v in p.color_hsv]
        plates_out.append(item)

    lens_out: dict[str, Any] = {"model": cfg.lens.model}
    if cfg.lens.calibrated_size is not None:
        lens_out["calibrated_size"] = list(cfg.lens.calibrated_size)
    if cfg.lens.camera_matrix is not None:
        lens_out["camera_matrix"] = [list(row) for row in cfg.lens.camera_matrix]
    if cfg.lens.dist_coeffs:
        lens_out["dist_coeffs"] = list(cfg.lens.dist_coeffs)
    if cfg.lens.model == "fisheye" or cfg.lens.balance:
        lens_out["balance"] = cfg.lens.balance

    th = cfg.thresholds
    out: dict[str, Any] = {
        "unit": cfg.unit,
        "bar_weight": cfg.bar_weight,
        "plates": plates_out,
        "lens": lens_out,
        "use_shape": cfg.use_shape,
        "use_color": cfg.use_color,
        "use_ocr": cfg.use_ocr,
        "analysis_fps": cfg.analysis_fps,
        "thresholds": {
            "motion_pixel_delta": th.motion_pixel_delta,
            "motion_fraction": th.motion_fraction,
            "settle_s": th.settle_s,
            "ttl_s": th.ttl_s,
            "auto_apply_confidence": th.auto_apply_confidence,
            "ocr_min_score": th.ocr_min_score,
            "plate_pixel_delta": th.plate_pixel_delta,
            "shape_weight": th.shape_weight,
            "length_tolerance_mm": th.length_tolerance_mm,
            "color_weight": th.color_weight,
            "color_hue_tolerance": th.color_hue_tolerance,
            "color_min_saturation": th.color_min_saturation,
            "color_min_value": th.color_min_value,
            "color_min_fraction": th.color_min_fraction,
        },
    }
    if cfg.sleeve is not None:
        out["sleeve"] = _sleeve_to_dict(cfg.sleeve)
    if cfg.sleeve_b is not None:
        out["sleeve_b"] = _sleeve_to_dict(cfg.sleeve_b)
    if cfg.storage_roi is not None:
        out["storage_roi"] = [[round(p[0], 5), round(p[1], 5)] for p in cfg.storage_roi]
    if cfg.bench_roi is not None:
        out["bench_roi"] = [[round(p[0], 5), round(p[1], 5)] for p in cfg.bench_roi]
    if cfg.reference_image:
        out["reference_image"] = cfg.reference_image
    if cfg.person_entity:
        out["person_entity"] = cfg.person_entity
    return out


def save(cfg: Config, path: str | Path, options_snapshot: dict[str, Any] | None = None) -> None:
    """Write [cfg] cleanly to YAML. If [options_snapshot] is given, records the current HA vision options so
    restarts preserve Web UI edits unless the user explicitly changes an option in Home Assistant later."""
    data = to_dict(cfg)
    if options_snapshot is not None:
        data["_ui_options_snapshot"] = {k: options_snapshot.get(k) for k in VISION_OPTION_KEYS if k in options_snapshot}
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    tmp.replace(dest)


def load(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return from_dict(yaml.safe_load(f) or {})


def load_with_options(path: str | Path, opts: dict[str, Any]) -> Config:
    """Load [path] and apply Home Assistant [opts], skipping vision options that haven't changed since the user
    last saved from the Web UI."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = from_dict(raw)
    snapshot = raw.get("_ui_options_snapshot")
    if isinstance(snapshot, dict):
        filtered = dict(opts)
        for k in VISION_OPTION_KEYS:
            if k in snapshot and filtered.get(k) == snapshot.get(k):
                filtered.pop(k, None)
        return apply_options(cfg, filtered)
    return apply_options(cfg, opts)


def with_context(cfg: Config, unit: str | None, bar: float | None, plates: list[dict] | None) -> Config:
    """The phone knows the gym's bar and plates from Liftosaur; its counts win, sizes & tape colors stay from the file."""
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
            merged.append(Plate(
                w,
                int(p["count"]),
                known.diameter_mm if known else 0.0,
                known.thickness_mm if known else 0.0,
                known.color if known else None,
                known.color_hsv if known else None,
            ))
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
    """"45:450/76, 25:450/43:blue, 10x4:450/25:green" -> weight[xcount]:diameter_mm/thickness_mm[:color].
    Count defaults to 2 (one pair); the phone sends the real counts from Liftosaur for every set anyway."""
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        head = parts[0]
        size = parts[1] if len(parts) > 1 else ""
        color = parts[2].strip().lower() or None if len(parts) > 2 else None
        weight, _, count = head.lower().partition("x")
        d, _, t = size.partition("/")
        out.append(Plate(float(weight), int(count) if count else 2, float(d) if d else 0.0, float(t) if t else 0.0, color))
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
    if "use_color" in opts and opts["use_color"] is not None:
        out = replace(out, use_color=bool(opts["use_color"]))
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
        face = float(opts["outer_face_mm"]) if opts.get("outer_face_mm") is not None else (base.outer_face_mm if base else 0.0)
        out = replace(out, sleeve=Sleeve(axis=(axis[0], axis[1]), roi=roi, sleeve_length_mm=length, outer_face_mm=face))
    elif out.sleeve is not None:
        length = float(opts["sleeve_length_mm"]) if opts.get("sleeve_length_mm") else out.sleeve.sleeve_length_mm
        face = float(opts["outer_face_mm"]) if opts.get("outer_face_mm") is not None else out.sleeve.outer_face_mm
        if length != out.sleeve.sleeve_length_mm or face != out.sleeve.outer_face_mm:
            out = replace(out, sleeve=replace(out.sleeve, sleeve_length_mm=length, outer_face_mm=face))
    return out
