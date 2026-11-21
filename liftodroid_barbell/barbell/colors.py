"""Colored electrical tape detection along the barbell sleeve.

Wrap a strip of colored electrical tape around the outer circumference (rim) of each plate denomination (for example:
red = 45, blue = 25, green = 10). Even when plates sit behind the outer plate and have identical bumper diameters,
their colored rims remain visible along the sleeve from overhead and 3/4 angles.

Works both with single narrow tape strips (e.g. 19 mm / 3/4" electrical tape centered on a 76 mm plate -> one distinct
band per plate) and with full-width colored bumper rims (touching plates of the same color -> wide band split by
plate thickness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .config import Plate, Thresholds
from .geometry import Strip
from .slabs import glare_mask

# Named vinyl electrical tape presets: (H_deg 0..360, S_pct 0..100, V_pct 0..100, UI hex, is_white)
NAMED_COLORS: dict[str, tuple[float, float, float, str, bool]] = {
    "red": (0.0, 85.0, 75.0, "#d92626", False),
    "orange": (24.0, 90.0, 90.0, "#e67300", False),
    "yellow": (55.0, 88.0, 92.0, "#ebd417", False),
    "green": (130.0, 78.0, 68.0, "#1fa83b", False),
    "cyan": (185.0, 80.0, 80.0, "#19b8cc", False),
    "blue": (220.0, 82.0, 78.0, "#2161c7", False),
    "purple": (280.0, 72.0, 72.0, "#8a33b8", False),
    "pink": (325.0, 75.0, 85.0, "#d93693", False),
    "white": (0.0, 0.0, 94.0, "#f0f0f0", True),
}


@dataclass(frozen=True)
class ColorSpec:
    key: str
    h_deg: float
    s_pct: float
    v_pct: float
    is_white: bool
    weights: tuple[float, ...]
    min_thickness_mm: float
    calibrated: bool = False


@dataclass(frozen=True)
class ColorBand:
    """One contiguous colored tape band detected along the sleeve (mm from the collar)."""

    color: str
    start_mm: float
    end_mm: float
    coverage: float
    estimated_plates: int

    @property
    def width_mm(self) -> float:
        return self.end_mm - self.start_mm

    @property
    def center_mm(self) -> float:
        return 0.5 * (self.start_mm + self.end_mm)

    def to_dict(self) -> dict:
        return {
            "color": self.color,
            "start_mm": round(self.start_mm, 1),
            "end_mm": round(self.end_mm, 1),
            "width_mm": round(self.width_mm, 1),
            "coverage": round(self.coverage, 3),
            "estimated_plates": self.estimated_plates,
        }


@dataclass(frozen=True)
class ColorProfile:
    """Detected colored tape bands and per-column color assignments along the sleeve strip."""

    bands: tuple[ColorBand, ...]
    column_colors: tuple[str | None, ...]
    width_by_color_mm: dict[str, float]
    count_by_color: dict[str, int]
    mm_per_px: float
    configured_colors: tuple[str, ...]
    # Per-color 2D match mask (0..255 uint8, same HxW as strip), useful for UI diagnostics.
    masks: dict[str, np.ndarray]

    @property
    def has_evidence(self) -> bool:
        return bool(self.bands)

    def color_at_mm(self, x_mm: float) -> str | None:
        idx = int(round(x_mm / self.mm_per_px))
        if 0 <= idx < len(self.column_colors):
            return self.column_colors[idx]
        return None


def color_hex(name: str | None, hsv: tuple[float, float, float] | None = None) -> str:
    if hsv is not None:
        h = int(round((hsv[0] % 360.0) / 2.0)) % 180
        s = int(round(max(0.0, min(100.0, hsv[1])) * 2.55))
        v = int(round(max(0.0, min(100.0, hsv[2])) * 2.55))
        bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        return f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"
    if name and name.lower() in NAMED_COLORS:
        return NAMED_COLORS[name.lower()][3]
    return "#888888"


def color_bgr(name: str | None, hsv: tuple[float, float, float] | None = None) -> tuple[int, int, int]:
    hx = color_hex(name, hsv).lstrip("#")
    return (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))


def nearest_color_name(h_deg: float, s_pct: float, v_pct: float) -> str:
    if s_pct < 18.0 and v_pct >= 62.0:
        return "white"
    best_name, best_dist = "red", 999.0
    for name, (nh, _, _, _, is_white) in NAMED_COLORS.items():
        if is_white:
            continue
        d = abs((h_deg % 360.0) - nh)
        d = min(d, 360.0 - d)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def sample_hsv_patch(frame: np.ndarray, x_norm: float, y_norm: float, radius_px: int = 6) -> dict:
    """Sample median HSV around normalized (x_norm, y_norm) in a BGR frame for the UI eyedropper."""
    h, w = frame.shape[:2]
    cx = int(round(max(0.0, min(1.0, x_norm)) * (w - 1)))
    cy = int(round(max(0.0, min(1.0, y_norm)) * (h - 1)))
    x0, x1 = max(0, cx - radius_px), min(w, cx + radius_px + 1)
    y0, y1 = max(0, cy - radius_px), min(h, cy + radius_px + 1)
    patch = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # Circular mean for hue so red wrapping across 0°/360° averages cleanly.
    angles = hsv[:, 0].astype(np.float64) * (2.0 * np.pi / 180.0)
    mean_angle = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
    h_deg = (np.degrees(mean_angle) + 360.0) % 360.0
    s_pct = float(np.median(hsv[:, 1])) / 255.0 * 100.0
    v_pct = float(np.median(hsv[:, 2])) / 255.0 * 100.0
    tup = (round(h_deg, 1), round(s_pct, 1), round(v_pct, 1))
    return {
        "hsv": list(tup),
        "hex": color_hex(None, tup),
        "nearest_name": nearest_color_name(*tup),
    }


def color_specs(plates: Iterable[Plate]) -> list[ColorSpec]:
    """Group configured plates by their tape color key."""
    grouped: dict[str, list[Plate]] = {}
    for p in plates:
        key = p.color_key
        if key is None:
            continue
        grouped.setdefault(key, []).append(p)
    specs: list[ColorSpec] = []
    for key, plist in grouped.items():
        # Prefer explicit eyedropped HSV if any plate in this group has it; else fall back to named preset.
        cal = next((p.color_hsv for p in plist if p.color_hsv is not None), None)
        if cal is not None:
            h_deg, s_pct, v_pct = cal
            is_white = s_pct < 20.0 and v_pct >= 62.0
            calibrated = True
        elif key in NAMED_COLORS:
            h_deg, s_pct, v_pct, _, is_white = NAMED_COLORS[key]
            calibrated = False
        else:
            continue
        thicknesses = [p.thickness_mm for p in plist if p.thickness_mm > 0]
        specs.append(ColorSpec(
            key=key,
            h_deg=h_deg,
            s_pct=s_pct,
            v_pct=v_pct,
            is_white=is_white,
            weights=tuple(sorted({p.weight for p in plist}, reverse=True)),
            min_thickness_mm=min(thicknesses) if thicknesses else 0.0,
            calibrated=calibrated,
        ))
    return specs


def _match_spec(hsv: np.ndarray, spec: ColorSpec, valid_px: np.ndarray, th: Thresholds) -> np.ndarray:
    """Return per-pixel match strength in [0, 1] for [spec] on an OpenCV HSV image."""
    h_deg = hsv[:, :, 0].astype(np.float32) * 2.0
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    if spec.is_white:
        ok = valid_px & (s <= 50.0) & (v >= 180.0)
        return ok.astype(np.float32)
    dh = np.abs(h_deg - spec.h_deg)
    dh = np.minimum(dh, 360.0 - dh)
    tol = max(6.0, float(th.color_hue_tolerance))
    # Full match within tol, smooth linear falloff out to 1.5 * tol.
    hue_score = np.clip((1.5 * tol - dh) / (0.5 * tol), 0.0, 1.0)
    min_s = float(th.color_min_saturation)
    if spec.calibrated:
        # Adapt to lower-saturation lighting if the user eyedropped the real tape under gym lighting.
        min_s = min(min_s, max(25.0, spec.s_pct * 2.55 * 0.45))
    min_v = float(th.color_min_value)
    if spec.calibrated:
        min_v = min(min_v, max(25.0, spec.v_pct * 2.55 * 0.45))
    ok = valid_px & (s >= min_s) & (v >= min_v) & (hue_score > 0.0)
    return np.where(ok, hue_score, 0.0).astype(np.float32)


def detect_colors(strip: Strip, plates: Iterable[Plate], reference: Strip | None = None,
                  thresholds: Thresholds = Thresholds()) -> ColorProfile | None:
    """Detect colored electrical tape bands along [strip]. Returns None if no plates have tape colors configured."""
    specs = color_specs(plates)
    if not specs or strip.image.ndim != 3:
        return None
    blur = cv2.GaussianBlur(strip.image, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    glare = glare_mask(strip) > 0
    valid_px = ~glare
    if reference is not None and reference.image.shape == strip.image.shape:
        ref_blur = cv2.GaussianBlur(reference.image, (3, 3), 0)
        delta = max(15, thresholds.plate_pixel_delta // 2)
        changed = cv2.absdiff(blur, ref_blur).max(axis=2) > delta
        valid_px = valid_px & changed

    h, w = strip.image.shape[:2]
    mm_per_px = 1.0 / strip.px_per_mm
    smooth_k = max(1, int(round(strip.px_per_mm * 3.0)) | 1)
    kernel = np.ones(smooth_k, dtype=np.float32) / smooth_k

    col_scores: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for spec in specs:
        m = _match_spec(hsv, spec, valid_px, thresholds)
        masks[spec.key] = (np.clip(m, 0.0, 1.0) * 255).astype(np.uint8)
        raw_col = m.mean(axis=0)
        col_scores[spec.key] = np.convolve(raw_col, kernel, mode="same") if w >= smooth_k else raw_col

    min_frac = max(0.02, float(thresholds.color_min_fraction))
    keys = [s.key for s in specs]
    spec_by_key = {s.key: s for s in specs}
    score_mat = np.stack([col_scores[k] for k in keys], axis=0)  # (C, W)
    best_idx = np.argmax(score_mat, axis=0)
    best_val = score_mat[best_idx, np.arange(w)]

    raw_cols: list[str | None] = [keys[int(i)] if float(v) >= min_frac else None for i, v in zip(best_idx, best_val)]

    # Bridge tiny 1..3 mm speckles inside a tape band without bridging across separate plates.
    max_gap_px = max(1, int(round(strip.px_per_mm * 3.0)))
    cols = list(raw_cols)
    i = 0
    while i < w:
        if cols[i] is None:
            j = i
            while j < w and cols[j] is None:
                j += 1
            if i > 0 and j < w and cols[i - 1] == cols[j] and (j - i) <= max_gap_px:
                for k in range(i, j):
                    cols[k] = cols[i - 1]
            i = j
        else:
            i += 1

    min_band_px = max(2, int(round(strip.px_per_mm * 4.0)))
    bands: list[ColorBand] = []
    i = 0
    while i < w:
        c = cols[i]
        if c is None:
            i += 1
            continue
        j = i
        while j < w and cols[j] == c:
            j += 1
        if j - i >= min_band_px:
            start_mm = i * mm_per_px
            end_mm = j * mm_per_px
            width_mm = end_mm - start_mm
            cov = float(col_scores[c][i:j].mean())
            t = spec_by_key[c].min_thickness_mm
            # Single narrow strip of electrical tape (~19 mm) on a thicker plate -> 1 plate.
            # Full-width colored rims touching across multiple identical plates -> width / thickness.
            est = max(1, int(round(width_mm / t))) if (t > 0 and width_mm > 1.5 * t) else 1
            bands.append(ColorBand(c, start_mm, end_mm, cov, est))
        else:
            for k in range(i, j):
                cols[k] = None
        i = j

    width_by_color: dict[str, float] = {k: 0.0 for k in keys}
    count_by_color: dict[str, int] = {k: 0 for k in keys}
    for b in bands:
        width_by_color[b.color] += b.width_mm
        count_by_color[b.color] += b.estimated_plates

    return ColorProfile(
        bands=tuple(bands),
        column_colors=tuple(cols),
        width_by_color_mm=width_by_color,
        count_by_color=count_by_color,
        mm_per_px=mm_per_px,
        configured_colors=tuple(keys),
        masks=masks,
    )
