"""Colored electrical tape detection along the barbell sleeve.

Wrap a strip of colored electrical tape around the outer circumference (rim) of each plate denomination (for example:
red = 45, blue = 25, green = 10). Even when plates sit behind the outer plate and have identical bumper diameters,
their colored rims remain visible along the sleeve from overhead and 3/4 angles.

Uses per-frame auto-white-balanced CIE L*a*b* chromaticity (which separates brightness/exposure from color hue and
stays consistent across dark garage lighting, bright LEDs, and overhead glare) and samples the central vertical band
of the sleeve strip (where curved 3/4 plate arcs are nearly vertical and top/bottom glare and background clutter are
excluded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .config import Plate, Thresholds
from .geometry import Strip
from .slabs import glare_mask

# Named vinyl electrical tape presets: (H_deg 0..360, S_pct, V_pct, UI hex, is_white, Lab_angle_deg)
# Empirical Lab angles match real vinyl tape under dark, bright, and glare-washed gym lighting
# (sRGB pure red maps to ~35° in Lab, whereas real crimson/glared red tape centers around ~16°).
NAMED_COLORS: dict[str, tuple[float, float, float, str, bool, float]] = {
    "red": (0.0, 85.0, 75.0, "#d92626", False, 16.0),
    "orange": (24.0, 90.0, 90.0, "#e67300", False, 54.0),
    "yellow": (55.0, 88.0, 92.0, "#ebd417", False, 96.0),
    "green": (130.0, 78.0, 68.0, "#1fa83b", False, 146.0),
    "cyan": (185.0, 80.0, 80.0, "#19b8cc", False, 205.0),
    "blue": (220.0, 82.0, 78.0, "#2161c7", False, 282.0),
    "purple": (280.0, 72.0, 72.0, "#8a33b8", False, 316.0),
    "pink": (325.0, 75.0, 85.0, "#d93693", False, 344.0),
    "white": (0.0, 0.0, 94.0, "#f0f0f0", True, 0.0),
}


@dataclass(frozen=True)
class ColorSpec:
    key: str
    h_deg: float
    s_pct: float
    v_pct: float
    lab_angle: float
    lab_chroma: float
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


def _hsv_to_bgr_pixel(h_deg: float, s_pct: float, v_pct: float) -> np.ndarray:
    h = int(round((h_deg % 360.0) / 2.0)) % 180
    s = int(round(max(0.0, min(100.0, s_pct)) * 2.55))
    v = int(round(max(0.0, min(100.0, v_pct)) * 2.55))
    return cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]


def _hsv_to_lab_chroma_angle(h_deg: float, s_pct: float, v_pct: float) -> tuple[float, float]:
    """Convert (H_deg, S_pct, V_pct) into CIE L*a*b* (chroma, angle_deg)."""
    bgr = _hsv_to_bgr_pixel(h_deg, s_pct, v_pct).reshape(1, 1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    a, b = float(lab[1]) - 128.0, float(lab[2]) - 128.0
    chroma = float(np.hypot(a, b))
    angle = (float(np.degrees(np.arctan2(b, a))) + 360.0) % 360.0
    return chroma, angle


def color_hex(name: str | None, hsv: tuple[float, float, float] | None = None) -> str:
    if hsv is not None:
        bgr = _hsv_to_bgr_pixel(*hsv)
        return f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"
    if name and name.lower() in NAMED_COLORS:
        return NAMED_COLORS[name.lower()][3]
    return "#888888"


def color_bgr(name: str | None, hsv: tuple[float, float, float] | None = None) -> tuple[int, int, int]:
    hx = color_hex(name, hsv).lstrip("#")
    return (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))


def _white_balanced_lab(bgr: np.ndarray, roi_slice: slice | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert BGR image to auto-white-balanced CIE L*a*b* channels: (L, ac, bc, chroma, angle_deg).

    Estimates the neutral white-balance bias (a0, b0) from the lowest-chroma half of pixels (black rubber plates,
    chrome bar, and neutral gym surfaces) so warm/cool LEDs and daylight shifts cancel out.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch = lab[:, :, 0]
    a_raw = lab[:, :, 1] - 128.0
    b_raw = lab[:, :, 2] - 128.0
    a_ref = a_raw[roi_slice, :] if roi_slice is not None else a_raw
    b_ref = b_raw[roi_slice, :] if roi_slice is not None else b_raw
    raw_c = np.hypot(a_ref, b_ref)
    med_c = float(np.percentile(raw_c, 50))
    neut = raw_c <= max(med_c, 4.0)
    a0 = float(np.clip(np.median(a_ref[neut]), -14.0, 14.0)) if neut.any() else 0.0
    b0 = float(np.clip(np.median(b_ref[neut]), -14.0, 14.0)) if neut.any() else 0.0
    ac = a_raw - a0
    bc = b_raw - b0
    chroma = np.hypot(ac, bc)
    angle = (np.degrees(np.arctan2(bc, ac)) + 360.0) % 360.0
    return l_ch, ac, bc, chroma, angle


def nearest_color_name(h_deg: float, s_pct: float, v_pct: float) -> str:
    chroma, angle = _hsv_to_lab_chroma_angle(h_deg, s_pct, v_pct)
    if chroma < 9.0 and v_pct >= 62.0:
        return "white"
    best_name, best_dist = "red", 999.0
    for name, (_, _, _, _, is_white, n_ang) in NAMED_COLORS.items():
        if is_white:
            continue
        d = abs(angle - n_ang)
        d = min(d, 360.0 - d)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def sample_hsv_patch(frame: np.ndarray, x_norm: float, y_norm: float, radius_px: int = 6) -> dict:
    """Sample median color around normalized (x_norm, y_norm) in a BGR frame for the UI eyedropper,
    applying the same neutral Lab white-balance used during detection so picks transfer across lighting."""
    h, w = frame.shape[:2]
    cx = int(round(max(0.0, min(1.0, x_norm)) * (w - 1)))
    cy = int(round(max(0.0, min(1.0, y_norm)) * (h - 1)))
    x0, x1 = max(0, cx - radius_px), min(w, cx + radius_px + 1)
    y0, y1 = max(0, cy - radius_px), min(h, cy + radius_px + 1)

    l_ch, ac, bc, _, _ = _white_balanced_lab(frame)
    pl = float(np.median(l_ch[y0:y1, x0:x1]))
    pa = float(np.median(ac[y0:y1, x0:x1]))
    pb = float(np.median(bc[y0:y1, x0:x1]))
    lab_px = np.array([[[np.clip(pl, 0, 255), np.clip(pa + 128.0, 0, 255), np.clip(pb + 128.0, 0, 255)]]], dtype=np.uint8)
    bgr_px = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)
    hsv_px = cv2.cvtColor(bgr_px, cv2.COLOR_BGR2HSV)[0, 0]

    h_deg = float(hsv_px[0]) * 2.0
    s_pct = float(hsv_px[1]) / 255.0 * 100.0
    v_pct = float(hsv_px[2]) / 255.0 * 100.0
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
        cal = next((p.color_hsv for p in plist if p.color_hsv is not None), None)
        if cal is not None:
            h_deg, s_pct, v_pct = cal
            lab_c, lab_ang = _hsv_to_lab_chroma_angle(h_deg, s_pct, v_pct)
            is_white = (key == "white") or (lab_c < 9.0 and v_pct >= 62.0)
            calibrated = True
        elif key in NAMED_COLORS:
            h_deg, s_pct, v_pct, _, is_white, lab_ang = NAMED_COLORS[key]
            lab_c, _ = _hsv_to_lab_chroma_angle(h_deg, s_pct, v_pct)
            calibrated = False
        else:
            continue
        thicknesses = [p.thickness_mm for p in plist if p.thickness_mm > 0]
        specs.append(ColorSpec(
            key=key,
            h_deg=h_deg,
            s_pct=s_pct,
            v_pct=v_pct,
            lab_angle=lab_ang,
            lab_chroma=lab_c,
            is_white=is_white,
            weights=tuple(sorted({p.weight for p in plist}, reverse=True)),
            min_thickness_mm=min(thicknesses) if thicknesses else 0.0,
            calibrated=calibrated,
        ))
    return specs


def _match_spec(l_ch: np.ndarray, chroma: np.ndarray, angle: np.ndarray, spec: ColorSpec,
                valid_px: np.ndarray, th: Thresholds) -> np.ndarray:
    """Return per-pixel match strength in [0, 1] for [spec] using white-balanced CIE L*a*b*."""
    if spec.is_white:
        # Strict white: bright non-glare neutral pixels only (avoids specular reflections & gray rubber).
        ok = valid_px & (chroma <= 7.5) & (l_ch >= 205.0) & (l_ch <= 246.0)
        return ok.astype(np.float32)

    d_ang = np.abs(angle - spec.lab_angle)
    d_ang = np.minimum(d_ang, 360.0 - d_ang)
    # Lab hue angles are perceptually uniform; slightly wider tolerance covers dark -> bright glare shifts cleanly.
    tol = max(10.0, float(th.color_hue_tolerance) * 1.15)
    hue_score = np.clip((1.4 * tol - d_ang) / (0.4 * tol), 0.0, 1.0)

    # Map the 0..255 slider (default 55) to a Lab chroma threshold (default ~13.5, safely above neutral rubber ~2..9
    # yet well below glare-washed tape ~16..50). Cool daylight shadows drift slightly toward blue (-b, ~270°), so
    # require a hair more chroma there unless eyedropped lower.
    min_chroma = max(7.0, float(th.color_min_saturation) * 0.245)
    if 240.0 <= spec.lab_angle <= 300.0:
        min_chroma += 2.0
    if spec.calibrated and spec.lab_chroma > 0:
        min_chroma = min(min_chroma, max(9.5, spec.lab_chroma * 0.45))
    min_l = max(15.0, float(th.color_min_value) * 0.45)
    ok = valid_px & (chroma >= min_chroma) & (l_ch >= min_l) & (l_ch <= 248.0) & (hue_score > 0.0)
    return np.where(ok, hue_score, 0.0).astype(np.float32)


def _exposure_matched_change_mask(blur: np.ndarray, ref_blur: np.ndarray, delta: int) -> np.ndarray:
    """Foreground change mask invariant to global brightness/exposure shifts between reference and live frame."""
    h, w = blur.shape[:2]
    margin = max(2, int(round(h * 0.12)))
    tip = max(2, int(round(w * 0.15)))
    bg = np.zeros((h, w), dtype=bool)
    bg[:margin, :] = True
    bg[h - margin:, :] = True
    bg[:, w - tip:] = True
    matched = ref_blur.astype(np.float32)
    cur = blur.astype(np.float32)
    for ch in range(3):
        ref_bg = matched[:, :, ch][bg]
        cur_bg = cur[:, :, ch][bg]
        r_med = float(np.median(ref_bg))
        c_med = float(np.median(cur_bg))
        if r_med > 12.0:
            gain = float(np.clip(c_med / r_med, 0.5, 2.0))
            matched[:, :, ch] = np.clip(matched[:, :, ch] * gain, 0.0, 255.0)
        else:
            matched[:, :, ch] = np.clip(matched[:, :, ch] + (c_med - r_med), 0.0, 255.0)
    return np.abs(cur - matched).max(axis=2) > delta


def detect_colors(strip: Strip, plates: Iterable[Plate], reference: Strip | None = None,
                  thresholds: Thresholds = Thresholds()) -> ColorProfile | None:
    """Detect colored electrical tape bands along [strip]. Returns None if no plates have tape colors configured."""
    specs = color_specs(plates)
    if not specs or strip.image.ndim != 3:
        return None
    h, w = strip.image.shape[:2]
    # Split the central plate region into upper & lower halves around the sleeve centerline (24%..76% of strip height).
    # Excludes the outer top/bottom tangent tails where curved 3/4 plate arcs stretch horizontally, while taking the
    # stronger half per column so one-sided ceiling glare on the rim doesn't wash out the tape color.
    y_top = int(round(h * 0.24))
    y_bot = max(y_top + 2, int(round(h * 0.76)))
    y_mid_lo = max(y_top + 1, int(round(h * 0.52)))
    y_mid_hi = min(y_bot - 1, int(round(h * 0.48)))
    wb_slice = slice(y_top, y_bot)

    blur = cv2.GaussianBlur(strip.image, (3, 3), 0)
    l_ch, cur_a, cur_b, chroma, angle = _white_balanced_lab(blur, roi_slice=wb_slice)
    glare = glare_mask(strip) > 0
    valid_px = ~glare
    if strip.roi_mask is not None and strip.roi_mask.shape == (h, w):
        valid_px = valid_px & (strip.roi_mask > 0)

    if reference is not None and reference.image.shape == strip.image.shape:
        ref_blur = cv2.GaussianBlur(reference.image, (3, 3), 0)
        _, ref_a, ref_b, _, _ = _white_balanced_lab(ref_blur, roi_slice=wb_slice)
        chroma_diff = np.hypot(cur_a - ref_a, cur_b - ref_b)
        ref_changed = _exposure_matched_change_mask(blur, ref_blur, max(16, thresholds.plate_pixel_delta // 2)) | (chroma_diff > 10.0)
        valid_px = valid_px & ref_changed

    # Exclude the shiny chrome barbell sleeve centerline when evaluating achromatic white tape.
    mm_per_px = 1.0 / strip.px_per_mm
    smooth_k = max(1, int(round(strip.px_per_mm * 3.0)) | 1)
    kernel = np.ones(smooth_k, dtype=np.float32) / smooth_k
    cy = h // 2
    bar_half = max(2, int(round(h * 0.08)))
    non_bar_full = np.ones((h, w), dtype=bool)
    non_bar_full[max(0, cy - bar_half):min(h, cy + bar_half), :] = False

    col_scores: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for spec in specs:
        m = _match_spec(l_ch, chroma, angle, spec, valid_px, thresholds)
        masks[spec.key] = (np.clip(m, 0.0, 1.0) * 255).astype(np.uint8)
        eff = np.where(non_bar_full, m, 0.0) if spec.is_white else m
        upper = eff[y_top:y_mid_lo, :].mean(axis=0)
        lower = eff[y_mid_hi:y_bot, :].mean(axis=0)
        raw_col = np.maximum(upper, lower)
        col_scores[spec.key] = np.convolve(raw_col, kernel, mode="same") if w >= smooth_k else raw_col

    min_frac = max(0.02, float(thresholds.color_min_fraction))
    keys = [s.key for s in specs]
    spec_by_key = {s.key: s for s in specs}

    # White requires higher column coverage so thin white lettering ("REP FITNESS") doesn't false-trigger.
    min_req = {k: (max(0.32, min_frac * 3.0) if spec_by_key[k].is_white else min_frac) for k in keys}
    score_mat = np.stack([col_scores[k] for k in keys], axis=0)  # (C, W)
    # Chromatic colors always take priority over achromatic white if both meet threshold.
    raw_cols: list[str | None] = []
    for x in range(w):
        best_k: str | None = None
        best_v = 0.0
        for ci, k in enumerate(keys):
            v = float(score_mat[ci, x])
            if v < min_req[k]:
                continue
            if best_k is None or (spec_by_key[best_k].is_white and not spec_by_key[k].is_white) or (
                spec_by_key[best_k].is_white == spec_by_key[k].is_white and v > best_v
            ):
                best_k = k
                best_v = v
        raw_cols.append(best_k)

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
