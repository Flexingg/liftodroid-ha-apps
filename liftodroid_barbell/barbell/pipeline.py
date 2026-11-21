"""One frame in, one Reading (and an annotated debug image) out."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .colors import ColorProfile, color_bgr, detect_colors
from .config import Config, Sleeve
from .fusion import Reading, SideEvidence, fuse
from .geometry import Strip, Undistorter, polygon_crop, sleeve_strip, to_px
from .ocr import Denomination, TextReader, denominations, read_all_orientations
from .slabs import Profile, profile


@dataclass
class Analysis:
    reading: Reading
    undistorted: np.ndarray
    annotated: np.ndarray
    profile_a: Profile | None
    strip_a: Strip | None = None
    colors_a: ColorProfile | None = None
    raw_profile_a: Profile | None = None


class Analyzer:
    def __init__(self, cfg: Config, reader: TextReader | None = None, reference: np.ndarray | None = None):
        if cfg.sleeve is None:
            raise ValueError("No sleeve configured: run calibration first")
        self.cfg = cfg
        self.reader = reader
        self.undistort = Undistorter(cfg.lens)
        # The reference is stored already undistorted (it's captured through this same pipeline).
        self.reference = reference
        self._max_d = max((p.diameter_mm for p in cfg.plates), default=450.0) or 450.0

    def with_config(self, cfg: Config) -> "Analyzer":
        out = Analyzer(cfg, self.reader, self.reference)
        out.undistort = self.undistort if cfg.lens == self.cfg.lens else out.undistort
        return out

    def _side(self, frame: np.ndarray, sleeve: Sleeve) -> tuple[SideEvidence, Strip, Profile | None, Profile, ColorProfile | None]:
        strip = sleeve_strip(frame, sleeve, self._max_d)
        ref = sleeve_strip(self.reference, sleeve, self._max_d) if self.reference is not None else None
        raw_prof = profile(strip, ref, self.cfg.thresholds.plate_pixel_delta, min_plate_mm=0.25 * min(
            (p.diameter_mm for p in self.cfg.plates if p.diameter_mm > 0), default=160.0))
        col_prof = detect_colors(strip, self.cfg.plates, ref, self.cfg.thresholds) if (
            self.cfg.use_color and self.cfg.has_tape_colors
        ) else None
        dens: list[tuple[Denomination, float | None]] = []
        if self.reader is not None and self.cfg.use_ocr:
            crop, _ = polygon_crop(frame, sleeve.roi)
            h, w = frame.shape[:2]
            x0, y0 = to_px(sleeve.roi, w, h).min(axis=0)
            hits = read_all_orientations(self.reader, crop, offset=(max(float(x0), 0.0), max(float(y0), 0.0)))
            for d in denominations(hits, [p.weight for p in self.cfg.plates], self.cfg.unit, self.cfg.thresholds.ocr_min_score):
                xs, ys = strip.to_strip @ np.array([d.centre[0], d.centre[1], 1.0], dtype=np.float32)
                on_sleeve = 0 <= xs <= strip.length and 0 <= ys <= strip.image.shape[0]
                dens.append((d, float(xs) / strip.px_per_mm if on_sleeve else None))
        use_prof = self.cfg.use_shape and (raw_prof.seen_any or raw_prof.valid.any())
        ev = SideEvidence(raw_prof if use_prof else None, tuple(dens), col_prof)
        return ev, strip, (raw_prof if self.cfg.use_shape else None), raw_prof, col_prof

    def analyze(self, frame: np.ndarray, expected_total: float | None = None) -> Analysis:
        und = self.undistort(frame)
        side_a, strip_a, prof_a, raw_prof_a, col_a = self._side(und, self.cfg.sleeve)
        side_b = self._side(und, self.cfg.sleeve_b)[0] if self.cfg.sleeve_b is not None else None
        reading = fuse(self.cfg, side_a, side_b, expected_total)
        return Analysis(
            reading=reading,
            undistorted=und,
            annotated=annotate(und, self.cfg, strip_a, prof_a, reading, col_a),
            profile_a=prof_a,
            strip_a=strip_a,
            colors_a=col_a,
            raw_profile_a=raw_prof_a,
        )


def annotate(frame: np.ndarray, cfg: Config, strip: Strip, prof: Profile | None, r: Reading,
             colors: ColorProfile | None = None) -> np.ndarray:
    out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    for poly, colour in ((cfg.sleeve.roi, (0, 200, 255)), (cfg.storage_roi, (180, 180, 180)), (cfg.bench_roi, (180, 120, 60))):
        if poly:
            cv2.polylines(out, [to_px(poly, w, h).astype(np.int32)], True, colour, 2)
    if cfg.sleeve_b is not None:
        cv2.polylines(out, [to_px(cfg.sleeve_b.roi, w, h).astype(np.int32)], True, (255, 200, 0), 2)
    (ax0, ay0), (ax1, ay1) = to_px(cfg.sleeve.axis, w, h)
    cv2.arrowedLine(out, (int(ax0), int(ay0)), (int(ax1), int(ay1)), (0, 255, 0), 2, tipLength=0.05)
    inv = cv2.invertAffineTransform(strip.to_strip)
    mid = strip.image.shape[0] / 2
    if prof is not None:
        for i, run in enumerate(prof.runs):
            x0, x1 = run.start_mm * strip.px_per_mm, run.end_mm * strip.px_per_mm
            half = run.height_mm * strip.px_per_mm / 2
            box = np.array([[x0, mid - half], [x1, mid - half], [x1, mid + half], [x0, mid + half]], dtype=np.float32)
            pts = cv2.transform(box.reshape(-1, 1, 2), inv).reshape(-1, 2).astype(np.int32)
            cv2.polylines(out, [pts], True, (255, 0, 255) if i % 2 else (255, 160, 0), 2)
    if colors is not None and colors.bands:
        half = strip.image.shape[0] * 0.36
        for band in colors.bands:
            x0, x1 = band.start_mm * strip.px_per_mm, band.end_mm * strip.px_per_mm
            box = np.array([[x0, mid - half], [x1, mid - half], [x1, mid + half], [x0, mid + half]], dtype=np.float32)
            pts = cv2.transform(box.reshape(-1, 1, 2), inv).reshape(-1, 2).astype(np.int32)
            p_match = next((p for p in cfg.plates if p.color_key == band.color), None)
            bgr = color_bgr(band.color, p_match.color_hsv if p_match else None)
            cv2.polylines(out, [pts], True, bgr, 3)
    label = "?" if r.detected_weight is None else f"{r.detected_weight:g} {r.unit}  {int(r.confidence * 100)}%"
    side = " + ".join(f"{p:g}" for p in r.per_side_breakdown) or "empty"
    for i, text in enumerate((label, f"per side: {side}", r.status)):
        y = 34 + i * 30
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def diagnostic_strip(analysis: Analysis, cfg: Config) -> np.ndarray | None:
    """Build a visual strip image showing: rectified sleeve + ruler, plate mask overlay, and detected tape colors."""
    strip = analysis.strip_a
    if strip is None:
        return None
    base = strip.image.copy() if strip.image.ndim == 3 else cv2.cvtColor(strip.image, cv2.COLOR_GRAY2BGR)
    h, w = base.shape[:2]
    # Scale to a comfortable width for the Web UI while keeping proportions clear.
    target_w = min(max(w, 420), 760)
    scale = target_w / max(w, 1)
    disp_h = max(90, min(220, int(round(h * scale))))
    top = cv2.resize(base, (target_w, disp_h), interpolation=cv2.INTER_AREA)

    # Ruler bar at the top (ticks every 50 mm from collar).
    ruler_h = 26
    ruler = np.full((ruler_h, target_w, 3), 24, dtype=np.uint8)
    total_mm = w / strip.px_per_mm
    mm = 0
    while mm <= total_mm:
        x = int(round((mm * strip.px_per_mm) * scale))
        if 0 <= x < target_w:
            cv2.line(ruler, (x, ruler_h - 10), (x, ruler_h - 1), (180, 180, 180), 1)
            if mm % 50 == 0 and x < target_w - 30:
                cv2.putText(ruler, f"{mm}", (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        mm += 25

    # Plate mask overlay (green tint over pixels classified as plate, cyan vertical line at measured stack end).
    prof = analysis.raw_profile_a or analysis.profile_a
    mask_panel = top.copy()
    if prof is not None:
        mask_r = cv2.resize(prof.mask, (target_w, disp_h), interpolation=cv2.INTER_NEAREST) > 0
        tint = mask_panel.copy()
        tint[mask_r] = (60, 210, 90)
        mask_panel = cv2.addWeighted(mask_panel, 0.55, tint, 0.45, 0)
        end_x = int(round((prof.stack_length_mm * strip.px_per_mm) * scale))
        if 0 < end_x < target_w:
            cv2.line(mask_panel, (end_x, 0), (end_x, disp_h - 1), (0, 255, 255), 2)
        cv2.putText(mask_panel, f"Plate mask (stack: {prof.stack_length_mm:.0f} mm)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(mask_panel, f"Plate mask (stack: {prof.stack_length_mm:.0f} mm)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(top, "Rectified sleeve (collar -> tip)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(top, "Rectified sleeve (collar -> tip)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    panels = [ruler, top, mask_panel]

    # Color tape bar (shown whenever tape colors are configured).
    col = analysis.colors_a
    if col is not None:
        bar_h = 38
        bar = np.full((bar_h, target_w, 3), 32, dtype=np.uint8)
        for band in col.bands:
            x0 = max(0, int(round((band.start_mm * strip.px_per_mm) * scale)))
            x1 = min(target_w - 1, int(round((band.end_mm * strip.px_per_mm) * scale)))
            p_match = next((p for p in cfg.plates if p.color_key == band.color), None)
            bgr = color_bgr(band.color, p_match.color_hsv if p_match else None)
            cv2.rectangle(bar, (x0, 4), (max(x0 + 2, x1), bar_h - 4), bgr, -1)
            cv2.rectangle(bar, (x0, 4), (max(x0 + 2, x1), bar_h - 4), (255, 255, 255), 1)
            tag = f"{band.color} x{band.estimated_plates}"
            cv2.putText(bar, tag, (x0 + 4, bar_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bar, tag, (x0 + 4, bar_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if not col.bands:
            cv2.putText(bar, "Rim tape: no colored bands detected", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
        panels.append(bar)

    return np.vstack(panels)
