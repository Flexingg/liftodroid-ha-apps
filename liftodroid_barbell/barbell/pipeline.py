"""One frame in, one Reading (and an annotated debug image) out."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

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

    def _side(self, frame: np.ndarray, sleeve: Sleeve) -> tuple[SideEvidence, Strip, Profile | None]:
        strip = sleeve_strip(frame, sleeve, self._max_d)
        ref = sleeve_strip(self.reference, sleeve, self._max_d) if self.reference is not None else None
        prof = profile(strip, ref, self.cfg.thresholds.plate_pixel_delta, min_plate_mm=0.25 * min(
            (p.diameter_mm for p in self.cfg.plates if p.diameter_mm > 0), default=160.0))
        dens: list[tuple[Denomination, float | None]] = []
        if self.reader is not None:
            crop, _ = polygon_crop(frame, sleeve.roi)
            h, w = frame.shape[:2]
            x0, y0 = to_px(sleeve.roi, w, h).min(axis=0)
            hits = read_all_orientations(self.reader, crop, offset=(max(float(x0), 0.0), max(float(y0), 0.0)))
            for d in denominations(hits, [p.weight for p in self.cfg.plates], self.cfg.unit, self.cfg.thresholds.ocr_min_score):
                xs, ys = strip.to_strip @ np.array([d.centre[0], d.centre[1], 1.0], dtype=np.float32)
                on_sleeve = 0 <= xs <= strip.length and 0 <= ys <= strip.image.shape[0]
                dens.append((d, float(xs) / strip.px_per_mm if on_sleeve else None))
        return SideEvidence(prof if prof.seen_any or prof.valid.any() else None, tuple(dens)), strip, prof

    def analyze(self, frame: np.ndarray, expected_total: float | None = None) -> Analysis:
        und = self.undistort(frame)
        side_a, strip_a, prof_a = self._side(und, self.cfg.sleeve)
        side_b = self._side(und, self.cfg.sleeve_b)[0] if self.cfg.sleeve_b is not None else None
        reading = fuse(self.cfg, side_a, side_b, expected_total)
        return Analysis(reading, und, annotate(und, self.cfg, strip_a, prof_a, reading), prof_a)


def annotate(frame: np.ndarray, cfg: Config, strip: Strip, prof: Profile | None, r: Reading) -> np.ndarray:
    out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    for poly, colour in ((cfg.sleeve.roi, (0, 200, 255)), (cfg.storage_roi, (180, 180, 180)), (cfg.bench_roi, (180, 120, 60))):
        if poly:
            cv2.polylines(out, [to_px(poly, w, h).astype(np.int32)], True, colour, 2)
    (ax0, ay0), (ax1, ay1) = to_px(cfg.sleeve.axis, w, h)
    cv2.arrowedLine(out, (int(ax0), int(ay0)), (int(ax1), int(ay1)), (0, 255, 0), 2, tipLength=0.05)
    if prof is not None:
        inv = cv2.invertAffineTransform(strip.to_strip)
        mid = strip.image.shape[0] / 2
        for i, run in enumerate(prof.runs):
            x0, x1 = run.start_mm * strip.px_per_mm, run.end_mm * strip.px_per_mm
            half = run.height_mm * strip.px_per_mm / 2
            box = np.array([[x0, mid - half], [x1, mid - half], [x1, mid + half], [x0, mid + half]], dtype=np.float32)
            pts = cv2.transform(box.reshape(-1, 1, 2), inv).reshape(-1, 2).astype(np.int32)
            cv2.polylines(out, [pts], True, (255, 0, 255) if i % 2 else (255, 160, 0), 2)
    label = "?" if r.detected_weight is None else f"{r.detected_weight:g} {r.unit}  {int(r.confidence * 100)}%"
    side = " + ".join(f"{p:g}" for p in r.per_side_breakdown) or "empty"
    for i, text in enumerate((label, f"per side: {side}", r.status)):
        y = 34 + i * 30
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out
