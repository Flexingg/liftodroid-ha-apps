"""Reading the white lettering on black plates.

The reader itself is swappable (RapidOCR / PaddleOCR on CPU in the app; a fake in tests). This module prepares the
image for white-on-black text, suppresses glare, tries every orientation (plates face any way from overhead), and
turns raw text into plate denominations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class TextHit:
    text: str
    score: float
    # Box centre in the image that was read (pixels), before rotation was undone.
    centre: tuple[float, float]


@dataclass(frozen=True)
class Denomination:
    weight: float
    score: float
    text: str
    # Where on the frame it was read (pixels), for placing it on the sleeve.
    centre: tuple[float, float]


class TextReader(Protocol):
    def read(self, image: np.ndarray) -> list[TextHit]: ...


class RapidOcrReader:
    """PP-OCR models through ONNX Runtime, CPU only. Imported lazily: tests and calibration don't need it."""

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        self._engine = RapidOCR()

    def read(self, image: np.ndarray) -> list[TextHit]:
        result, _ = self._engine(image)
        hits = []
        for box, text, score in result or []:
            pts = np.array(box, dtype=np.float32)
            hits.append(TextHit(str(text), float(score), (float(pts[:, 0].mean()), float(pts[:, 1].mean()))))
        return hits


def prepare(image: np.ndarray) -> np.ndarray:
    """White lettering on black: lift local contrast, keep the bright thin strokes, paint over blown-out glare."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    hot = (gray >= 250).astype(np.uint8) * 255
    blobs = cv2.morphologyEx(hot, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if blobs.any():
        gray = cv2.inpaint(gray, blobs, 5, cv2.INPAINT_TELEA)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    out = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    # Readers are trained on dark text on light paper.
    return cv2.bitwise_not(out)


_ROTATIONS = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _unrotate(c: tuple[float, float], rot, w: int, h: int) -> tuple[float, float]:
    x, y = c
    if rot is None:
        return x, y
    if rot == cv2.ROTATE_90_CLOCKWISE:  # rotated image is h wide, w tall
        return y, h - 1 - x
    if rot == cv2.ROTATE_180:
        return w - 1 - x, h - 1 - y
    return w - 1 - y, x


def read_all_orientations(reader: TextReader, image: np.ndarray, offset: tuple[float, float] = (0.0, 0.0)) -> list[TextHit]:
    prepared = prepare(image)
    h, w = prepared.shape[:2]
    hits: list[TextHit] = []
    for rot in _ROTATIONS:
        img = prepared if rot is None else cv2.rotate(prepared, rot)
        for hit in reader.read(img):
            x, y = _unrotate(hit.centre, rot, w, h)
            hits.append(TextHit(hit.text, hit.score, (x + offset[0], y + offset[1])))
    return hits


_NUMBER = re.compile(r"(\d{1,3}(?:[.,]\d{1,2})?)\s*(LBS?|KGS?|POUNDS?)?", re.IGNORECASE)
# Characters OCR commonly mistakes for digits on stencil fonts.
_FIX = str.maketrans({"O": "0", "o": "0", "S": "5", "s": "5", "I": "1", "l": "1", "Z": "2"})
_LEAD = re.compile(r"^[0-9OoSsIlZ.,]+")


def denominations(hits: list[TextHit], plate_weights: list[float], unit: str, min_score: float) -> list[Denomination]:
    """Plate weights named in the text. "45LB", "45 LBS", "25", "2.5", "20KG"; brand text like "REP FITNESS" is
    ignored. A unit that disagrees with the gym's ("20KG" in an lb gym) is dropped. Duplicates across rotations
    keep their best score."""
    best: dict[tuple[float, int, int], Denomination] = {}
    for h in hits:
        if h.score < min_score:
            continue
        raw = h.text.strip()
        # Fix look-alike letters only in the leading number ("4S" -> "45"), never in the unit ("LBS" stays).
        lead = _LEAD.match(raw)
        text = lead.group().translate(_FIX) + raw[lead.end():] if lead and any(c.isdigit() for c in lead.group()) else raw
        for m in _NUMBER.finditer(text):
            value = float(m.group(1).replace(",", "."))
            u = (m.group(2) or "").lower()
            if u.startswith("kg") and unit == "lb" or u.startswith(("lb", "pound")) and unit == "kg":
                continue
            match = next((w for w in plate_weights if abs(w - value) < 1e-6), None)
            if match is None:
                continue
            # Same plate read in several rotations lands on the same spot: one hit per ~20px cell.
            key = (match, int(h.centre[0] // 20), int(h.centre[1] // 20))
            score = h.score * (1.0 if u else 0.85)  # a unit next to the number makes it far less likely to be noise
            if key not in best or best[key].score < score:
                best[key] = Denomination(match, score, raw, h.centre)
    return sorted(best.values(), key=lambda d: -d.score)
