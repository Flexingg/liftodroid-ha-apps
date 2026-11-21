"""Plate slabs along the sleeve.

Seen from above, each plate on the sleeve is a slab across the strip: its width along the sleeve is the plate's
thickness, its extent across the strip is the plate's diameter. So a loaded sleeve is a height profile: tall for
45s, lower for 10s, zero past the last plate. Comparing that profile with what each candidate loading predicts is
the structural cue; it still works when lettering is hidden, glared or upside down.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Strip


@dataclass(frozen=True)
class Run:
    """A stretch of the sleeve with one plate height: one plate, or several of the same size side by side."""

    start_mm: float
    end_mm: float
    height_mm: float

    @property
    def width_mm(self) -> float:
        return self.end_mm - self.start_mm


@dataclass(frozen=True)
class Profile:
    """Plate height (mm) at every pixel column from the collar outwards; [valid] is False where glare or a hand
    makes the column unreadable (those columns are left out of every comparison)."""

    heights_mm: np.ndarray
    valid: np.ndarray
    mm_per_px: float
    stack_length_mm: float
    runs: tuple[Run, ...]
    glare_fraction: float
    mask: np.ndarray  # plate pixels in the strip, for the debug image

    @property
    def seen_any(self) -> bool:
        return self.stack_length_mm > 0 and bool(self.valid.any())

    def height_at_mm(self, x_mm: np.ndarray) -> np.ndarray:
        idx = np.clip((x_mm / self.mm_per_px).astype(int), 0, len(self.heights_mm) - 1)
        out = self.heights_mm[idx].copy()
        out[x_mm >= len(self.heights_mm) * self.mm_per_px] = 0.0
        return out

    def valid_at_mm(self, x_mm: np.ndarray) -> np.ndarray:
        idx = (x_mm / self.mm_per_px).astype(int)
        inside = idx < len(self.valid)
        out = np.ones_like(x_mm, dtype=bool)
        out[inside] = self.valid[idx[inside]]
        return out


def plate_mask(strip: Strip, reference: Strip | None, pixel_delta: int) -> np.ndarray:
    """Pixels that belong to a plate: different from the empty sleeve if we have it, otherwise just dark."""
    if reference is not None and reference.image.shape[:2] == strip.image.shape[:2]:
        if strip.image.ndim == 3 and reference.image.ndim == 3:
            diff = cv2.absdiff(cv2.GaussianBlur(strip.image, (5, 5), 0), cv2.GaussianBlur(reference.image, (5, 5), 0)).max(axis=2)
        else:
            gray = cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY) if strip.image.ndim == 3 else strip.image
            ref = cv2.cvtColor(reference.image, cv2.COLOR_BGR2GRAY) if reference.image.ndim == 3 else reference.image
            diff = cv2.absdiff(cv2.GaussianBlur(gray, (5, 5), 0), cv2.GaussianBlur(ref, (5, 5), 0))
        mask = (diff > pixel_delta).astype(np.uint8) * 255
    else:
        # Black plates: darker than the scene around them. Otsu adapts to the light in the room.
        gray = cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY) if strip.image.ndim == 3 else strip.image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # White lettering and rims leave holes in a black plate: close them, then drop specks.
    k = max(3, int(round(strip.px_per_mm * 6)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return mask


def glare_mask(strip: Strip) -> np.ndarray:
    """Blown-out pixels (light bouncing off glossy plates): unreadable, so ignored rather than trusted."""
    gray = cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY) if strip.image.ndim == 3 else strip.image
    hot = (gray >= 250).astype(np.uint8)
    # Lettering is thin; glare is a blob. Keep only blobs thicker than a few millimetres.
    k = max(3, int(round(strip.px_per_mm * 4)))
    return cv2.morphologyEx(hot, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def profile(strip: Strip, reference: Strip | None = None, pixel_delta: int = 35, min_plate_mm: float = 120.0,
            occlusion: np.ndarray | None = None) -> Profile:
    """[occlusion]: optional strip-sized mask of pixels known to be blocked (e.g. a person box)."""
    mask = plate_mask(strip, reference, pixel_delta)
    h, w = mask.shape
    mm_per_px = 1.0 / strip.px_per_mm
    centre = h // 2
    band = max(2, min(centre, int(round(strip.px_per_mm * 12))))

    # Plate extent per column: the run of plate pixels through the middle of the strip (where the sleeve is).
    heights = np.zeros(w, dtype=np.float32)
    cols = mask > 0
    for x in range(w):
        col = cols[:, x]
        hits = np.flatnonzero(col[centre - band:centre + band + 1])
        if hits.size == 0:
            continue
        top = centre - band + int(hits[0])
        while top > 0 and col[top - 1]:
            top -= 1
        bottom = centre - band + int(hits[-1])
        while bottom < h - 1 and col[bottom + 1]:
            bottom += 1
        heights[x] = (bottom - top + 1) * mm_per_px
    heights = cv2.medianBlur(heights.reshape(1, -1), 5).ravel() if w >= 5 else heights
    heights[heights < min_plate_mm] = 0.0

    glare = glare_mask(strip)
    bad = glare.any(axis=0)
    if occlusion is not None and occlusion.shape == mask.shape:
        bad |= (occlusion > 0).mean(axis=0) > 0.3
    valid = ~bad
    glare_fraction = float(glare.mean())

    # The stack runs from the collar outwards; a gap of more than ~1 cm ends it.
    gap_px = max(2, int(round(10 * strip.px_per_mm)))
    end = 0
    gap = 0
    started = False
    for x in range(w):
        if heights[x] > 0 or (not valid[x] and started):
            started = True
            end = x + 1
            gap = 0
        elif started:
            gap += 1
            if gap > gap_px:
                break
        elif x > gap_px * 3:
            break  # nothing near the collar: an empty sleeve
    heights[end:] = 0.0

    runs = _runs(heights[:end], mm_per_px)
    return Profile(heights, valid, mm_per_px, end * mm_per_px, tuple(runs), glare_fraction, mask)


def _runs(heights: np.ndarray, mm_per_px: float, rel_step: float = 0.1, min_px: int = 3) -> list[Run]:
    runs: list[Run] = []
    start = 0
    ref = None
    for x, hgt in enumerate(heights):
        if ref is None:
            if hgt > 0:
                start, ref = x, hgt
            continue
        if hgt <= 0 or abs(hgt - ref) / ref > rel_step:
            if x - start >= min_px:
                runs.append(Run(start * mm_per_px, x * mm_per_px, float(np.median(heights[start:x]))))
            start, ref = (x, hgt) if hgt > 0 else (x, None)
    if ref is not None and len(heights) - start >= min_px:
        runs.append(Run(start * mm_per_px, len(heights) * mm_per_px, float(np.median(heights[start:]))))
    return runs
