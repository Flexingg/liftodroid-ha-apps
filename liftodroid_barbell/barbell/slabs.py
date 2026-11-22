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
    start_mm: float = 0.0

    @property
    def end_mm(self) -> float:
        return self.start_mm + self.stack_length_mm

    @property
    def seen_any(self) -> bool:
        return self.stack_length_mm > 0 and bool(self.valid.any())

    def height_at_mm(self, x_mm: np.ndarray) -> np.ndarray:
        off = int(round(self.start_mm / self.mm_per_px))
        idx = off + (x_mm / self.mm_per_px).astype(int)
        inside = (idx >= 0) & (idx < len(self.heights_mm))
        out = np.zeros_like(x_mm, dtype=np.float32)
        out[inside] = self.heights_mm[idx[inside]]
        return out

    def valid_at_mm(self, x_mm: np.ndarray) -> np.ndarray:
        off = int(round(self.start_mm / self.mm_per_px))
        idx = off + (x_mm / self.mm_per_px).astype(int)
        inside = (idx >= 0) & (idx < len(self.valid))
        out = np.ones_like(x_mm, dtype=bool)
        out[inside] = self.valid[idx[inside]]
        return out


def _match_exposure(cur: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Adjust [ref] exposure/brightness to match [cur] using the strip's outer background margin
    (top/bottom border + rightmost tip) so garage light changes and camera auto-exposure don't light up the wall."""
    h, w = cur.shape[:2]
    margin = max(2, int(round(h * 0.12)))
    tip = max(2, int(round(w * 0.15)))
    bg = np.zeros((h, w), dtype=bool)
    bg[:margin, :] = True
    bg[h - margin:, :] = True
    bg[:, w - tip:] = True
    out = ref.astype(np.float32)
    cur_f = cur.astype(np.float32)
    if out.ndim == 3:
        for ch in range(3):
            r_med = float(np.median(out[:, :, ch][bg]))
            c_med = float(np.median(cur_f[:, :, ch][bg]))
            if r_med > 12.0:
                gain = float(np.clip(c_med / r_med, 0.5, 2.0))
                out[:, :, ch] = np.clip(out[:, :, ch] * gain, 0.0, 255.0)
            else:
                out[:, :, ch] = np.clip(out[:, :, ch] + (c_med - r_med), 0.0, 255.0)
    else:
        r_med = float(np.median(out[bg]))
        c_med = float(np.median(cur_f[bg]))
        if r_med > 12.0:
            gain = float(np.clip(c_med / r_med, 0.5, 2.0))
            out = np.clip(out * gain, 0.0, 255.0)
        else:
            out = np.clip(out + (c_med - r_med), 0.0, 255.0)
    return out.astype(np.uint8)


def plate_mask(strip: Strip, reference: Strip | None, pixel_delta: int) -> np.ndarray:
    """Pixels that belong to a plate: different from the exposure-matched empty sleeve if we have it, otherwise dark."""
    if reference is not None and reference.image.shape[:2] == strip.image.shape[:2]:
        if strip.image.ndim == 3 and reference.image.ndim == 3:
            cur_b = cv2.GaussianBlur(strip.image, (5, 5), 0)
            ref_b = _match_exposure(cur_b, cv2.GaussianBlur(reference.image, (5, 5), 0))
            diff = cv2.absdiff(cur_b, ref_b).max(axis=2)
        else:
            gray = cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY) if strip.image.ndim == 3 else strip.image
            ref = cv2.cvtColor(reference.image, cv2.COLOR_BGR2GRAY) if reference.image.ndim == 3 else reference.image
            cur_b = cv2.GaussianBlur(gray, (5, 5), 0)
            ref_b = _match_exposure(cur_b, cv2.GaussianBlur(ref, (5, 5), 0))
            diff = cv2.absdiff(cur_b, ref_b)
        mask = (diff > pixel_delta).astype(np.uint8) * 255
    else:
        # Black plates: darker than the scene around them. Otsu adapts to the light in the room.
        gray = cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY) if strip.image.ndim == 3 else strip.image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # White lettering, bar sleeve, and rims leave vertical holes in a black plate: close them, then drop specks.
    k = max(3, int(round(strip.px_per_mm * 14)) | 1)
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
    band = max(3, min(centre, int(round(strip.px_per_mm * 20))))
    bridge_px = max(2, int(round(strip.px_per_mm * 18)))

    # Plate extent per column: the run of plate pixels through the middle of the strip (where the sleeve is),
    # bridging small interior gaps from lettering, bar sleeve reflections, or black-on-black floor overlap.
    heights = np.zeros(w, dtype=np.float32)
    cols = mask > 0
    for x in range(w):
        col = cols[:, x]
        hits = np.flatnonzero(col[centre - band:centre + band + 1])
        if hits.size == 0:
            continue
        top = centre - band + int(hits[0])
        gap = 0
        y = top - 1
        while y >= 0:
            if col[y]:
                top = y
                gap = 0
            else:
                gap += 1
                if gap > bridge_px:
                    break
            y -= 1
        bottom = centre - band + int(hits[-1])
        gap = 0
        y = bottom + 1
        while y < h:
            if col[y]:
                bottom = y
                gap = 0
            else:
                gap += 1
                if gap > bridge_px:
                    break
            y += 1
        heights[x] = (bottom - top + 1) * mm_per_px
    heights = cv2.medianBlur(heights.reshape(1, -1), 5).ravel() if w >= 5 else heights
    heights[heights < min_plate_mm] = 0.0

    glare = glare_mask(strip)
    # Require a glare blob to cover >6% of the strip height (~35 mm) before invalidating a column, so thin specular
    # reflections along the chrome barbell sleeve never mark the entire empty sleeve unreadable.
    bad = (glare > 0).mean(axis=0) > 0.06
    if occlusion is not None and occlusion.shape == mask.shape:
        bad |= (occlusion > 0).mean(axis=0) > 0.3
    valid = ~bad
    glare_fraction = float(glare.mean())

    # Find the contiguous plate stack starting near the collar (allowing up to ~130 mm collar / J-cup offset so
    # slight sideways bar shifts in the rack never zero out the reading). Invalid (glared) columns bridge gaps
    # inside the stack without extending [end] indefinitely past the last real plate.
    gap_px = max(2, int(round(12 * strip.px_per_mm)))
    max_start_px = min(w, max(gap_px * 4, int(round(130 * strip.px_per_mm)), int(round(w * 0.35))))
    start = 0
    end = 0
    gap = 0
    started = False
    for x in range(w):
        if heights[x] > 0:
            if not started:
                start = x
                started = True
            end = x + 1
            gap = 0
        elif not valid[x] and started:
            # Glare/occlusion inside the stack: pause gap counter, wait to see if plate resumes.
            continue
        elif started:
            gap += 1
            if gap > gap_px:
                break
        elif x > max_start_px:
            break  # nothing near the collar: an empty sleeve
    if started:
        heights[:start] = 0.0
        heights[end:] = 0.0
        stack_len = (end - start) * mm_per_px
        start_mm = start * mm_per_px
    else:
        heights[:] = 0.0
        stack_len = 0.0
        start_mm = 0.0

    runs = _runs(heights[:end], mm_per_px) if started else []
    return Profile(heights, valid, mm_per_px, stack_len, tuple(runs), glare_fraction, mask, start_mm=start_mm)


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
