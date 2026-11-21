"""Synthetic overhead frames: a sleeve seen from above, plates as slabs across it."""

from __future__ import annotations

import cv2
import numpy as np

from barbell.config import Config, Plate, Sleeve, Thresholds, from_dict

W, H = 1280, 720
PX_PER_MM = 1.5
COLLAR = (300, 360)
SLEEVE_MM = 415.0


def sleeve(collar=COLLAR, direction=1) -> Sleeve:
    x0, y0 = collar
    x1 = x0 + direction * SLEEVE_MM * PX_PER_MM
    pad = 360
    xs = sorted([x0 - 10, x1 + 10])
    return Sleeve(
        axis=((x0 / W, y0 / H), (x1 / W, y0 / H)),
        roi=((xs[0] / W, (y0 - pad) / H), (xs[1] / W, (y0 - pad) / H), (xs[1] / W, (y0 + pad) / H), (xs[0] / W, (y0 + pad) / H)),
        sleeve_length_mm=SLEEVE_MM,
    )


def config(**kw) -> Config:
    from dataclasses import replace

    base = from_dict({"unit": "lb", "bar_weight": 45})
    return replace(base, sleeve=sleeve(), **kw)


def scene(cfg: Config, side: list[float], collar=COLLAR, direction=1, base: np.ndarray | None = None,
          floor: int = 120) -> np.ndarray:
    img = base.copy() if base is not None else np.full((H, W, 3), floor, dtype=np.uint8)
    rng = np.random.default_rng(1)
    if base is None:
        img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
        x0, y0 = collar
        x1 = int(x0 + direction * SLEEVE_MM * PX_PER_MM)
        cv2.rectangle(img, (min(x0, x1), y0 - 5), (max(x0, x1), y0 + 5), (205, 205, 205), -1)
    x = float(collar[0])
    y0 = collar[1]
    for w in side:
        p = cfg.plate_by_weight(w)
        t = p.thickness_mm * PX_PER_MM
        r = p.diameter_mm * PX_PER_MM / 2
        a, b = (x, x + t) if direction > 0 else (x - t, x)
        cv2.rectangle(img, (int(round(a)), int(y0 - r)), (int(round(b)) - 1, int(y0 + r)), (28, 28, 28), -1)
        # A white rim line and lettering-like strokes, which must not break the slab apart.
        cv2.line(img, (int(a) + 1, int(y0 - r) + 6), (int(a) + 1, int(y0 + r) - 6), (220, 220, 220), 1)
        x += direction * t
    return img


class FakeReader:
    """Returns the given hits on the first (unrotated) read only."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = 0

    def read(self, image):
        self.calls += 1
        return list(self.hits) if self.calls == 1 else []
