"""Motion gating with a settle timer: read the bar only once it has been still for a moment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class GateEvent(Enum):
    IDLE = "idle"          # nothing changed since the last reading
    MOVING = "moving"      # something is moving in the sleeve ROI
    SETTLING = "settling"  # it stopped; waiting out the settle time
    SETTLED = "settled"    # quiet for settle_s after movement: read now (fires once)


@dataclass
class MotionGate:
    pixel_delta: int = 18
    fraction: float = 0.015
    settle_s: float = 1.5

    def __post_init__(self) -> None:
        self._prev: np.ndarray | None = None
        self._last_motion: float | None = None
        self._dirty = True  # first quiet stretch after start counts as "settled"
        self.last_motion_ts: float | None = None
        self.moved_fraction = 0.0

    def reset(self) -> None:
        self.__post_init__()

    def force(self) -> None:
        """"Detect now": the next quiet frame fires SETTLED even without movement."""
        self._dirty = True
        self._last_motion = None

    def feed(self, t: float, gray: np.ndarray, mask: np.ndarray | None = None) -> GateEvent:
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        prev, self._prev = self._prev, g
        if prev is None or prev.shape != g.shape:
            self._last_motion = t
            self._dirty = True
            return GateEvent.MOVING
        diff = cv2.absdiff(g, prev) > self.pixel_delta
        if mask is not None:
            area = max(int(np.count_nonzero(mask)), 1)
            moved = int(np.count_nonzero(diff & (mask > 0)))
        else:
            area = diff.size
            moved = int(np.count_nonzero(diff))
        self.moved_fraction = moved / area
        if self.moved_fraction > self.fraction:
            self._last_motion = t
            self.last_motion_ts = t
            self._dirty = True
            return GateEvent.MOVING
        if not self._dirty:
            return GateEvent.IDLE
        since = t - (self._last_motion if self._last_motion is not None else t - self.settle_s)
        if since >= self.settle_s:
            self._dirty = False
            return GateEvent.SETTLED
        return GateEvent.SETTLING
