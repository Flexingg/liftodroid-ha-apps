"""Lens undistortion, ROI masks and the rectified sleeve strip."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Lens, Point, Sleeve


class Undistorter:
    """Removes fisheye / barrel distortion. Maps are built once per frame size, so each frame is one remap."""

    def __init__(self, lens: Lens):
        self.lens = lens
        self._maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.lens.model == "none":
            return frame
        h, w = frame.shape[:2]
        maps = self._maps.get((w, h))
        if maps is None:
            maps = self._build(w, h)
            self._maps[(w, h)] = maps
        return cv2.remap(frame, maps[0], maps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _build(self, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        k = np.array(self.lens.camera_matrix, dtype=np.float64)
        if self.lens.calibrated_size and tuple(self.lens.calibrated_size) != (w, h):
            sx, sy = w / self.lens.calibrated_size[0], h / self.lens.calibrated_size[1]
            k = k.copy()
            k[0, 0] *= sx
            k[0, 2] *= sx
            k[1, 1] *= sy
            k[1, 2] *= sy
        d = np.array(self.lens.dist_coeffs, dtype=np.float64)
        if self.lens.model == "fisheye":
            d = d.reshape(4, 1) if d.size == 4 else np.zeros((4, 1))
            new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, (w, h), np.eye(3), balance=self.lens.balance)
            return cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), new_k, (w, h), cv2.CV_16SC2)
        new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, (w, h), 0)
        return cv2.initUndistortRectifyMap(k, d, None, new_k, (w, h), cv2.CV_16SC2)


def to_px(points: tuple[Point, ...] | list[Point], w: int, h: int) -> np.ndarray:
    return np.array([[p[0] * w, p[1] * h] for p in points], dtype=np.float32)


def polygon_mask(shape: tuple[int, int], polygon: tuple[Point, ...]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [to_px(polygon, w, h).astype(np.int32)], 255)
    return mask


def polygon_crop(frame: np.ndarray, polygon: tuple[Point, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The polygon's bounding box, and the polygon's mask inside it."""
    h, w = frame.shape[:2]
    pts = to_px(polygon, w, h).astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    x, y = max(x, 0), max(y, 0)
    bw, bh = min(bw, w - x), min(bh, h - y)
    crop = frame[y:y + bh, x:x + bw]
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [pts - [x, y]], 255)
    return crop, mask


@dataclass(frozen=True)
class Strip:
    """The sleeve, rotated so the collar is at x=0 and the tip at x=length, centred vertically."""

    image: np.ndarray
    px_per_mm: float
    # Affine that maps frame pixels into the strip (2x3), for drawing back on the frame.
    to_strip: np.ndarray

    @property
    def length(self) -> int:
        return self.image.shape[1]


def sleeve_strip(frame: np.ndarray, sleeve: Sleeve, max_plate_diameter_mm: float) -> Strip:
    h, w = frame.shape[:2]
    (x0, y0), (x1, y1) = to_px(sleeve.axis, w, h)
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 4:
        raise ValueError("sleeve axis is too short; calibrate the sleeve again")
    px_per_mm = length / sleeve.sleeve_length_mm
    half = int(math.ceil(max_plate_diameter_mm * px_per_mm * 0.6)) + 4
    angle = math.atan2(y1 - y0, x1 - x0)
    c, s = math.cos(-angle), math.sin(-angle)
    # Rotate about the collar, then move the collar to (0, half).
    m = np.array([[c, -s, -(c * x0 - s * y0)], [s, c, -(s * x0 + c * y0) + half]], dtype=np.float32)
    strip = cv2.warpAffine(frame, m, (int(round(length)), 2 * half), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Strip(strip, px_per_mm, m)
