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
    """The sleeve, rotated so the collar->tip axis runs left-to-right, centred vertically.

    [collar_x_px] is the collar's x coordinate in the strip (0 when the ROI starts at the collar, >0 when the ROI
    extends behind the collar to include the visible top rim of plates on overhead/3-4 cameras).
    [axis_length_px] is the collar-to-tip pixel length used to scale [sleeve_length_mm].
    [roi_mask] is an optional HxW uint8 mask (0/255) of the user's sleeve.roi inside the strip.
    """

    image: np.ndarray
    px_per_mm: float
    # Affine that maps frame pixels into the strip (2x3), for drawing back on the frame.
    to_strip: np.ndarray
    collar_x_px: float = 0.0
    axis_length_px: float | None = None
    roi_mask: np.ndarray | None = None

    @property
    def length(self) -> int:
        return self.image.shape[1]

    @property
    def axis_px(self) -> float:
        return self.axis_length_px if self.axis_length_px is not None else float(self.image.shape[1])


def sleeve_strip(frame: np.ndarray, sleeve: Sleeve, max_plate_diameter_mm: float) -> Strip:
    h, w = frame.shape[:2]
    (x0, y0), (x1, y1) = to_px(sleeve.axis, w, h)
    axis_len = math.hypot(x1 - x0, y1 - y0)
    if axis_len < 4:
        raise ValueError("sleeve axis is too short; calibrate the sleeve again")
    px_per_mm = axis_len / sleeve.sleeve_length_mm
    half = int(math.ceil(max_plate_diameter_mm * px_per_mm * 0.6)) + 4
    angle = math.atan2(y1 - y0, x1 - x0)
    c, s = math.cos(-angle), math.sin(-angle)

    # Project the user's ROI into strip space so plates & rim tape behind the collar (overhead / 3-4 views)
    # are never clipped off the left, top, or bottom of the strip.
    pad_left = 0.0
    strip_right = axis_len
    roi_px = to_px(sleeve.roi, w, h) if sleeve.roi else None
    if roi_px is not None and len(roi_px) >= 3:
        dx = roi_px[:, 0] - x0
        dy = roi_px[:, 1] - y0
        rx = c * dx - s * dy
        ry = s * dx + c * dy
        min_rx, max_rx = float(rx.min()), float(rx.max())
        max_abs_ry = float(np.abs(ry).max())
        if min_rx < -15.0:
            pad_left = min(math.ceil(-min_rx), axis_len * 1.5)
        if max_rx > axis_len + 15.0:
            strip_right = min(math.ceil(max_rx), axis_len * 1.5)
        if max_abs_ry + 4 > half:
            half = min(int(math.ceil(max_abs_ry)) + 4, int(max(h, w) * 0.6))

    strip_w = max(4, int(round(strip_right + pad_left)))
    strip_h = max(4, 2 * half)
    # Rotate about the collar, then move the collar to (pad_left, half).
    m = np.array([
        [c, -s, -(c * x0 - s * y0) + pad_left],
        [s, c, -(s * x0 + c * y0) + half],
    ], dtype=np.float32)
    strip = cv2.warpAffine(frame, m, (strip_w, strip_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    roi_mask = None
    if roi_px is not None and len(roi_px) >= 3:
        pts_s = cv2.transform(roi_px.reshape(-1, 1, 2), m).reshape(-1, 2).astype(np.int32)
        roi_mask = np.zeros((strip_h, strip_w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [pts_s], 255)

    return Strip(strip, px_per_mm, m, collar_x_px=pad_left, axis_length_px=axis_len, roi_mask=roi_mask)
