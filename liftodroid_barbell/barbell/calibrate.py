"""Calibration and offline testing, run inside the app container (or on any machine with the requirements).

  python -m barbell.calibrate grab --url rtsp://... --out frame.jpg
  python -m barbell.calibrate intrinsics --images boards/ --board 9x6 --square-mm 25 --fisheye
  python -m barbell.calibrate undistort --config barbell.yaml --frame frame.jpg --out und.jpg
  python -m barbell.calibrate grid --frame und.jpg --out grid.jpg
  python -m barbell.calibrate reference --config barbell.yaml --frame empty.jpg --out reference.png
  python -m barbell.calibrate analyze --config barbell.yaml --frame frame.jpg [--expected 225] [--ocr] --out annotated.jpg
  python -m barbell.calibrate evaluate --config barbell.yaml --dir labelled/ [--ocr]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

from .config import load
from .geometry import Undistorter
from .pipeline import Analyzer


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Can't read image {path}")
    return img


def cmd_grab(a) -> None:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(a.url, cv2.CAP_FFMPEG)
    frame = None
    for _ in range(15):  # skip the first frames: they are often grey or half-decoded
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        sys.exit("No frame from the stream")
    cv2.imwrite(a.out, frame)
    print(f"Saved {a.out} ({frame.shape[1]}x{frame.shape[0]})")


def cmd_intrinsics(a) -> None:
    cols, rows = (int(x) for x in a.board.lower().split("x"))
    objp = np.zeros((1, rows * cols, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * a.square_mm
    obj, img_pts, size = [], [], None
    for path in sorted(glob.glob(os.path.join(a.images, "*"))):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            print(f"  no board in {os.path.basename(path)}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        obj.append(objp)
        img_pts.append(corners.reshape(1, -1, 2))
    if len(obj) < 8:
        sys.exit(f"Only {len(obj)} usable photos; take 10-20 with the board all over the frame, edges included")
    if a.fisheye:
        k, d = np.zeros((3, 3)), np.zeros((4, 1))
        rms, k, d, _, _ = cv2.fisheye.calibrate(obj, img_pts, size, k, d, flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW)
        model = "fisheye"
    else:
        rms, k, d, _, _ = cv2.calibrateCamera([o.reshape(-1, 3) for o in obj], [p.reshape(-1, 2) for p in img_pts], size, None, None)
        model = "pinhole"
    print(f"# RMS reprojection error {rms:.3f} px (below ~1 is good) from {len(obj)} photos")
    print("lens:")
    print(f"  model: {model}")
    print(f"  calibrated_size: [{size[0]}, {size[1]}]")
    print("  camera_matrix:")
    for row in k:
        print(f"    - [{', '.join(f'{v:.4f}' for v in row)}]")
    print(f"  dist_coeffs: [{', '.join(f'{v:.6f}' for v in np.ravel(d))}]")
    print("  balance: 0.0")


def cmd_undistort(a) -> None:
    cfg = load(a.config)
    cv2.imwrite(a.out, Undistorter(cfg.lens)(_read(a.frame)))
    print(f"Saved {a.out}")


def cmd_grid(a) -> None:
    img = _read(a.frame)
    h, w = img.shape[:2]
    for i in range(1, 20):
        f = i / 20
        colour = (0, 255, 255) if i % 2 == 0 else (0, 140, 255)
        cv2.line(img, (int(f * w), 0), (int(f * w), h), colour, 1)
        cv2.line(img, (0, int(f * h)), (w, int(f * h)), colour, 1)
        if i % 2 == 0:
            cv2.putText(img, f"{f:.1f}", (int(f * w) + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
            cv2.putText(img, f"{f:.1f}", (3, int(f * h) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    cv2.imwrite(a.out, img)
    print(f"Saved {a.out}: read ROI corners off the grid as (x, y) fractions")


def cmd_reference(a) -> None:
    cfg = load(a.config)
    cv2.imwrite(a.out, Undistorter(cfg.lens)(_read(a.frame)))
    print(f"Saved {a.out}. Set reference_image: {a.out} in your config. Re-take it if the camera or lighting changes.")


def _analyzer(cfg, use_ocr: bool) -> Analyzer:
    reader = None
    if use_ocr:
        from .ocr import RapidOcrReader

        reader = RapidOcrReader()
    ref = cv2.imread(cfg.reference_image) if cfg.reference_image and os.path.exists(cfg.reference_image) else None
    return Analyzer(cfg, reader, ref)


def cmd_analyze(a) -> None:
    cfg = load(a.config)
    res = _analyzer(cfg, a.ocr).analyze(_read(a.frame), a.expected)
    if a.out:
        cv2.imwrite(a.out, res.annotated)
    print(json.dumps(res.reading.to_payload(), indent=2))
    if res.profile_a is not None:
        print("slabs (from collar):", [(round(r.start_mm), round(r.end_mm), round(r.height_mm)) for r in res.profile_a.runs])


def cmd_evaluate(a) -> None:
    """Files named with their true total first ("225_rack.jpg", "135-2.png"): reports accuracy and how confidence
    separates right from wrong, so you can pick auto_apply_confidence."""
    cfg = load(a.config)
    an = _analyzer(cfg, a.ocr)
    rows = []
    for path in sorted(glob.glob(os.path.join(a.dir, "*"))):
        m = re.match(r"^(\d+(?:\.\d+)?)", os.path.basename(path))
        img = cv2.imread(path)
        if not m or img is None:
            continue
        truth = float(m.group(1))
        r = an.analyze(img).reading
        rows.append((os.path.basename(path), truth, r.detected_weight, r.confidence))
        mark = "ok " if r.detected_weight == truth else "BAD"
        print(f"{mark} {os.path.basename(path):30s} truth {truth:>6g}  read {r.detected_weight!s:>6}  conf {r.confidence:.2f}  {r.status}")
    if not rows:
        sys.exit("No labelled frames")
    right = [c for _, t, d, c in rows if d == t]
    wrong = [c for _, t, d, c in rows if d != t]
    print(f"\n{len(right)}/{len(rows)} right")
    for thr in (0.6, 0.7, 0.8, 0.9, 0.95):
        auto = sum(c >= thr for c in right)
        bad = sum(c >= thr for c in wrong)
        print(f"  auto_apply_confidence {thr:.2f}: {auto} applied by themselves, {bad} WRONG applied, {len(rows) - auto - bad} asked")
    print("Pick the lowest threshold with 0 wrong applied.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="barbell.calibrate")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grab"); g.add_argument("--url", required=True); g.add_argument("--out", default="frame.jpg"); g.set_defaults(f=cmd_grab)
    i = sub.add_parser("intrinsics"); i.add_argument("--images", required=True); i.add_argument("--board", default="9x6")
    i.add_argument("--square-mm", type=float, default=25.0); i.add_argument("--fisheye", action="store_true"); i.set_defaults(f=cmd_intrinsics)
    u = sub.add_parser("undistort"); u.add_argument("--config", required=True); u.add_argument("--frame", required=True); u.add_argument("--out", default="und.jpg"); u.set_defaults(f=cmd_undistort)
    gr = sub.add_parser("grid"); gr.add_argument("--frame", required=True); gr.add_argument("--out", default="grid.jpg"); gr.set_defaults(f=cmd_grid)
    r = sub.add_parser("reference"); r.add_argument("--config", required=True); r.add_argument("--frame", required=True); r.add_argument("--out", default="/config/reference.png"); r.set_defaults(f=cmd_reference)
    an = sub.add_parser("analyze"); an.add_argument("--config", required=True); an.add_argument("--frame", required=True)
    an.add_argument("--expected", type=float); an.add_argument("--ocr", action="store_true"); an.add_argument("--out"); an.set_defaults(f=cmd_analyze)
    ev = sub.add_parser("evaluate"); ev.add_argument("--config", required=True); ev.add_argument("--dir", required=True); ev.add_argument("--ocr", action="store_true"); ev.set_defaults(f=cmd_evaluate)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
