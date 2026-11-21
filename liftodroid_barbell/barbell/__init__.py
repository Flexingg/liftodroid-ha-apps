"""Hands-free barbell weight reading from an overhead (fisheye) gym camera.

Layers, each usable on its own:
  config     -> typed settings (lens, ROIs, plate inventory, thresholds)
  inventory  -> which per-side loadings the plates you own can make
  geometry   -> fisheye undistortion and the rectified sleeve strip
  motion     -> motion gate + settle timer on the sleeve ROI
  slabs      -> plate slabs along the sleeve (thickness / diameter)
  ocr        -> white-lettering text reading and denomination parsing
  fusion     -> combines every cue into a ranked, scored reading
  pipeline   -> frame in, Reading out (plus an annotated debug image)
  service    -> the always-on worker: RTSP, MQTT discovery, commands
"""

__version__ = "0.1.2"
