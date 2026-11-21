# Changelog

## 0.1.3

- Fix: a reading without a weight (e.g. only the outer plate seen) was rejected by Home Assistant, so the weight
  sensor and its attributes never updated. The sensor now goes to "unknown" and keeps every attribute.

## 0.1.2

- Fix: the app couldn't find the MQTT broker (and Home Assistant's API) because the Supervisor token wasn't passed
  to it; it now reads the token where the container keeps it. A missing broker is retried instead of crashing.
- Configure everything from the Configuration tab: `unit`, `bar_weight`, `plates`, `sleeve_roi`, `sleeve_axis`,
  `sleeve_length_mm`, `use_shape` (they override barbell.yaml).
- `use_shape: false` for cameras that look along the bar: lettering and the plate inventory decide on their own.
- New "Save empty bar" button: unload the bar, press it, and that frame becomes the empty-sleeve reference.

## 0.1.1

- Automatically create `barbell.yaml` from the built-in example on first start so no manual file copy is needed.

## 0.1.0

- First release: fisheye undistortion, motion gate with settle timer, plate slabs along the sleeve, white-lettering
  OCR, inventory matching, optional second-sleeve symmetry check, MQTT discovery entities, calibration CLI.
