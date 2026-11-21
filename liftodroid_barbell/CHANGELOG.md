# Changelog

## 0.2.0

- **Built-in Web UI (Home Assistant Ingress):** Click **Open Web UI** (or pin to sidebar) to visually drag the sleeve
  ROI & collar-to-tip axis on a live camera frame, save the empty-bar reference, edit plates, tweak sliders, view the
  straightened sleeve strip & plate mask diagnostics, and test reads live without editing YAML or restarting.
- **Angled / 3/4 camera plate-thickness calibration:** Added `outer_face_mm` and a 1-point / 2-point thickness
  calibration helper in the Web UI so total stack thickness can identify plates behind the outer one even when the
  camera sees the outer plate face at an angle.
- **Colored electrical tape backup (or boost):** Assign tape colors (`red`, `blue`, `yellow`, `green`, `orange`,
  `purple`, `pink`, `white`) or sample the exact HSV right off the camera photo with the UI eyedropper to identify
  plates by colored rim tape around their outer circumference.

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
