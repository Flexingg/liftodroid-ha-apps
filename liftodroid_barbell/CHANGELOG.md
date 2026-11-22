# Changelog

## 0.2.2

- **Full-ROI rectified strip (no more clipped rim colors behind Collar):** `sleeve_strip()` now extends backward
  behind the Collar (`C`) marker to cover your entire yellow ROI box and masks both plate thickness and rim colors
  strictly to the ROI — so overhead / 3/4 views where `C` sits on the visible sleeve no longer chop off hidden
  plates or colored rim tape behind `C`.
- **Glare-resilient dual-half rim color sampling:** Reads the cleaner un-glared side of the plate rim when ceiling
  lights wash out one half of the plate, and fuses tape colors together with plate thickness.
- **Fixed 2-point thickness calibration:** **1-pt Calibrate** and **Record 2-pt Sample** now automatically grab a
  fresh live frame, refresh the diagnostic strip immediately, scale against the collar-to-tip axis even with
  backward ROI padding, and show exact pixel/mm diagnostics if a solve fails.
- **Rack-post gap bridging:** Bridges up to 24 mm horizontal gaps inside a plate (e.g. a black rack upright behind
  a black bumper plate) and selects the primary contiguous plate stack.

## 0.2.1

- **Lighting-invariant tape colors:** Switched rim tape detection to per-frame auto-white-balanced CIE L\*a\*b\*
  chromaticity sampled in the central sleeve band so colors stay accurate across lights on/off, auto-exposure shifts,
  and overhead glare, and prevented glare/chrome/white lettering from false-triggering `white` tape.
- **Exposure-normalized plate mask:** Automatically normalizes empty-bar reference exposure to the live frame before
  subtracting background so brightness shifts no longer turn the wall green.
- **Collar offset & gap bridging:** Plate stack detection now finds the plate start up to ~130 mm past the collar
  marker (fixing `Measured 0 mm` when `C` is placed on the J-cup or the bar shifts sideways in the rack) and bridges
  interior plate gaps from lettering or black-on-black overlap.

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
