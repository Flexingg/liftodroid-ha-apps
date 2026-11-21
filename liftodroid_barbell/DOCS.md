# LiftoDroid Barbell Camera

Reads the weight on your barbell from an overhead camera, so LiftoDroid logs what you actually lifted. Nothing
runs on your phone but a lightweight status check; the vision work happens here, and only while a workout is on.

## How it works

1. **Idle by default.** The camera stream is closed. When LiftoDroid starts a barbell set it *arms* this app
   (MQTT `liftodroid/barbell/cmd/arm`) with the set, the planned weight and your bar and plates from Liftosaur.
   Arming expires by itself (20 minutes, renewed each set), so a phone that dies never leaves the camera running.
2. **Motion gate and settle timer.** Frames are sampled at `analysis_fps` (2 by default) and undistorted. The
   sleeve region is watched for change; once it has been still for `settle_s` (1.5 s) after movement, so you've
   stepped away and there's no blur, the sleeve is read once. If `person_entity` is "on" (you're in the rack) it
   waits until it turns off. "Read now" (button, or the phone) forces a read.
3. **Four cues, fused.**
   * **Plate slabs & thickness.** From above or a 3/4 angle, each plate adds its thickness along the sleeve (and its
     diameter across it). The sleeve is straightened into a strip and compared against the empty-bar reference to
     measure the stack from the collar outwards — identifying plates hidden behind the outer plate. For angled 3/4
     cameras, a 2-point calibration in the Web UI solves both perspective scale and outer-face overhang (`outer_face_mm`).
   * **Colored rim electrical tape (optional backup or boost).** Wrap a strip of colored electrical tape around the
     outer circumference of each plate denomination (e.g. red = 45, blue = 25, green = 10) and pick the colors right
     off the live camera photo with the Web UI eyedropper. Every plate's colored rim stays visible along the sleeve
     even behind the outer plate.
   * **Lettering.** White markings ("45LB", "REP FITNESS") are lifted with CLAHE and a top-hat filter, glare
     blobs are painted over, and an on-device OCR model (PP-OCR on ONNX Runtime) reads them in all four
     orientations. Only numbers that are plate weights you own count; a kg number in an lb gym is dropped.
   * **Inventory.** Every per-side loading your plates can make (pairs only) is scored against thickness, tape colors,
     and lettering. The planned weight nudges ties, never overrides clear camera evidence.
   Probabilities are summed per total (45 a side and 35+10 a side both make 135), and the best total wins.
4. **Publish.** `sensor.barbell_camera_weight` (state = total; attributes below), an annotated
   `image.barbell_camera_last_reading`, sleeve motion, and a confidence sensor. The phone decides what to do.

## Install & Setup (Web UI)

**From GitHub (recommended).** Settings → Apps → App store → ⋮ → **Repositories**, add the LiftoDroid app
repository URL (the public mirror, e.g. `https://github.com/Flexingg/liftodroid-ha-apps`; see the repository's
`homeassistant/README.md`), then install **LiftoDroid Barbell Camera** from the store. Home Assistant builds it
on your machine the first time (a few minutes). New versions arrive as normal app updates.

Then:

1. It needs the Mosquitto broker app (already running if you use VisionState).
2. In the app's **Configuration** tab: set `camera_url` (RTSP/RTSPS stream) or `camera_entity`, and optionally
   `person_entity: binary_sensor.in_squat_rack_person`.
3. Start the app and click **Open Web UI** (or turn on **Show in sidebar**):
   * Click **Grab Live Frame**, then drag the yellow **ROI** box around the sleeve and drag the green **Collar (C) →
     Tip (T)** arrow along the sleeve.
   * Unload the bar and click **Save Empty Bar** (gives clean background subtraction for thickness and ignores
     colored gym gear behind the bar).
   * In **Plates & Tape Colors**, enter your plates' diameters & thicknesses (mm). Optionally assign electrical tape
     colors (or click **Pick** to sample the tape right off the live camera frame).
   * In **Detection & Thickness**, use **1-pt Calibrate** (overhead camera) or **Record 2-pt Sample → Solve 2-pt**
     (angled / 3/4 camera, e.g. 135 lb then 225 lb) to lock in exact thickness scaling.
   * In **Test & Read**, click **Grab & Test Now**, inspect the **Sleeve Strip Diagnostics** (rectified strip + green
     plate mask + tape color bar), adjust any sliders live, then click **Save & Apply** (no restart required).

## CLI Calibration (optional, for fisheye lens intrinsics & batch evaluation)

For severe fisheye distortion or batch-evaluating a folder of saved frames, open a terminal in the app from the SSH
app (protection mode off):
`docker exec -it $(docker ps -qf name=liftodroid_barbell) bash`, then `cd /app` and use
`/opt/venv/bin/python -m barbell.calibrate …`. Work in `/config` so files survive updates.

1. **Grab a frame**: `grab --url "$URL" --out /config/frame.jpg`.
2. **Lens (fisheye).** Print a 9×6 checkerboard (25 mm squares), tape it to something flat, and grab 15–20 frames
   with it all over the image: centre, every edge, every corner, tilted. Then
   `intrinsics --images /config/boards --board 9x6 --square-mm 25 --fisheye`
   and paste the `lens:` block into `barbell.yaml`. An RMS under ~1 px is good. If straight rack uprights already
   look straight on screen, `model: none` is fine.
3. **Regions.** `undistort --config /config/barbell.yaml --frame /config/frame.jpg --out /config/und.jpg`, then
   `grid --frame /config/und.jpg --out /config/grid.jpg`, open it, and read off:
   * `sleeve.axis`: the collar (inner end, where the first plate sits) → the sleeve tip;
   * `sleeve.roi`: a box around the sleeve and the biggest plate on it, with margin;
   * optionally `sleeve_b` (the other sleeve, for uneven-loading warnings), `storage_roi`, `bench_roi`.
   Set `sleeve_length_mm` to your bar's real collar-to-tip length (scale comes from it).
4. **Plates.** Measure each plate's diameter and hub thickness in mm. Bumper plates are all 450 mm across, so for
   bumpers thickness and lettering do the work.
5. **Empty reference.** Rack the bar with no plates and good light, grab a frame, then
   `reference --config /config/barbell.yaml --frame /config/empty.jpg --out /config/reference.png`.
   Re-take it if the camera moves or the lighting changes a lot.
6. **Check.** Load a known weight, grab, and
   `analyze --config /config/barbell.yaml --frame /config/f.jpg --ocr --out /config/annotated.jpg`.
   The annotated image shows the regions, the slabs found (alternating colours) and the result.

## Tune with recorded frames

Save frames named by their true total (`225_a.jpg`, `135-warmup.jpg`, `45_empty.jpg` …), as many loadings and
lighting conditions as you can, including a hand in frame and glare. Then
`evaluate --config /config/barbell.yaml --dir /config/labelled --ocr`. It prints every read and, for each candidate
`auto_apply_confidence`, how many readings would have been applied by themselves and how many of those were
wrong. Choose the lowest threshold with **zero wrong**, and set the same value in LiftoDroid.

| If… | Change |
|---|---|
| Reads while you're still loading | raise `settle_s`, or set `person_entity` |
| Never settles (fan, flicker) | raise `motion_pixel_delta` / `motion_fraction` |
| Slabs missing or merged | re-take the reference; adjust `plate_pixel_delta` |
| Right plates, wrong count | check `sleeve_length_mm` and plate thicknesses |
| Too unsure on clear frames | raise `shape_weight` a little |
| Confident but wrong | lower `shape_weight`; check the lens calibration near the frame edge |

## What Home Assistant gets

`sensor.barbell_camera_weight` attributes (also the retained JSON on `liftodroid/barbell/state`):

| Attribute | Meaning |
|---|---|
| `detected_weight` | total on the bar, bar included (null when unreadable) |
| `unit` | lb or kg |
| `confidence` | 0–1, probability of that total |
| `per_side_breakdown` | plates on the sleeve, inner to outer |
| `per_side_b`, `symmetric` | the other sleeve, if configured, and whether both match |
| `raw_ocr_matches` | every denomination read: weight, score, text, sleeve, position along the sleeve |
| `needs_confirmation` | true when the phone should ask instead of applying |
| `status` | `ok`, `low_confidence`, `asymmetric`, `no_evidence` |
| `reasons`, `alternatives` | why it's unsure; the next best totals |
| `frame_timestamp`, `last_motion_ts` | when the frame was taken; the last movement before it |
| `context` | the set it was read for (`set_id`, `exercise`, `expected`) |

Commands (MQTT, base `liftodroid/barbell`): `cmd/arm` (JSON), `cmd/disarm`, `cmd/detect`, `cmd/enabled`
(`ON`/`OFF`).

## Privacy

Frames never leave Home Assistant. Only the annotated frame of the last reading is kept (as an image entity).
