# Issues Found in Image Quality & Compliance Engine

## Issue 1: Cap Detection Runs TWICE with Different Models (Redundancy)

**Files involved:** `detectors/cap_detector.py` + `detectors/classification_pipeline.py`

| Source | Model | Function |
|--------|-------|----------|
| Step 8 in pipeline | `models/best.pt` | `detect_cap()` in `cap_detector.py` |
| Step 12 in pipeline | `models/cap_detection_model.pt` | `detect_cap_classifier()` in `classification_pipeline.py` |

**Impact:** Both run on every API call. They return slightly different output schemas (`color` vs `hsv_color`). Results may differ. Wastes ~50-100ms and memory loading two different models.

---

## Issue 2: Tshirt/Uniform Detection Runs TWICE with Different Approaches (Redundancy)

**Files involved:** `detectors/uniform_detector.py` + `detectors/classification_pipeline.py`

| Source | Method | Function |
|--------|--------|----------|
| Step 7 in pipeline | MediaPipe Pose + K-Means clustering | `detect_uniform()` in `uniform_detector.py` |
| Step 12 in pipeline | YOLO `tshirt_detection_model.pt` + HSV | `detect_tshirt()` in `classification_pipeline.py` |

**Impact:** Two completely different approaches giving potentially conflicting results. One says "Red" the other says "Blue" — which to trust? Wastes memory with both models loaded.

---

## Issue 3: HSV Color Analysis Code Duplicated 3 Times

**Files involved:** `detectors/cap_detector.py`, `detectors/classification_pipeline.py`, `test_cap_with_color.py`

The `_COLOR_RANGES` dictionary (11 color ranges) and `_detect_dominant_color()` function are copy-pasted identically across 3 files with slightly different names:
- `cap_detector.py` → `_detect_dominant_color()`
- `classification_pipeline.py` → `_detect_dominant_color_hsv()`
- `test_cap_with_color.py` → `detect_dominant_color()`

**Impact:** Bug fix in one file won't propagate. Maintainability nightmare. ~80 lines duplicated × 3 = ~160 wasted lines.

---

## Issue 4: Response Has Duplicate Data Sections

**File involved:** `workers/pipeline.py`

The API response contains BOTH:
- `checks.uniform` and `checks.cap` (from Steps 7/8 — legacy detectors)
- `classification_pipeline.tshirt` and `classification_pipeline.cap` (from Step 12)

PRD Section 11 only specifies `checks.uniform` and `checks.cap`. Having a separate `classification_pipeline` block is:
- Confusing for API consumers
- Wastes response size (more bandwidth)
- Makes scoring ambiguous

---

## Issue 5: Pipeline Order Doesn't Match Desired Flow

**Current (in pipeline.py):**
1. Face Detection
2. Person Detection
3. Face Visibility
4. Face Distance
5. Face Orientation
6. Blur Detection
7. Lighting Detection
8. Uniform Detection (MediaPipe)
9. Cap Detection (`best.pt`)
10. Accessory Detection
11. Screenshot Risk
12. Compliance Score
13. Classification Pipeline (Tshirt → Cap)

**Desired Flow:**
1. Person Detection → Check exactly ONE person
2. Cap Detection (`cap_detection_model.pt`) → Crop cap → HSV color analysis
3. Tshirt Detection (`tshirt_detection_model.pt`) → Crop tshirt → HSV color analysis
4. If tshirt color is Red → PASS
5. Other checks (blur, lighting, orientation, etc.)

---

## Issue 6: Red=Pass Special Logic Missing

There is no special rule that says "if tshirt is Red, auto-PASS regardless of other scores". All scoring is purely mathematical weighted average.

---

## Issue 7: Render Free Tier Memory Constraint (300 MB RAM)

**Models currently loaded simultaneously:**
1. `blaze_face_short_range.tflite` (MediaPipe FaceDetector) — ~500 KB
2. `face_landmarker.task` (MediaPipe FaceLandmarker) — ~8 MB
3. `pose_landmarker_lite.task` (MediaPipe Pose) — ~8 MB
4. `best.pt` (YOLO cap model in `cap_detector.py`) — ~6 MB
5. `tshirt_detection_model.pt` (YOLO tshirt in `classification_pipeline.py`) — ~6 MB
6. `cap_detection_model.pt` (YOLO cap in `classification_pipeline.py`) — ~6 MB
7. ONNX models (if used)

Total model memory: ~35 MB
Python runtime + OpenCV + NP + dependencies: ~150-200 MB
Processing memory: ~50-100 MB

**Risk:** Currently close to limit. Loading both `best.pt` AND `cap_detection_model.pt` is unnecessary waste. Need to pick ONE cap detection model.

---

## Issue 8: Test Files Duplication

| Test File | What it tests |
|-----------|--------------|
| `test_best_cap_model.py` | Tests `best.pt` only (cap) |
| `test_cap_with_color.py` | Tests `best.pt` + HSV (duplicates production logic) |
| `test_classification_pipeline.py` | Tests `tshirt_detection_model.pt` + `cap_detection_model.pt` |

The first two are redundant if we consolidate to use `tshirt_detection_model.pt` + `cap_detection_model.pt`.
</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>
