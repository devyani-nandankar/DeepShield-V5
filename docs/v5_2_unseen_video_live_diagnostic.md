# DeepShield V5.2 — Live Pipeline Trace & Diagnostic Report for Unseen Video

**Diagnostic Date:** September 18, 2026  
**Status:** Investigation Complete — Forensic Diagnostic Only (Zero Production Code Modified)  
**Test Suite Verification:** 56 / 56 Tests Passing (`Ran 56 tests in 42.229s, OK`)  
**Investigated Case Profile:**
- **V5.2 Temporal Model Score:** `0.7496` (Threshold: `0.30`, Prediction: `Fake`)
- **Phase 1 Calibrated Score:** `0.0758` (Threshold: `0.21`, Prediction: `Real`)
- **Final Multimodal Score:** `0.0758`
- **Video Reliability Weight:** `0.05`
- **Final Website Verdict:** `Uncertain`

---

## 1. Executive Summary & Root Cause Overview

This forensic diagnostic investigated why uploaded unseen videos processed via the web frontend at `http://127.0.0.1:5000` resulted in `Final Verdict: UNCERTAIN` with `fused_score = 0.0758` and `reliability = 0.05`, despite the DeepShield V5.2 Dedicated Temporal Video Model outputting strong manipulation evidence (`score = 0.7496`).

### The Three Smoking-Gun Root Causes

1. **Root Cause 1 — Stale Background Flask Runtime Process (`task-5027`):**
   - The Flask development server process (`PID 19164 / 19184`) was launched at `07:49:01` (`task-5027.log`) with `Debug mode: off`.
   - The Phase 2 fusion arbitration update (giving V5.2 temporal authority over Phase 1) was committed to `backend/fusion.py` and `backend/video_detector.py` hours later (around `12:48`).
   - Because Flask was running without auto-reload in a daemon background task, Python cached the pre-correction module bytecode in memory.
   - When the user uploaded the video between `12:54` and `12:58`, the server executed the legacy pre-correction logic where **ANY** disagreement (`p1_pred != v5_2_pred`: `Real != Fake`) unconditionally forced `conflict_detected = True`, `verdict = "Uncertain"`, and `overall_prediction = "Uncertain"`.

2. **Root Cause 2 — In the Pre-Correction Runtime, `fused_score` Completely Ignored V5.2:**
   - In the pre-correction `backend/fusion.py`, `vid["representative_score"]` was mapped exclusively to Phase 1's `video_calibrated_score` (`0.0758`).
   - The V5.2 score was not blended into `representative_score`.
   - When the user uploaded a video without an image, `MultimodalFusionEngine.evaluate_fusion` calculated:
     $$\text{fused\_score} = \text{representative\_score} = 0.0758$$
   - This exact value (`0.0758`) was assigned to `fusion_score` and displayed on the frontend banner.

3. **Root Cause 3 — Compounding Multiplicative Penalty Resulted in Reliability = 0.05:**
   - In `backend/fusion.py` (lines 348–366), video reliability starts at `0.90` and undergoes compounding multiplicative penalties:
     - Base reliability: `0.90`
     - Low bitrate / compression degradation penalty: $\times 0.40 \implies 0.36$
     - Volatility / standard deviation penalty (`temporal_volatility >= 0.20` or `score_std >= 0.25`): $\times 0.50 \implies 0.18$
     - Uncertainty penalty (`verdict == "Uncertain"`): $\times 0.30 \implies 0.054$
     - Resulting displayed reliability: $\text{round}(0.054, 2) = \mathbf{0.05}$!

---

## 2. Step 1 — Trace of the Exact Uploaded Video

By inspecting the server access logs (`task-5027.log`), three consecutive POST requests to `/api/predict/video` were recorded at `12:54:02`, `12:54:40`, and `12:58:27`. Keyframe thumbnail generation timestamps and frame indices pinpointed the uploaded files:
- **12:54:02:** Uploaded video with sampled frame indices `f0, f50, f110, f120, f130` (matched `unseen_videos/fake/WhatsApp Video 2026-09-15 at 1.31.13 PM.mp4`, 142 total frames).
- **12:54:40:** Uploaded video with sampled frame indices `f196, f476, f868, f924, f1008` (matched `unseen_videos/fake/WhatsApp Video 2026-09-15 at 1.33.18 PM.mp4`, 1102 total frames, frame step 28).
- **12:58:27:** Uploaded video with sampled frame indices `f22, f44, f506, f572, f616` (matched 899 total frames, frame step 22, generating Grad-CAM overlay `gradcam_overlay_7271d7260228.jpg`).

### Video Characteristics and Variable Trace Table

| Pipeline Variable | Value | Explanation / Mathematical Source |
| :--- | :--- | :--- |
| **Total Frames** | 899–1102 frames | Video stream property |
| **Analyzed / Sampled Frames** | 40 frames | Bounded by `target_max_frames = 40` |
| **Valid Face Frames (YuNet)** | 16–40 frames | Detected faces with 1:1 square crop geometry |
| **Number of T=16 Sequences** | 1–3 sequences | Staged for temporal BiLSTM Model A |
| **Phase 1 Raw Median Score** | $\approx 0.7148$ | Raw EfficientNetB2 frame classification scores |
| **Phase 1 Calibrated Score** | $\mathbf{0.0758}$ | Inverted WhatsApp calibration: $0.21 \times \left(1 - \frac{0.7148 - 0.21}{0.79}\right) = 0.0758$ |
| **Phase 1 Prediction** | **Real** | Because calibrated score $0.0758 < 0.21$ |
| **V5.2 Temporal Score** | $\mathbf{0.7496}$ | Raw sigmoid output from BiLSTM Model A |
| **V5.2 Calibrated Threshold** | **0.30** | Stored in `backend/config.py` (`VIDEO_TEMPORAL_THRESHOLD`) |
| **V5.2 Prediction** | **Fake** | Because temporal score $0.7496 \ge 0.30$ |
| **Live Server Verdict** | **Uncertain** | Hard conflict in stale server process |
| **Multimodal Fused Score** | **0.0758** | Fallback to Phase 1 `video_calibrated_score` |
| **Video Reliability Weight** | **0.05** | $0.90 \times 0.40 \times 0.50 \times 0.30 = 0.054 \to 0.05$ |

---

## 3. Step 2 — Exactly Where V5.2 Was Lost in Execution

Tracing the call stack from HTTP request to JSON response:

```
[Browser Upload]
       │
       ▼
1. Flask Endpoint: backend/app.py:predict_video()
       │
       ▼
2. Video Detector: backend/video_detector.py:predict()
       ├── Phase 1 Evaluation (EfficientNetB2 on face frames)
       │     └─ raw median: 0.7148 -> video_calibrated_score: 0.0758 -> p1_pred: Real
       │
       ├── V5.2 Temporal Evaluation (BiLSTM Model A)
       │     └─ temporal_score: 0.7496 -> v5_2_pred: Fake
       │
       └── [Inter-Pipeline Consensus Check]
             ├─ In Stale Memory (task-5027):
             │    if p1_pred != v5_2_pred:
             │        conflict_detected = True
             │        verdict = "Uncertain"  <-- [LOST HERE IN VIDEO_DETECTOR]
             │
             └─ In Disk Code:
                  elif is_v5_2_high_conf:
                      verdict = "Likely Fake" (arbitrates correctly)
       │
       ▼
3. Fusion Engine: backend/app.py -> evaluate_fusion(video_prediction=result)
       ├── parse_video_evidence(video_data)
       │     ├─ In Stale Memory:
       │     │    representative_score = video_calibrated_score = 0.0758 <-- [LOST IN FUSION]
       │     │
       │     └─ reliability calculation:
       │          reliability = 0.90 * 0.40 * 0.50 * 0.30 = 0.054 -> 0.05
       │
       └── fuse_multimodal_evidence(...)
             ├─ fused_score = representative_score = 0.0758
             └─ Condition 1 (if conflict_detected):
                  overall_prediction = "Uncertain"  <-- [LOST IN FUSION VERDICT]
       │
       ▼
4. Report Generator: backend/report_generator.py:generate_report()
       ├── Line 53-54:
       │     if prediction_result.get("verdict") == "Uncertain" or fusion.get("overall_prediction") == "Uncertain":
       │         verdict = "Uncertain"  <-- [LOCKED IN REPORT GENERATOR]
       │
       ▼
5. Frontend UI: frontend/index.html:renderResults()
       ├── verdictTag: "INCONCLUSIVE / UNCERTAIN"
       ├── verdictTitle: "Uncertain"
       ├── metricScore: "0.0758"
       └── metricConfidence: "50.0%"
```

---

## 4. Step 3 — Exact Arithmetic Multipliers for Video Reliability = 0.05

In `backend/fusion.py` (lines 348–366), video reliability is computed dynamically via sequential degradation factors:

$$\text{reliability} = 0.90 \quad (\text{nominal high-quality baseline})$$

1. **Compression Degradation Multiplier ($\times 0.40$):**
   - The uploaded video exhibited compression artifacts, texture blur, or low bitrate encoding standard for WhatsApp videos (`compression_degradation = True`).
   $$\text{reliability} = 0.90 \times 0.40 = 0.36$$

2. **Temporal Volatility Multiplier ($\times 0.50$):**
   - Frame score fluctuations across keyframes yielded `temporal_volatility >= 0.20` or `score_std >= 0.25`:
   $$\text{reliability} = 0.36 \times 0.50 = 0.18$$

3. **Uncertainty Multiplier ($\times 0.30$):**
   - Because the legacy conflict logic forced `verdict = "Uncertain"`, line 361 applied the uncertainty dampener:
   $$\text{reliability} = 0.18 \times 0.30 = 0.054$$

4. **Floating-Point Rounding:**
   $$\text{round}(0.054, 2) = \mathbf{0.05}$$

This explains the exact figure `Video reliability = 0.05`.

---

## 5. Step 4 — Evaluation of `is_v5_2_high_conf`

In the updated code on disk (`backend/video_detector.py` lines 646–650 and `backend/fusion.py` lines 268–274), the high-confidence arbitration criteria are defined as:

```python
is_v5_2_high_conf = (
    v5_2_face_frames >= 16 and
    v5_2_score is not None and
    (v5_2_score <= 0.22 or v5_2_score >= 0.40)
)
```

Evaluating these conditions against the uploaded video's parameters:
- **Condition A (`v5_2_face_frames >= 16`):**
  - For the uploaded video, `v5_2_face_frames = 16` (or 40 sampled face frames). $\implies$ **TRUE**.
- **Condition B (`v5_2_score is not None`):**
  - $0.7496$ is present. $\implies$ **TRUE**.
- **Condition C (`v5_2_score <= 0.22 or v5_2_score >= 0.40`):**
  - $0.7496 \ge 0.40$. $\implies$ **TRUE** (distance from threshold $0.30$ is $+0.4496$).

**Result:** `is_v5_2_high_conf` evaluates to **`TRUE`**.

### Why did it not arbitrate in the live test?
Because the running Flask server in `task-5027` was started at `07:49`, before `is_v5_2_high_conf` was written. In that process memory, the check was simply:
```python
if p1_pred != v5_2_pred:
    conflict_detected = True
    verdict = "Uncertain"
```
Which evaluated `Real != Fake` $\implies$ `True`, immediately overriding V5.2.

---

## 6. Step 5 — Audit of Remaining Backend Conflict Checks

A systematic audit across all backend files was performed to identify any secondary checks that could force `verdict = "Uncertain"`:

### 1. `backend/report_generator.py` (Lines 53–54)
```python
verdict = fusion.get("overall_prediction", fusion.get("prediction", prediction_result.get("prediction", "Real")))
if prediction_result.get("prediction") == "Uncertain" or prediction_result.get("verdict") == "Uncertain" or fusion.get("overall_prediction") == "Uncertain":
    verdict = "Uncertain"
```
- **Finding:** If the underlying `prediction_result` (from `video_detector.predict`) returned `"Uncertain"`, `report_generator` enforces `"Uncertain"` at the report level, even if `fusion` had resolved it.
- **Status:** Once the server runs the current code where `video_detector` correctly sets `verdict = "Likely Fake"`, this condition will not trigger.

### 2. `backend/fusion.py` Condition 2 (Line 720)
```python
elif (not img["available"]) and vid["available"] and (vid["verdict"] == "Uncertain"):
    overall_pred = "Uncertain"
```
- **Finding:** For video-only uploads, if `vid["verdict"] == "Uncertain"`, fusion defaults to `"Uncertain"`.

### 3. `backend/video_detector.py` Boundary Margin Checks (Lines 464–479)
```python
elif 0.18 <= median_score <= 0.24:
    verdict = "Uncertain"
elif temporal_volatility >= 0.40 and score_std >= 0.35 and (0.15 <= median_score <= 0.35):
    verdict = "Uncertain"
```
- **Finding:** For raw median $0.7148$, this boundary check is bypassed ($0.7148 > 0.24$).

---

## 7. Step 6 — Why `fused_score = 0.0758`

In `backend/fusion.py` (pre-correction runtime):
1. In `parse_video_evidence`, `representative_score` was assigned:
   $$\text{representative\_score} = \text{video\_calibrated\_score} = 0.0758$$
2. In `fuse_multimodal_evidence`:
   - `weights = [vid['reliability_weight']] = [0.05]`
   - `scores = [vid['representative_score']] = [0.0758]`
   - Multimodal formula:
   $$\text{fused\_score} = \frac{0.05 \times 0.0758}{0.05} = \mathbf{0.0758}$$

On the updated codebase on disk:
- `calibrated_temporal_score = calibrate_temporal_score(0.7496, 0.30) = 0.7174`
- When V5.2 arbitrates:
  $$\text{representative\_score} = \text{calibrated\_temporal\_score} = 0.7174$$
  $$\text{fused\_score} = 0.7174 \ge 0.21 \implies \text{Likely Fake (Confidence: 0.85, Risk: Critical)}$$

---

## 8. Verification: Current Codebase vs Live Server Runtime

Running the exact input profile through the current codebase on disk (`scratch/trace_eval.py`):

```text
--- Current backend/fusion.py on Disk ---
Overall Prediction: Likely Fake
Fusion Score: 0.7174
Conflict Detected: False
Video Evidence Rep Score: 0.7174
Video Reliability Weight: 0.18
Video Verdict: Likely Fake
Conflict Factors: ['Secondary spatial heuristic (Real, score: 0.0758) disagreed with primary V5.2 temporal model (Fake, score: 0.7496).']
Corroborating Factors: ['V5.2 temporal evidence was given higher reliability than the legacy Phase 1 spatial heuristic because sufficient face frames were available and the temporal score was outside the uncertainty boundary.']

--- Current backend/report_generator.py on Disk ---
Report Verdict: Likely Fake
Report Confidence: 0.85
Report Risk Level: Critical
Report Multimodal Fusion Score: 0.7174
Report Video Reliability Weight: 0.18
```

And executing the complete unit test suite:
```text
Ran 56 tests in 42.229s
OK
```

---

## 9. Conclusion & Required Operational Step

The diagnostic confirms:
1. **The current codebase on disk already correctly arbitrates the unseen video to `Likely Fake` with `fusion_score = 0.7174` and `confidence = 0.85`.**
2. The discrepancy observed on the website occurred because the background Flask server process (`task-5027`) has been running continuously since `07:49` and never loaded the updated modules.
3. Restarting the Flask server process will immediately activate the V5.2 temporal authority on live uploads without modifying any model weights, thresholds, or code.
