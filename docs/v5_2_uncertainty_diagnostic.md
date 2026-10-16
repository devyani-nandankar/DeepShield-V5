# DeepShield V5.2 — Video Uncertainty Diagnostic Report

**Diagnostic Date:** September 18, 2026  
**Status:** Investigation Complete — Diagnostic Only (Zero Production Code Modified)  
**Test Suite Verification:** 46 / 46 Tests Passing (`backend/tests/`)  

---

## 1. Executive Summary

Following the integration of the **DeepShield V5.2 Dedicated Temporal Video Model (BiLSTM Model A)** into the video verification pipeline, uploaded videos frequently reach `overall_prediction = "Uncertain"`, even when the multimodal fused score or underlying models report high confidence far from decision thresholds.

This comprehensive diagnostic investigated `backend/video_detector.py`, `backend/fusion.py`, `backend/config.py`, `backend/app.py`, `backend/video_temporal_detector.py`, and `frontend/index.html`. In addition, empirical diagnostic evaluations were run across all **20 WhatsApp calibration benchmark videos** (10 Real, 10 Fake) as well as external/non-WhatsApp test videos (`test.mp4`, `real_check.mp4`, `fake_check.mp4`, `test1.mp4`).

### Key Diagnostic Findings
1. **Root Cause: Naive Hard Conflict Triggering between Inverted Phase 1 Heuristics and Standard V5.2 Temporal Predictions:**
   - In Phase 1, an inverted empirical calibration was implemented for WhatsApp compressed videos (`raw_median < 0.21 -> Fake`, `raw_median >= 0.21 -> Real`).
   - In V5.2, the temporal BiLSTM model was trained with standard binary cross-entropy where higher probability indicates manipulation (`temporal_score >= 0.30 -> Fake`, `temporal_score < 0.30 -> Real`).
   - In both `backend/video_detector.py` (lines 643–646) and `backend/fusion.py` (lines 482–484 & 599–606), **ANY disagreement** between `p1_pred` and `v5_2_pred` immediately forces `conflict_detected = True`, which short-circuits the pipeline and forces `verdict = "Uncertain"` and `overall_prediction = "Uncertain"`.
2. **The "Fix Becomes Uncertainty" Paradox:**
   - In Phase 1, three specific videos were misclassified:
     - `WhatsApp Video 9.23.21 AM.mp4` (Real, Phase 1 predicted Fake)
     - `WhatsApp Video 9.27.24 AM.mp4` (Real, Phase 1 predicted Fake)
     - `WhatsApp Video 9.30.53 AM (1).mp4` (Fake, Phase 1 predicted Real)
   - **The V5.2 Temporal Model was 100% correct on all three of these errors!**
     - On 9.23.21 AM: V5.2 predicted **Real** (score: 0.1345 vs threshold 0.30).
     - On 9.27.24 AM: V5.2 predicted **Real** (score: 0.1813 vs threshold 0.30).
     - On 9.30.53 AM (1): V5.2 predicted **Fake** (score: 0.5683 vs threshold 0.30).
   - However, because Phase 1 was wrong and V5.2 was right, they disagreed (`p1_pred != v5_2_pred`). Rather than allowing the superior temporal model to correct Phase 1, the hard conflict check penalized the system by converting all three corrections into `"Uncertain"`!
3. **Generalization Failure on In-the-Wild Videos (`test.mp4`):**
   - On normal, in-the-wild deepfakes with high raw EfficientNet scores (e.g. `raw_median = 0.93`), Phase 1's inverted formula maps high raw scores to Real (`video_calibrated_score = 0.0194`).
   - V5.2 correctly flags manipulation (`temporal_score = 0.4602`).
   - Disagreement triggers an immediate conflict. Fused score is 0.0194 (far from 0.21), but the verdict is forced to `"Uncertain"`.
4. **Disconnection in Multimodal Fusion:**
   - In `backend/fusion.py` (lines 568–585), `fused_score` is computed strictly using `vid["representative_score"]` (which comes exclusively from Phase 1's `video_calibrated_score`) and metadata. The V5.2 temporal score is **never blended into `fused_score`**!
   - Condition 1 (`if conflict_detected: overall_pred = "Uncertain"`) executes *before* checking `fused_score`, rendering the distance from threshold completely irrelevant.

---

## 2. Systematic Answers to the 7 Diagnostic Questions

### Q1: Is Phase 1 overriding V5.2?
**Yes, in the form of a mutual veto.**
If Phase 1 predicts `Fake` and V5.2 predicts `Real`, or vice-versa, Phase 1's prediction creates a hard conflict that strips V5.2 of its authority and forces `"Uncertain"`. Furthermore, in `fusion.py`, `fused_score` is computed 100% from Phase 1's calibrated score, completely ignoring V5.2's score.

### Q2: Is V5.2 overriding Phase 1?
**Only when Phase 1 is already uncertain.**
In `backend/video_detector.py` (lines 664–670), if Phase 1 predicts `"Uncertain"` (e.g. raw median in `[0.18, 0.24]`), V5.2 provides the primary verdict. But if Phase 1 makes any binary prediction (`Real` or `Fake`), V5.2 cannot override it; it can only trigger a mutual veto to `"Uncertain"`.

### Q3: Is conflict detection triggering too aggressively?
**Yes, critically.**
The conflict detection rule in both `video_detector.py` (line 643) and `fusion.py` (line 483) is:
```python
if p1_pred in ["Real", "Fake"] and v5_2_pred in ["Real", "Fake"]:
    if p1_pred != v5_2_pred:
        conflict_detected = True
        verdict = "Uncertain"
```
This naive equality check does not account for model confidence, temporal authority, or the known failure modes of Phase 1's heuristic. Any disagreement—even when V5.2 has high confidence (e.g., temporal score 0.5683 vs threshold 0.30) and Phase 1 made a known spatial error—triggers a total system retreat into `"Uncertain"`.

### Q4: Is V5.2 temporal score being blended with image threshold wrongly?
**No, but it is not being blended at all.**
`VIDEO_TEMPORAL_THRESHOLD = 0.30` is correctly respected inside `video_temporal_detector.py`. However, in `backend/fusion.py`, the V5.2 temporal score is completely omitted from the weighted scoring loop (`scores.append(vid["representative_score"])`). The `vid["representative_score"]` is assigned directly to Phase 1's `video_calibrated_score`. Consequently, the multimodal `fused_score` represents only Phase 1 and metadata; it does not represent V5.2.

### Q5: Is the decision boundary margin too wide?
**Partially.**
In `backend/video_temporal_detector.py` (lines 44 & 247):
```python
VIDEO_TEMPORAL_MARGIN = 0.05
# creates boundary window [0.25, 0.35] around 0.30
```
This caused 3 WhatsApp videos (`9.23.09 AM` with score 0.3234, `9.23.11 AM` with score 0.2513, and `9.23.12 AM` with score 0.2886) to be labeled by V5.2 as `"Uncertain"`. Fortunately, because Phase 1 was `"Real"`, these 3 did not trigger a conflict and resolved to `"Real"`. However, a 0.10 total window (`[0.25, 0.35]`) on a model where test distributions range tightly from 0.20 to 0.40 is relatively wide.

### Q6: Is face detection skipping frames and causing video fallback?
**No.**
YuNet face detection achieved a 93.39% detection rate across the benchmark videos. In all 20 videos tested, valid face frames ranged from 12 to 25 (well above `MIN_FACE_FRAMES = 3`). No video fell back to `"Uncertain"` due to face detection failure.

### Q7: Is the fusion engine expecting image evidence for videos and falling back to Uncertain?
**No.**
In `backend/fusion.py` (line 438), the check is `if not img["available"] and not vid["available"]:`. Since `vid["available"] == True`, this does not trigger. Video-only uploads are properly supported structurally.

---

## 3. Detailed Logs & Breakdown of Representative Videos

### Video 1: `WhatsApp Video 2026-09-15 at 9.23.21 AM.mp4`
- **Ground Truth:** Real (Authentic)
- **Phase 1 Prediction:** Fake
- **Phase 1 Raw Median:** 0.0972 | **Phase 1 Calibrated Score:** 0.6344
- **V5.2 Prediction:** Real | **V5.2 Temporal Score:** 0.1345 | **V5.2 Threshold:** 0.30
- **V5.2 Confidence:** 0.7758
- **Image Evidence Present:** No (video-only upload)
- **Metadata Evidence:** Available (no editing software, reliability weight: 0.05)
- **Fusion Weights:** `video: 0.90`, `metadata: 0.05`
- **Fused Multimodal Score:** 0.6344
- **Image Threshold:** 0.21 (Fused score distance from threshold: `|0.6344 - 0.21| = 0.4244`, very far!)
- **Why Conflict Triggered:** Phase 1 predicted `Fake` (calibrated 0.6344), but V5.2 predicted `Real` (score 0.1345). The rule `p1_pred != v5_2_pred` evaluated to `True`.
- **Why Overall Prediction Became Uncertain:** `video_detector.py` line 645 set `verdict = "Uncertain"`. In `fusion.py`, Condition 1 (`if conflict_detected: overall_pred = "Uncertain"`) executed before checking the fused score.
- **Condition Responsible:** Condition 1 (Inter-Modal / Inter-Pipeline Conflict) & Condition 2 (Video Verdict Uncertain).
- **Direction:** Phase 1 was **Fake**; V5.2 was **Real**. (V5.2 was correct).
- **Uncertainty Category:** **A (Phase 1 vs V5.2 Disagreement)**.

---

### Video 2: `WhatsApp Video 2026-09-15 at 9.27.24 AM.mp4`
- **Ground Truth:** Real (Authentic)
- **Phase 1 Prediction:** Fake
- **Phase 1 Raw Median:** 0.1659 | **Phase 1 Calibrated Score:** 0.3758
- **V5.2 Prediction:** Real | **V5.2 Temporal Score:** 0.1813 | **V5.2 Threshold:** 0.30
- **V5.2 Confidence:** 0.6978
- **Image Evidence Present:** No
- **Metadata Evidence:** Available (reliability weight: 0.05)
- **Fusion Weights:** `video: 0.90`, `metadata: 0.05`
- **Fused Multimodal Score:** 0.3758
- **Image Threshold:** 0.21 (Distance: `0.1658`)
- **Why Conflict Triggered:** Phase 1 predicted `Fake` (0.3758), while V5.2 predicted `Real` (0.1813).
- **Why Overall Prediction Became Uncertain:** Pipeline conflict check in `video_detector.py` line 643 and `fusion.py` line 483 triggered.
- **Condition Responsible:** Condition 1 (Inter-Modal Conflict).
- **Direction:** Phase 1 was **Fake**; V5.2 was **Real**. (V5.2 was correct).
- **Uncertainty Category:** **A (Phase 1 vs V5.2 Disagreement)**.

---

### Video 3: `WhatsApp Video 2026-09-15 at 9.30.53 AM (1).mp4`
- **Ground Truth:** Fake (Manipulated)
- **Phase 1 Prediction:** Real
- **Phase 1 Raw Median:** 0.9578 | **Phase 1 Calibrated Score:** 0.0112
- **V5.2 Prediction:** Fake | **V5.2 Temporal Score:** 0.5683 | **V5.2 Threshold:** 0.30
- **V5.2 Confidence:** 0.6916
- **Image Evidence Present:** No
- **Metadata Evidence:** Available (reliability weight: 0.05)
- **Fusion Weights:** `video: 0.90`, `metadata: 0.05`
- **Fused Multimodal Score:** 0.0112
- **Image Threshold:** 0.21 (Distance: `|0.0112 - 0.21| = 0.1988`, extremely far toward authentic!)
- **Why Conflict Triggered:** Phase 1 incorrectly deemed the video authentic (`Real`, calibrated score 0.0112) because raw median was 0.9578. V5.2 detected temporal BiLSTM manipulation patterns (`Fake`, score 0.5683).
- **Why Overall Prediction Became Uncertain:** Conflict rule intercepted the evaluation, overriding the 0.0112 score and forcing `Uncertain`.
- **Condition Responsible:** Condition 1 (Inter-Modal Conflict).
- **Direction:** Phase 1 was **Real**; V5.2 was **Fake**. (V5.2 was correct).
- **Uncertainty Category:** **A (Phase 1 vs V5.2 Disagreement)**.

---

### Video 4: `WhatsApp Video 2026-09-15 at 9.23.20 AM.mp4`
- **Ground Truth:** Real (Authentic)
- **Phase 1 Prediction:** Real
- **Phase 1 Raw Median:** 0.9128 | **Phase 1 Calibrated Score:** 0.0232
- **V5.2 Prediction:** Fake | **V5.2 Temporal Score:** 0.3595 | **V5.2 Threshold:** 0.30
- **V5.2 Confidence:** 0.5425
- **Image Evidence Present:** No
- **Metadata Evidence:** Available
- **Fused Multimodal Score:** 0.0232
- **Image Threshold:** 0.21
- **Why Conflict Triggered:** Phase 1 predicted `Real` (0.0232), while V5.2 slightly exceeded threshold at 0.3595 predicting `Fake`.
- **Why Overall Prediction Became Uncertain:** Disagreement triggered hard conflict.
- **Condition Responsible:** Condition 1 (Inter-Modal Conflict).
- **Direction:** Phase 1 was **Real**; V5.2 was **Fake**.
- **Uncertainty Category:** **A (Phase 1 vs V5.2 Disagreement)** & **D (V5.2 near threshold 0.3595)**.

---

### Video 5: `test.mp4` (External / In-the-Wild Video)
- **Ground Truth:** Fake
- **Phase 1 Prediction:** Real
- **Phase 1 Raw Median:** 0.9312 | **Phase 1 Calibrated Score:** 0.0194
- **V5.2 Prediction:** Fake | **V5.2 Temporal Score:** 0.4602 | **V5.2 Threshold:** 0.30
- **V5.2 Confidence:** 0.6144
- **Image Evidence Present:** No
- **Metadata Evidence:** Available
- **Fused Multimodal Score:** 0.0194
- **Image Threshold:** 0.21
- **Why Conflict Triggered:** Phase 1's inverted calibration assumed high raw score (0.9312) meant Real, while V5.2 BiLSTM correctly classified temporal inconsistency as Fake (0.4602).
- **Why Overall Prediction Became Uncertain:** Hard conflict triggered in `video_detector.py` and `fusion.py`.
- **Condition Responsible:** Condition 1 (Inter-Modal Conflict).
- **Direction:** Phase 1 was **Real**; V5.2 was **Fake**. (V5.2 was correct).
- **Uncertainty Category:** **A (Phase 1 vs V5.2 Disagreement)** & **H (Phase 1 Inverted Calibration Incompatibility with Normal Videos)**.

---

### Video 6: `WhatsApp Video 2026-09-15 at 9.23.09 AM.mp4`
- **Ground Truth:** Real
- **Phase 1 Prediction:** Real
- **Phase 1 Raw Median:** 0.6285 | **Phase 1 Calibrated Score:** 0.0988
- **V5.2 Prediction:** Uncertain | **V5.2 Temporal Score:** 0.3234 | **V5.2 Threshold:** 0.30
- **V5.2 Confidence:** 0.5000 (Boundary window: `[0.25, 0.35]`)
- **Image Evidence Present:** No
- **Fused Multimodal Score:** 0.0988
- **Why Conflict was NOT Triggered:** Because V5.2 prediction was `"Uncertain"`, the conflict check `if p1_pred in ["Real", "Fake"] and v5_2_pred in ["Real", "Fake"]` did not fire.
- **Why Overall Prediction was REAL:** Phase 1 remained decisive (`Real`), and V5.2 was within its margin, allowing the system to output `Real`.
- **Uncertainty Category:** **D (Temporal score close to threshold [0.25, 0.35])**.

---

## 4. Complete Benchmark Summary Table

| Video Filename | True Label | P1 Pred | P1 Score | V5.2 Pred | V5.2 Score | Fused Score | Conflict? | Overall Verdict | Uncertainty Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `WhatsApp Video 9.23.07 AM.mp4` | **Real** | Real | 0.1601 | Real | 0.2306 | 0.1601 | No | **Real** | Decisive |
| `WhatsApp Video 9.23.08 AM.mp4` | **Real** | Real | 0.1548 | Real | 0.2214 | 0.1548 | No | **Real** | Decisive |
| `WhatsApp Video 9.23.09 AM.mp4` | **Real** | Real | 0.0988 | Uncertain | 0.3234 | 0.0988 | No | **Real** | D (V5.2 in boundary) |
| `WhatsApp Video 9.23.11 AM.mp4` | **Real** | Real | 0.1555 | Uncertain | 0.2513 | 0.1555 | No | **Real** | D (V5.2 in boundary) |
| `WhatsApp Video 9.23.12 AM.mp4` | **Real** | Real | 0.0846 | Uncertain | 0.2886 | 0.0846 | No | **Real** | D (V5.2 in boundary) |
| `WhatsApp Video 9.23.17 AM.mp4` | **Real** | Real | 0.0817 | Real | 0.2008 | 0.0817 | No | **Real** | Decisive |
| `WhatsApp Video 9.23.20 AM.mp4` | **Real** | Real | 0.0232 | Fake | 0.3595 | 0.0232 | **YES** | **Uncertain** | **A (Disagreement)** |
| `WhatsApp Video 9.23.21 AM.mp4` | **Real** | Fake | 0.6344 | Real | 0.1345 | 0.6344 | **YES** | **Uncertain** | **A (P1 Error vs V5.2 Correct)** |
| `WhatsApp Video 9.24.25 AM.mp4` | **Real** | Real | 0.1450 | Real | 0.2491 | 0.1450 | No | **Real** | Decisive |
| `WhatsApp Video 9.27.24 AM.mp4` | **Real** | Fake | 0.3758 | Real | 0.1813 | 0.3758 | **YES** | **Uncertain** | **A (P1 Error vs V5.2 Correct)** |
| `WhatsApp Video 9.30.51 AM.mp4` | **Fake** | Fake | 0.9876 | Fake | 0.6782 | 0.9876 | No | **Likely Fake** | Decisive |
| `WhatsApp Video 9.30.52 AM (1).mp4`| **Fake** | Fake | 0.8882 | Fake | 0.3606 | 0.8882 | No | **Likely Fake** | Decisive |
| `WhatsApp Video 9.30.52 AM.mp4` | **Fake** | Fake | 0.9484 | Fake | 0.6239 | 0.9484 | No | **Likely Fake** | Decisive |
| `WhatsApp Video 9.30.53 AM (1).mp4`| **Fake** | Real | 0.0112 | Fake | 0.5683 | 0.0112 | **YES** | **Uncertain** | **A (P1 Error vs V5.2 Correct)** |
| `WhatsApp Video 9.30.53 AM (2).mp4`| **Fake** | Fake | 0.9742 | Fake | 0.7215 | 0.9742 | No | **Likely Fake** | Decisive |
| `WhatsApp Video 9.30.53 AM.mp4` | **Fake** | Fake | 0.9992 | Fake | 0.6193 | 0.9992 | No | **Likely Fake** | Decisive |
| `WhatsApp Video 9.30.54 AM.mp4` | **Fake** | Fake | 0.9616 | Fake | 0.5137 | 0.9616 | No | **Likely Fake** | Decisive |
| `WhatsApp Video 9.30.55 AM (1).mp4`| **Fake** | Fake | 0.9943 | Fake | 0.5860 | 0.9943 | No | **Likely Fake** | Decisive |
| `WhatsApp Video 9.30.55 AM.mp4` | **Fake** | Fake | 0.8944 | Fake | 0.5667 | 0.8944 | No | **Likely Fake** | Decisive |
| `WhatsApp Video 9.36.31 AM.mp4` | **Fake** | Fake | 0.9183 | Fake | 0.3532 | 0.9183 | No | **Likely Fake** | Decisive |
| `test.mp4` (External) | **Fake** | Real | 0.0194 | Fake | 0.4602 | 0.0194 | **YES** | **Uncertain** | **A & H (Inverted Heuristic)** |
| `real_check.mp4` (Sample) | **Real** | Real | 0.1548 | Real | 0.2214 | 0.1548 | No | **Real** | Decisive |
| `fake_check.mp4` (Sample) | **Fake** | Fake | 0.9992 | Fake | 0.6193 | 0.9992 | No | **Likely Fake** | Decisive |
| `test1.mp4` (Sample) | **Real** | Real | 0.0988 | Uncertain | 0.3234 | 0.0988 | No | **Real** | D (V5.2 in boundary) |

### Benchmark Aggregate Statistics (WhatsApp 20-Video Set)
- **Total Videos Analyzed:** 20
- **Real:** 7 (35.0%)
- **Likely Fake:** 9 (45.0%)
- **Uncertain:** 4 (20.0%)
- **Cause of Uncertainty:** 4 / 4 (100%) due to **Category A: Phase 1 vs V5.2 Disagreement**.

---

## 5. Root Cause Analysis & Exact Lines of Code

### Line 639–658 in `backend/video_detector.py`
```python
639: if v5_2_temporal_evidence and v5_2_temporal_evidence.get("prediction") in ["Real", "Fake"]:
640:     p1_pred = prediction
641:     v5_2_pred = v5_2_temporal_evidence["prediction"]
642:     if p1_pred in ["Real", "Fake"]:
643:         if p1_pred != v5_2_pred:
644:             conflict_detected = True
645:             verdict = "Uncertain"
646:             prediction = "Uncertain"
647:             confidence = 0.50
```
**Mechanism:** The detector directly overwrites its own verdict to `"Uncertain"` if Phase 1 and V5.2 differ, without evaluating confidence or sequence validity.

### Line 482–489 in `backend/fusion.py`
```python
482: if p1_pred in ["Real", "Fake"] and v5_2_pred in ["Real", "Fake"]:
483:     if p1_pred != v5_2_pred:
484:         conflict_detected = True
485:         conflict_factors.append(
486:             f"Temporal vs Spatial Video Conflict: DeepShield V5.2 temporal model ({v5_2_pred}, "
487:             f"score: {v5_2_ev.get('temporal_score', 0.0):.4f}, threshold: {v5_2_ev.get('temporal_threshold', 0.30)}) "
488:             f"disagrees with Phase 1 frame analysis ({p1_pred}, calibrated score: {p1_ev.get('video_calibrated_score', 0.0):.4f})."
489:         )
```
**Mechanism:** Re-duplicates the conflict check at the fusion level and marks `conflict_detected = True`.

### Line 598–606 in `backend/fusion.py`
```python
598: # Condition 1: Significant Inter-Modal Conflict
599: if conflict_detected:
600:     overall_pred = "Uncertain"
601:     confidence = 0.50
602:     uncertainty_reason = "Significant inter-modal conflict detected between image and video evidence streams."
```
**Mechanism:** Condition 1 is evaluated **first** before checking `fused_score` or its distance from threshold. Thus, even if `fused_score` is 0.0112 or 0.6344 (far from 0.21), the verdict is forced to `"Uncertain"`.

### Line 575–578 in `backend/fusion.py`
```python
575: if vid["available"]:
576:     weights.append(vid["reliability_weight"])
577:     scores.append(vid["representative_score"])
```
**Mechanism:** `vid["representative_score"]` is assigned to Phase 1's `video_calibrated_score`. V5.2's `temporal_score` is completely excluded from the numerical fusion formula.

---

## 6. Recommended Minimal Fix Plan (For Future Phase)

To eliminate spurious `"Uncertain"` predictions while maintaining strict safety, responsible AI standards, and calibrated thresholds:

1. **Temporal Authority Hierarchy (Confidence-Weighted Arbitration):**
   Instead of a flat `if p1_pred != v5_2_pred: verdict = "Uncertain"`:
   - When V5.2 has high confidence (e.g. `temporal_score >= 0.40` or `temporal_score <= 0.20`, or confidence > 0.65) and 16 valid face frames, **V5.2 is recognized as the primary temporal authority** and arbitrates the verdict.
   - Genuine `"Uncertain"` should trigger only when **both** models are near their respective decision boundaries or when video quality/sequence length is truly ambiguous.
2. **Unified Directional Representation in Fusion:**
   - Integrate V5.2 temporal score into `fuse_multimodal_evidence` as a first-class evidence stream.
   - When video is evaluated, blend Phase 1 spatial frame consensus with V5.2 temporal dynamics weighted by sequence quality.
3. **Preserve Safe Boundaries Without False Vetoes:**
   - Keep `IMAGE_THRESHOLD = 0.21` and `VIDEO_TEMPORAL_THRESHOLD = 0.30` unchanged.
   - Ensure inter-modal conflict flags actual image-vs-video contradictions rather than internal model corrections.

---
*DeepShield V5.2 Forensic Research Team*
