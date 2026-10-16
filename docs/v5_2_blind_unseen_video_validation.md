# DeepShield V5.2: Blind Unseen Video Validation Report
**Date:** 2026-09-18  
**Status:** Complete & Verified  
**Evaluation Scope:** Blind evaluation of current production pipeline on held-out unseen mobile videos  
**Thresholds:** `IMAGE_THRESHOLD = 0.21`, `VIDEO_TEMPORAL_THRESHOLD = 0.30`, `VIDEO_TEMPORAL_MARGIN = 0.03`  

> [!NOTE]
> **Scientific Integrity Notice:** This evaluation was conducted on an internal held-out validation corpus of in-the-wild mobile messaging videos. It is intended for internal pipeline verification and generalizability assessment, not as an exhaustive large-scale scientific benchmark across multiple synthetic manipulation domains.

---

## 1. Executive Summary

A completely blind evaluation was conducted on **19 unseen in-the-wild videos** (9 authentic Real, 10 manipulated Fake). All videos were transmitted through mobile compression (WhatsApp transcoding) and had never been used for model training, validation checkpoint selection, threshold calibration, or fusion rule tuning.

Ground-truth labels were kept strictly segregated from the inference pipeline; the detector operated exclusively on raw file paths.

### Key Benchmark Metrics:
- **Total Unseen Videos:** 19 (9 Real, 10 Fake)
- **Overall Decisive Coverage:** **94.74%** (18 / 19 videos resolved decisively)
- **Overall Uncertainty Rate:** **5.26%** (1 / 19 videos returned Uncertain)
- **Decisive Accuracy:** **100.00%** (18 / 18 correct among decisive verdicts)
- **Overall Accuracy:** **94.74%** (18 / 19 correct across all evaluated videos)
- **Precision (Decisive):** **1.0000** (100.00%)
- **Recall (Decisive):** **1.0000** (100.00%)
- **F1 Score (Decisive):** **1.0000**
- **False Positives:** **0** (0 / 8 decisive Real videos falsely flagged)
- **False Negatives:** **0** (0 / 10 decisive Fake videos missed)

---

## 2. Dataset Composition & Isolation Protocol

### 2.1 Video Selection Criteria:
1. **Strict Exclusion of Development Assets:**
   - `backend/test_images/real_check.mp4` (excluded)
   - `backend/test_images/fake_check.mp4` (excluded)
   - `backend/test_images/test.mp4` (excluded)
   - Previously tested unseen video: `WhatsApp Video 2026-09-15 at 12.54.43 PM (1).mp4` (excluded)
   - 20-video WhatsApp calibration benchmark (`videos_real/` and `videos_fake/`) (strictly excluded)
2. **Corpus Assets:**
   - **Real Partition (9 videos):** Authentic mobile recordings, diverse subjects, varying resolutions ($406 \times 720$ to $848 \times 480$), framerates ($18.0$ to $63.1$ FPS), natural head movements, and mobile camera noise.
   - **Fake Partition (10 videos):** Synthetic deepfakes circulating on mobile messaging platforms, portrait orientation ($720 \times 1280$), high frame counts ($142$ to $1,402$ frames), face swapping, and reenactment under social media compression.

---

## 3. Comprehensive Per-Video Forensic Results

The table below documents the detailed forensic readout across every stage of the pipeline for all 19 unseen videos:

| # | Filename | Ground Truth | Frames (Total/Face) | Phase 1 Pred (Score, Volatility) | V5.2 Pred (Score, Conf) | Video Reliability | Fused Score | Conflict? | Final Verdict | Uncertainty Reason |
| :-: | :--- | :-: | :-: | :--- | :--- | :-: | :-: | :-: | :---: | :--- |
| 1 | `12.49.50 PM.mp4` | **Real** | 6 / 6 | Fake (0.6991, v=0.17) | Real (0.2410, c=0.60) | 0.15 | 0.4339 | True | **Uncertain** | Significant inter-modal conflict detected between image and video evidence streams. |
| 2 | `12.54.42 PM (1).mp4` | **Real** | 7 / 7 | Real (0.0093, v=0.38) | Real (0.1338, c=0.78) | 0.18 | 0.0599 | False | **Real** | None |
| 3 | `12.54.42 PM.mp4` | **Real** | 40 / 40 | Fake (0.9199, v=0.11) | Real (0.1572, c=0.74) | 0.36 | 0.1100 | False | **Real** | None |
| 4 | `12.54.43 PM.mp4` | **Real** | 22 / 16 | Real (0.0680, v=0.20) | Real (0.0841, c=0.86) | 0.18 | 0.0625 | False | **Real** | None |
| 5 | `12.54.44 PM (1).mp4` | **Real** | 40 / 40 | Fake (0.9401, v=0.14) | Real (0.1266, c=0.79) | 0.18 | 0.0886 | False | **Real** | None |
| 6 | `12.54.44 PM.mp4` | **Real** | 40 / 40 | Real (0.1058, v=0.34) | Real (0.2058, c=0.66) | 0.18 | 0.1288 | False | **Real** | None |
| 7 | `12.54.45 PM.mp4` | **Real** | 40 / 40 | Real (0.1397, v=0.27) | Real (0.1753, c=0.71) | 0.18 | 0.1295 | False | **Real** | None |
| 8 | `12.54.51 PM (1).mp4` | **Real** | 40 / 24 | Uncertain (0.2059, v=0.28) | Real (0.2069, c=0.66) | 0.18 | 0.1754 | False | **Real** | None |
| 9 | `12.54.51 PM.mp4` | **Real** | 40 / 20 | Real (0.0748, v=0.31) | Real (0.2564, c=0.57) | 0.18 | 0.1376 | False | **Real** | None |
| 10 | `1.31.13 PM.mp4` | **Fake** | 15 / 15 | Fake (0.6445, v=0.28) | Fake (0.6739, c=0.77) | 0.18 | 0.6370 | False | **Likely Fake** | None |
| 11 | `1.33.18 PM.mp4` | **Fake** | 40 / 40 | Fake (0.5797, v=0.28) | Fake (0.7311, c=0.81) | 0.18 | 0.6498 | False | **Likely Fake** | None |
| 12 | `1.33.34 PM.mp4` | **Fake** | 40 / 40 | Real (0.0174, v=0.21) | Fake (0.8056, c=0.86) | 0.18 | 0.7806 | False | **Likely Fake** | None |
| 13 | `1.35.59 PM (1).mp4` | **Fake** | 40 / 26 | Real (0.1913, v=0.32) | Fake (0.4419, c=0.60) | 0.45 | 0.3701 | False | **Likely Fake** | None |
| 14 | `1.35.59 PM.mp4` | **Fake** | 31 / 31 | Fake (0.7177, v=0.31) | Fake (0.7712, c=0.84) | 0.18 | 0.7321 | False | **Likely Fake** | None |
| 15 | `1.36.00 PM (1).mp4` | **Fake** | 40 / 35 | Fake (0.9328, v=0.19) | Fake (0.7069, c=0.79) | 0.36 | 0.7746 | False | **Likely Fake** | None |
| 16 | `1.36.00 PM.mp4` | **Fake** | 40 / 40 | Real (0.0014, v=0.02) | Fake (0.8018, c=0.86) | 0.36 | 0.7763 | False | **Likely Fake** | None |
| 17 | `1.43.30 PM.mp4` | **Fake** | 40 / 40 | Fake (0.9844, v=0.06) | Fake (0.6408, c=0.74) | 0.18 | 0.7505 | False | **Likely Fake** | None |
| 18 | `1.43.31 PM (1).mp4` | **Fake** | 20 / 20 | Fake (0.5872, v=0.11) | Fake (0.7774, c=0.84) | 0.18 | 0.6841 | False | **Likely Fake** | None |
| 19 | `1.43.31 PM.mp4` | **Fake** | 37 / 32 | Real (0.1034, v=0.19) | Fake (0.7723, c=0.84) | 0.18 | 0.7430 | False | **Likely Fake** | None |

---

## 4. Performance Metrics & Confusion Matrix

### 4.1 Outcome Distribution Across Full Dataset (N = 19)

| Ground Truth | Decisive: Real | Decisive: Likely Fake | Uncertain | Total | Category Accuracy (Decisive) |
| :--- | :-: | :-: | :-: | :-: | :-: |
| **Real Videos** | **8** | 0 | 1 | 9 | **100.0%** (8 / 8) |
| **Fake Videos** | 0 | **10** | 0 | 10 | **100.0%** (10 / 10) |
| **Total** | **8** | **10** | **1** | **19** | **100.0%** (18 / 18) |

### 4.2 Confusion Matrix (Decisive Predictions Only)
In accordance with scientific reporting standards, `Uncertain` results are not forced into binary classes.

```
                      PREDICTED
                 Real         Likely Fake
ACTUAL  Real     TN = 8         FP = 0
        Fake     FN = 0         TP = 10
```

### 4.3 Statistical Performance Summary:
- **Decisive Sample Size:** 18 / 19 videos
- **True Positives (TP):** 10
- **True Negatives (TN):** 8
- **False Positives (FP):** 0
- **False Negatives (FN):** 0
- **Precision:** $\frac{TP}{TP + FP} = \frac{10}{10 + 0} = \mathbf{1.0000\text{ (100.0%)}}$
- **Recall:** $\frac{TP}{TP + FN} = \frac{10}{10 + 0} = \mathbf{1.0000\text{ (100.0%)}}$
- **F1 Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \mathbf{1.0000}$
- **Decisive Accuracy:** $\frac{TP + TN}{18} = \frac{18}{18} = \mathbf{100.00\%}$
- **Overall Coverage:** $\frac{18}{19} = \mathbf{94.74\%}$
- **Overall Full-Set Accuracy:** $\frac{TP + TN}{19} = \frac{18}{19} = \mathbf{94.74\%}$

---

## 5. Failure Case & Uncertainty Analysis

Across the entire 19-video blind test set, exactly **one video** yielded an `Uncertain` verdict, and **zero videos** were misclassified.

### Detailed Analysis of Uncertain Video:
- **Video:** `WhatsApp Video 2026-09-15 at 12.49.50 PM.mp4` (Index 1)
- **Ground Truth:** Real
- **Clip Characteristics:** Extremely short recording (59 total frames, $2.56\text{ seconds}$ at $23.0\text{ FPS}$, resolution $480 \times 864$). Only 6 keyframes could be sampled and verified.
- **Model Evidence Streams:**
  - **Phase 1 Spatial Heuristic:** Predicted `Fake` (raw median score: $0.0573$, calibrated fake score: $0.6991$, volatility: $0.1701$). Under Phase 1's heuristic mapping, very low raw frame scores indicate synthetic manipulation.
  - **V5.2 Dedicated Temporal Model:** Predicted `Real` (temporal score: **$0.2410$**, threshold: $0.30$, confidence: $0.5983$).
- **Failure Classification:**
  - **Primary Cause: C. Phase 1 Disagreement & B. Insufficient Valid Face Frames**
  - Because the video contained only 6 face frames, the V5.2 high-confidence authority threshold (which requires $\ge 16$ valid face frames) was not met.
  - Consequently, V5.2 could not unilaterally override Phase 1. The pipeline correctly detected a strong inter-modal contradiction between Phase 1 (`Fake`, $0.6991$) and V5.2 (`Real`, $0.2410$) and reported `Uncertain` rather than guessing incorrectly.
  - **Assessment:** DeepShield acted safely. In a real-world setting, a 2.5-second clip with conflicting spatial and temporal indicators warrants forensic caution rather than a false positive conviction.

---

## 6. Generalization & Operational Limitations

1. **Validation Corpus Scope:**
   - This evaluation demonstrates strong resilience to WhatsApp transcoding artifacts across mobile devices.
   - However, this is an internal validation set of 19 clips. True scientific generalization requires testing on external public benchmarks (e.g., DFDC held-out, Celeb-DF v2 full test set, FaceForensics++ c23/c40 held-out).
2. **Ultra-Short Video Clips (< 3 seconds):**
   - Clips under 3 seconds provide fewer than 16 face frames, preventing V5.2 temporal authority from overriding Phase 1 when spatial compression artifacts trigger Phase 1 false alarms.
3. **Threshold Stability:**
   - No thresholds were adjusted during this blind evaluation. All metrics reflect the raw performance of DeepShield V5.2 under calibrated constants (`IMAGE_THRESHOLD = 0.21`, `VIDEO_TEMPORAL_THRESHOLD = 0.30`, `VIDEO_TEMPORAL_MARGIN = 0.03`).

---

## 7. Regression Testing Confirmation

- **Command:** `python -m unittest discover -s backend/tests -p "test_*.py"`
- **Outcome:** **Ran 65 tests in 34.066s — OK**
- Zero production code modifications were introduced.
- Full backward compatibility maintained across all backend services.
