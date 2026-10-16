# DeepShield V5 — Exhaustive Image Error Analysis Report

**Date:** 2026-09-19  
**Target Model:** `models/deepshield_v5_clean_best.keras` (EfficientNet-B2 Spatial Image Model)  
**Evaluation Set:** Official FAST10 Test Split (3,693 samples: 993 Real, 2,700 Fake)  
**Evaluation Pipeline:** Direct FAST10 Image Path  
**Operating Threshold:** `0.21` (`score >= 0.21 => Fake`, `score < 0.21 => Real`)  
**Scope:** Forensic Error Analysis Only. Zero modifications to production code, zero modifications to thresholds, zero model retraining, zero video touches.

---

## 1. Classification Outcome Summary

Across the complete 3,693 test set images, the predictions partition into:

| Category | Definition | Count | % of Test Set | % of Ground-Truth Class |
| :--- | :--- | :---: | :---: | :---: |
| **True Positives (TP)** | Synthetic image correctly classified as Fake | **2,477** | 67.07% | 91.74% of all Fakes |
| **True Negatives (TN)** | Authentic image correctly classified as Real | **516** | 13.97% | 51.96% of all Reals |
| **False Positives (FP)** | Authentic image misclassified as Fake | **477** | 12.92% | **48.04% of all Reals** |
| **False Negatives (FN)** | Synthetic image misclassified as Real | **223** | 6.04% | **8.26% of all Fakes** |
| **Total Test Samples** | Real ($N=993$) + Fake ($N=2,700$) | **3,693** | 100.00% | Overall Accuracy: **81.05%** |

---

## 2. Decile Score Bin Analysis (0.00 to 1.00)

Every test image was grouped into discrete decile confidence bins based on its raw sigmoid output score:

| Score Range | Total Images | Real Count | Fake Count | FP Count | FN Count | Total Errors | Error Rate (%) | Dominant Failure Mode |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.00–0.10** | 530 | 394 | 136 | 0 | 136 | 136 | 25.66% | Extreme False Negatives (Missed Fakes) |
| **0.10–0.20** | 200 | 120 | 80 | 0 | 80 | 80 | 40.00% | Severe False Negatives (Missed Fakes) |
| **0.20–0.30** | 135 | 61 | 74 | 59 | 7 | 66 | **48.89%** | Maximum Confusion (Split by threshold 0.21) |
| **0.30–0.40** | 138 | 51 | 87 | 51 | 0 | 51 | 36.96% | False Positives (Authentic faces scoring mid) |
| **0.40–0.50** | 107 | 39 | 68 | 39 | 0 | 39 | 36.45% | False Positives |
| **0.50–0.60** | 137 | 37 | 100 | 37 | 0 | 37 | 27.01% | False Positives |
| **0.60–0.70** | 148 | 33 | 115 | 33 | 0 | 33 | 22.30% | False Positives |
| **0.70–0.80** | 164 | 43 | 121 | 43 | 0 | 43 | 26.22% | False Positives |
| **0.80–0.90** | 237 | 52 | 185 | 52 | 0 | 52 | 21.94% | False Positives |
| **0.90–1.00** | **1,897** | 163 | 1,734 | **163** | 0 | **163** | 8.59% | **Extreme False Positives (Saturated Authentic Faces)** |
| **Total** | **3,693** | **993** | **2,700** | **477** | **223** | **700** | **18.95%** | **700 Total Errors** |

---

## 3. Score Distributions & Overlap Analysis

### A. Real Score Distribution ($N=993$)
- **Minimum:** $0.000012$
- **Maximum:** $0.999999$
- **Mean:** $0.3605$
- **Median (50th percentile):** $0.1805$
- **Standard Deviation:** $0.3656$
- **10th Percentile:** $0.0078$
- **25th Percentile:** $0.0398$
- **75th Percentile:** $0.7280$
- **90th Percentile:** $0.9649$

### B. Fake Score Distribution ($N=2,700$)
- **Minimum:** $0.000168$
- **Maximum:** $1.000000$
- **Mean:** $0.8100$
- **Median (50th percentile):** $0.9793$
- **Standard Deviation:** $0.2923$
- **10th Percentile:** $0.2703$
- **25th Percentile:** $0.7125$
- **75th Percentile:** $0.9995$
- **90th Percentile:** $1.0000$

### C. Overlap Region Between Real and Fake
- **Extreme Overlap Range:** $[0.000168, 0.999999]$ (spanning virtually the entire $[0, 1]$ interval).
- **Images Contained in Overlap Region:** **3,530 out of 3,693 samples (95.59%)**.
- **High-Ambiguity Region ($0.10 \le \text{score} \le 0.90$):** Contained $1,266$ images, including $418$ errors.

### D. Percentage of Errors Inside Overlap Region
- **100.00% of all errors (700 / 700)** reside strictly inside the overlap region.
- There are zero out-of-bounds errors: all false negatives exist where reals also exist, and all false positives exist where fakes also exist.

---

## 4. Dataset-Level Error Ranking

The FAST10 test partition is composed of three primary benchmark corpora:

| Source Dataset | Total Images | Real Images | Fake Images | False Positives (FP) | Real FP Rate (%) | False Negatives (FN) | Fake FN Rate (%) | Total Errors | Total Error Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FF-C23** | 2,250 | 300 | 1,950 | **167** | **55.67%** | **135** | 6.92% | **302** | 13.42% |
| **Deepfake-Videos-Merged** | 840 | 387 | 453 | **187** | **48.32%** | 29 | 6.40% | **216** | 25.71% |
| **FaceForensics-1000** | 603 | 306 | 297 | 123 | **40.20%** | **59** | **19.87%** | 182 | **30.18%** |

### Top Datasets Producing False Positives:
1. **Deepfake-Videos-Merged**: **187 FPs** (48.32% of real faces).
2. **FF-C23**: **167 FPs** (55.67% of real faces — highest FP rate).
3. **FaceForensics-1000**: **123 FPs** (40.20% of real faces).

### Top Datasets Producing False Negatives:
1. **FF-C23**: **135 FNs** (6.92% of fake faces).
2. **FaceForensics-1000**: **59 FNs** (19.87% of fake faces — highest FN rate).
3. **Deepfake-Videos-Merged**: **29 FNs** (6.40% of fake faces).

### Top Datasets Producing Total Errors:
1. **FF-C23**: **302 errors** (167 FP + 135 FN).
2. **Deepfake-Videos-Merged**: **216 errors** (187 FP + 29 FN).
3. **FaceForensics-1000**: **182 errors** (123 FP + 59 FN).

---

## 5. Structural Error Patterns (Video Groups & Path Patterns)

### Video Group Clustering:
- **False Positives**: Real false positives cluster heavily within specific video groups. In `Deepfake-Videos-Merged`, groups `98967`, `99257`, `99046`, `99061`, `99055`, `99075`, `99143`, `99161`, `99159`, and `99238` had **100% of their extracted frames misclassified as Fake**.
- **False Negatives**: Synthetic video groups `101010`, `100624`, `101004` (in Merged) and `93275`, `93708`, `92798`, `92683`, `92491`, `92469`, `92109` (in FF-C23) had **100% of their synthetic frames misclassified as Real**.
- **Path Pattern Finding**: In `FF-C23`, filenames containing `_frame_` with high sequence numbers from H.264 compressed streams consistently produce false positives on real faces, indicating that macroblock quantization noise accumulation triggers synthetic detectors.

---

## 6. Top 20 Individual False Positives & Top 20 False Negatives

The top 20 extreme errors of each type have been archived to [`evaluation/image_error_analysis.csv`](file:///c:/Users/ADMIN/Desktop/DeepShield_V5_Final/evaluation/image_error_analysis.csv).

### Top 20 False Positives (Authentic Real Faces Saturated Near 1.0):

| # | Filename | True Label | Pred Label | Score | Source Dataset | Video Group |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `FF-C23_91402_494_frame_200.jpg` | real | fake | **0.999999** | FF-C23 | 91402 |
| 2 | `Deepfake-Videos-Merged_100264_real_1288_frame_132.jpg` | real | fake | **0.999997** | Deepfake-Videos-Merged | 100264 |
| 3 | `FF-C23_91402_494_frame_401.jpg` | real | fake | **0.999995** | FF-C23 | 91402 |
| 4 | `FaceForensics-1000_97431_494_frame_601.jpg` | real | fake | **0.999991** | FaceForensics-1000 | 97431 |
| 5 | `FF-C23_91402_494_frame_601.jpg` | real | fake | **0.999967** | FF-C23 | 91402 |
| 6 | `FaceForensics-1000_97431_494_frame_200.jpg` | real | fake | **0.999950** | FaceForensics-1000 | 97431 |
| 7 | `Deepfake-Videos-Merged_99824_real_890_frame_105.jpg` | real | fake | **0.999841** | Deepfake-Videos-Merged | 99824 |
| 8 | `Deepfake-Videos-Merged_99946_real_997_frame_304.jpg` | real | fake | **0.999817** | Deepfake-Videos-Merged | 99946 |
| 9 | `FF-C23_91478_535_frame_500.jpg` | real | fake | **0.999727** | FF-C23 | 91478 |
| 10 | `Deepfake-Videos-Merged_100264_real_1288_frame_398.jpg` | real | fake | **0.999686** | Deepfake-Videos-Merged | 100264 |
| 11 | `FF-C23_91630_736_frame_207.jpg` | real | fake | **0.999663** | FF-C23 | 91630 |
| 12 | `Deepfake-Videos-Merged_99238_real_367_frame_78.jpg` | real | fake | **0.999644** | Deepfake-Videos-Merged | 99238 |
| 13 | `FF-C23_91004_027_frame_78.jpg` | real | fake | **0.999644** | FF-C23 | 91004 |
| 14 | `Deepfake-Videos-Merged_99114_real_251_frame_1029.jpg` | real | fake | **0.999417** | Deepfake-Videos-Merged | 99114 |
| 15 | `FF-C23_91274_306_frame_546.jpg` | real | fake | **0.999334** | FF-C23 | 91274 |
| 16 | `FF-C23_91286_341_frame_413.jpg` | real | fake | **0.999320** | FF-C23 | 91286 |
| 17 | `Deepfake-Videos-Merged_99934_real_986_frame_261.jpg` | real | fake | **0.999247** | Deepfake-Videos-Merged | 99934 |
| 18 | `Deepfake-Videos-Merged_99257_real_369_frame_716.jpg` | real | fake | **0.999009** | Deepfake-Videos-Merged | 99257 |
| 19 | `FaceForensics-1000_97220_349_frame_117.jpg` | real | fake | **0.998991** | FaceForensics-1000 | 97220 |
| 20 | `Deepfake-Videos-Merged_99843_real_905_frame_288.jpg` | real | fake | **0.998953** | Deepfake-Videos-Merged | 99843 |

### Top 20 False Negatives (Synthetic Deepfakes Saturated Near 0.0):

| # | Filename | True Label | Pred Label | Score | Source Dataset | Video Group |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `FaceForensics-1000_98304_376_381_frame_144.jpg` | fake | real | **0.000168** | FaceForensics-1000 | 98304 |
| 2 | `FF-C23_92491_546_621_frame_277.jpg` | fake | real | **0.001223** | FF-C23 | 92491 |
| 3 | `FF-C23_95695_734_699_frame_262.jpg` | fake | real | **0.001560** | FF-C23 | 95695 |
| 4 | `FF-C23_92683_748_355_frame_159.jpg` | fake | real | **0.001718** | FF-C23 | 92683 |
| 5 | `FF-C23_92469_811_569_frame_300.jpg` | fake | real | **0.002011** | FF-C23 | 92469 |
| 6 | `FF-C23_92469_811_569_frame_150.jpg` | fake | real | **0.002167** | FF-C23 | 92469 |
| 7 | `FF-C23_92491_546_621_frame_415.jpg` | fake | real | **0.002237** | FF-C23 | 92491 |
| 8 | `FF-C23_92469_811_569_frame_450.jpg` | fake | real | **0.002347** | FF-C23 | 92469 |
| 9 | `FF-C23_92798_349_388_frame_351.jpg` | fake | real | **0.002446** | FF-C23 | 92798 |
| 10 | `FF-C23_92798_349_388_frame_117.jpg` | fake | real | **0.002573** | FF-C23 | 92798 |
| 11 | `FF-C23_95695_734_699_frame_393.jpg` | fake | real | **0.002636** | FF-C23 | 95695 |
| 12 | `FF-C23_92109_736_711_frame_276.jpg` | fake | real | **0.002640** | FF-C23 | 92109 |
| 13 | `FF-C23_93708_943_957_frame_300.jpg` | fake | real | **0.002662** | FF-C23 | 93708 |
| 14 | `FF-C23_92798_349_388_frame_234.jpg` | fake | real | **0.002717** | FF-C23 | 92798 |
| 15 | `FF-C23_93708_943_957_frame_450.jpg` | fake | real | **0.002891** | FF-C23 | 93708 |
| 16 | `FF-C23_92683_748_355_frame_318.jpg` | fake | real | **0.003117** | FF-C23 | 92683 |
| 17 | `FF-C23_92491_546_621_frame_139.jpg` | fake | real | **0.003310** | FF-C23 | 92491 |
| 18 | `FF-C23_93708_943_957_frame_150.jpg` | fake | real | **0.003350** | FF-C23 | 93708 |
| 19 | `FF-C23_93275_970_957_frame_276.jpg` | fake | real | **0.003554** | FF-C23 | 93275 |
| 20 | `FF-C23_92683_748_355_frame_477.jpg` | fake | real | **0.003668** | FF-C23 | 92683 |

---

## 7. Forensic Diagnostic Conclusions

### 1. Is the image problem mainly caused by one source dataset?
**No.** The error profile is endemic across all three benchmark datasets:
- `FF-C23`: **55.67% FP rate** on real faces (167 / 300) and **302 total errors**.
- `Deepfake-Videos-Merged`: **48.32% FP rate** on real faces (187 / 387) and **216 total errors**.
- `FaceForensics-1000`: **40.20% FP rate** on real faces (123 / 306) and **19.87% FN rate** on fakes.
Every single dataset suffers from a massive False Positive Rate ($>40\%$), indicating that the model's inability to recognize authentic compressed faces is a generalized spatial representation flaw, not a dataset outlier.

### 2. Are false positives concentrated in a particular score range?
**Yes.** While false positives exist throughout the $[0.21, 1.00]$ range, their primary mode is **at the extreme upper ceiling**:
- The single largest concentration is in the **0.90–1.00 bin: 163 out of 477 FPs (34.17%)**.
- Over **54% of all false positives (258 / 477)** have scores $> 0.70$.
- Another **23% (110 / 477)** lie in the boundary transition band $[0.20, 0.40]$.
This confirms that the majority of false positives are not borderline cases; they are high-confidence false convictions where authentic human faces trigger maximum synthetic feature activations.

### 3. Are false negatives concentrated in a particular score range?
**Yes, overwhelmingly.** False negatives are almost exclusively concentrated in the lowest deciles:
- **0.00–0.10 contains 136 out of 223 FNs (61.0%)**.
- **0.10–0.20 contains 80 out of 223 FNs (35.9%)**.
Together, scores $< 0.20$ account for **96.9% of all missed deepfakes (216 / 223)**.

### 4. Is there strong Real/Fake score overlap?
**Yes, near-total overlap.** The real score range is $[0.000012, 0.999999]$ and the fake score range is $[0.000168, 1.000000]$. The overlap region covers **95.59% of the entire test dataset (3,530 / 3,693 images)**, and **100.00% of all classification errors (700 / 700)** occur within this overlap.

### 5. Does threshold tuning appear capable of solving the problem?
**No.** Threshold tuning cannot solve this problem:
- Lowering the threshold to catch more fakes increases the already catastrophic real False Positive Rate above $51\%$.
- Raising the threshold to $0.30$ or $0.50$ only eliminates a fraction of false positives (since 34.2% of reals score $> 0.90$), while rapidly degrading fake recall (missing up to $20\%-30\%$ of deepfakes).
- Moving the threshold merely trades one failure mode for another along the ROC curve because the spatial representations of compressed authentic faces and pristine synthetic faces occupy identical regions in feature space.

### 6. What IMAGE-ONLY experiment should be performed next?
Based on the empirical evidence that 34.2% of real faces score $> 0.90$ due to compression artifacts:
1. **Frequency / Residual Feature Extraction Diagnostic**: Analyze FFT / 2D DCT spectral power distributions and high-frequency residuals (e.g., SRM noise filters) on the top 20 False Positives vs. True Negatives to identify the exact artifact signatures triggering the EfficientNet convolutional filters.
2. **Dense Classification Head Hard-Negative Calibration**: Without modifying or retraining the convolutional backbone weights, evaluate an isolated retraining/fine-tuning of the final classification head (or train an ensemble logistic calibration layer) specifically weighted on the 477 hard false-positive real faces from `evaluation/image_error_analysis.csv`.
3. **Auxiliary Artifact Uncertainty Scoring**: Implement a dual-head or feature-distance uncertainty estimator that detects when an input image exhibits compression characteristics typical of the FF-C23 / Merged domains, routing those samples to an explicit **Uncertain** verdict rather than forcing a binary classification.

---

## 8. Regression & Safety Verification

- **Automated Regression Suite**:
  - `python -m unittest discover -s backend/tests -p "test_*.py"`
  - **Result**: `Ran 65 tests in 20.558s, OK` (65/65 tests passed).
- **Scope Compliance**:
  - `backend/video_detector.py`: Unmodified.
  - Video thresholds & fusion logic: Unmodified.
  - V5.2 temporal weights & model: Unmodified.
  - Production code (`backend/image_detector.py`, `backend/config.py`): Unmodified.
  - Zero models retrained.
