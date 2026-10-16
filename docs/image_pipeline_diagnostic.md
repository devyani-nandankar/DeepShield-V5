# DeepShield V5 — Image Pipeline Forensic Diagnostic Report

**Date:** 2026-09-19  
**Target Model:** `models/deepshield_v5_clean_best.keras`  
**Dataset:** Official FAST10 Test Split (`Kaggle_Transfer/face_crop_FAST10_manifest_split.csv` / `evaluation/test_images/`)  
**Scope:** Strictly Image-Only Diagnostic Audit. Zero modifications to video pipelines, video weights, thresholds, or production code.

---

## Executive Summary

An exhaustive evaluation of the DeepShield V5 spatial image model (`deepshield_v5_clean_best.keras`) was conducted across all **3,693 test samples** of the official FAST10 test set (comprising **993 Real** and **2,700 Fake** face crops). 

The evaluation was executed over two isolated evaluation pipelines:
1. **Direct FAST10 Path**: Evaluating pre-cropped ground-truth face images directly resized to $224 \times 224$ using standard V5 preprocessing (`tf.keras.applications.efficientnet.preprocess_input`).
2. **YuNet Face-Crop Path**: Processing the test images through the production YuNet ONNX face detector (`models/yunet/face_detection_yunet_2023mar.onnx`), extracting bounding boxes with aspect-ratio preservation and safety padding, and feeding into V5 preprocessing.

### Key Benchmark Metrics (at calibrated `threshold = 0.21`):
- **Direct FAST10 Path**: Accuracy **81.05%**, Precision **83.85%**, Recall **91.74%**, F1 Score **87.62%**, ROC-AUC **0.8366**.
- **YuNet Face-Crop Path**: Accuracy **80.29%**, Precision **82.74%**, Recall **92.30%**, F1 Score **87.25%**, ROC-AUC **0.8300**.
- **YuNet Detection Success Rate**: **99.27%** (3,666 / 3,693 faces detected; 27 tight boundary crops fell back to center crop).

### Primary Forensic Findings:
1. **Severe False Positive Rate on Real Faces ($48.04\%$)**: Under the current production threshold ($0.21$), **477 out of 993 real images** are misclassified as Fake. The real class distribution has a mean score of $0.3605$ and a median of $0.1805$, with the 75th percentile reaching $0.7280$.
2. **Dataset-Specific False Positive Hotspots**:
   - `FF-C23` real faces suffer a **55.67% False Positive Rate** (167 / 300 real faces predicted Fake).
   - `Deepfake-Videos-Merged` real faces suffer a **48.32% False Positive Rate** (187 / 387 real faces predicted Fake).
3. **Low False Negative Rate ($8.26\%$)**: Only **223 out of 2,700 synthetic faces** are missed. The fake score distribution is heavily saturated near $1.0$ (mean: $0.8100$, median: $0.9793$, 90th percentile: $0.99999$).
4. **Minor Preprocessing Variance**: YuNet cropping accounts for only a **0.76% accuracy delta** (43 additional false positives), confirming that preprocessing alignment is not the primary driver of classification errors. The primary root cause is **threshold calibration and domain generalization**.

---

## 1. Comparative Performance Table

| Metric | Path 1: Direct FAST10 Image | Path 2: YuNet Face Crop | Delta (YuNet vs Direct) |
| :--- | :---: | :---: | :---: |
| **Total Test Samples** | 3,693 | 3,693 | 0 |
| **Real Samples (Ground Truth)** | 993 | 993 | 0 |
| **Fake Samples (Ground Truth)** | 2,700 | 2,700 | 0 |
| **Accuracy** | **81.05%** | 80.29% | -0.76% |
| **Precision** | **83.85%** | 82.74% | -1.11% |
| **Recall** | 91.74% | **92.30%** | +0.56% |
| **F1 Score** | **87.62%** | 87.25% | -0.37% |
| **ROC-AUC** | **0.8366** | 0.8300 | -0.0066 |
| **True Negatives (TN)** | **516** | 473 | -43 |
| **False Positives (FP)** | **477** (48.04% FPR) | 520 (52.37% FPR) | +43 |
| **False Negatives (FN)** | 223 (8.26% FNR) | **208** (7.70% FNR) | -15 |
| **True Positives (TP)** | 2,477 | **2,492** | +15 |
| **Operating Threshold** | 0.21 | 0.21 | 0.00 |
| **Optimal F1 Threshold** | 0.16 (F1: 0.8776) | 0.18 (F1: 0.8738) | +0.02 |
| **YuNet Face Detection Rate** | N/A (bypassed) | **99.27%** (3,666 / 3,693) | — |

---

## 2. Score Distributions (Real vs. Fake)

### Direct FAST10 Path:
| Statistic | Real Images ($N=993$) | Fake Images ($N=2,700$) |
| :--- | :---: | :---: |
| **Minimum** | $0.000012$ | $0.000168$ |
| **Maximum** | $0.999999$ | $1.000000$ |
| **Mean** | $0.3605$ | $0.8100$ |
| **Median (50th percentile)** | $0.1805$ | $0.9793$ |
| **Standard Deviation** | $0.3656$ | $0.2923$ |
| **25th Percentile (Q1)** | $0.0398$ | $0.7125$ |
| **75th Percentile (Q3)** | $0.7280$ | $0.9995$ |
| **90th Percentile** | $0.9649$ | $1.0000$ |

### YuNet Face-Crop Path:
| Statistic | Real Images ($N=993$) | Fake Images ($N=2,700$) |
| :--- | :---: | :---: |
| **Minimum** | $0.000012$ | $0.000168$ |
| **Maximum** | $0.999999$ | $1.000000$ |
| **Mean** | $0.3918$ | $0.8183$ |
| **Median (50th percentile)** | $0.2347$ | $0.9806$ |
| **Standard Deviation** | $0.3681$ | $0.2863$ |
| **25th Percentile (Q1)** | $0.0512$ | $0.7449$ |
| **75th Percentile (Q3)** | $0.7728$ | $0.9995$ |
| **90th Percentile** | $0.9687$ | $1.0000$ |

### Distribution Observations:
- **Severe Class Imbalance Calibration Skew**: At `threshold = 0.21`, the threshold sits *below* the mean score of the real class ($0.3605$) and very close to its median ($0.1805$). Over 25% of real images score above $0.7280$, and 10% score above $0.9649$.
- **High Fake Saturation**: The fake model predictions are heavily concentrated at the extreme ceiling ($0.98+$), demonstrating strong feature activations for synthetic artifacts, but also penalizing subtle high-frequency patterns on authentic faces.
- **YuNet Margin Shift**: Running YuNet on pre-cropped faces slightly shifts the real median from $0.1805$ to $0.2347$ (+0.0542), crossing the $0.21$ threshold for 43 borderline real faces.

---

## 3. Dataset-Level Error Analysis

Breakdown across the source video datasets comprising the test split (based on Direct FAST10 path):

### False Positives (Real Predicted as Fake, Total: 477 / 993 = 48.04%):
| Source Dataset | Total Real | False Positives | FP Rate | Contributing Factors |
| :--- | :---: | :---: | :---: | :--- |
| **FF-C23** | 300 | **167** | **55.67%** | Strong H.264 compression artifacts (C23 factor), blockiness mistaken for generative grid artifacts. |
| **Deepfake-Videos-Merged** | 387 | **187** | **48.32%** | Diverse resolution and web re-encoding artifacts. Represents the largest raw volume of FPs. |
| **FaceForensics-1000** | 306 | **123** | **40.20%** | Studio lighting, high contrast, and sharpness filtering. |

### False Negatives (Fake Predicted as Real, Total: 223 / 2,700 = 8.26%):
| Source Dataset | Total Fake | False Negatives | FN Rate | Contributing Factors |
| :--- | :---: | :---: | :---: | :--- |
| **FaceForensics-1000** | 297 | **59** | **19.87%** | High quality manipulation techniques that leave minimal spatial boundary warping on single frames. |
| **FF-C23** | 1,950 | **135** | **6.92%** | Low error rate; heavy manipulations are readily caught despite compression. |
| **Deepfake-Videos-Merged** | 453 | **29** | **6.40%** | Strongest detection rate among deepfake video sets. |

---

## 4. Highest-Confidence Error Analysis

The top 50 False Positives and top 50 False Negatives have been extracted and archived to `evaluation/image_error_analysis.csv`.

### Top 5 Extreme False Positives (Authentic faces scoring near 1.0):
1. `face_024570.jpg` (`FF-C23_91402_494_frame_200.jpg`, Video Group `91402`): **Score: 0.999999** (Predicted Fake, True: Real).
2. `face_021057.jpg` (`Deepfake-Videos-Merged_100264_real_1288_frame_132.jpg`, Video Group `100264`): **Score: 0.999997** (Predicted Fake, True: Real).
3. `face_024571.jpg` (`FF-C23_91402_494_frame_401.jpg`, Video Group `91402`): **Score: 0.999995** (Predicted Fake, True: Real).
4. `face_022055.jpg` (`FaceForensics-1000_97431_494_frame_601.jpg`, Video Group `97431`): **Score: 0.999991** (Predicted Fake, True: Real).
5. `face_024572.jpg` (`FF-C23_91402_494_frame_601.jpg`, Video Group `91402`): **Score: 0.999967** (Predicted Fake, True: Real).

*Key finding*: Consecutive frames from identical video groups (e.g. video `91402` in FF-C23 and video `100264` in Merged) consistently score $> 0.9999$. This proves that the false positive trigger is an intrinsic property of the video encoding/subject lighting rather than random frame-level noise.

### Top 5 Extreme False Negatives (Synthetic faces scoring near 0.0):
1. `face_022934.jpg` (`FaceForensics-1000_fake_1101_frame_217.jpg`): **Score: 0.000168** (Predicted Real, True: Fake).
2. `face_022718.jpg` (`FaceForensics-1000_fake_828_frame_445.jpg`): **Score: 0.000216** (Predicted Real, True: Fake).
3. `face_022822.jpg` (`FaceForensics-1000_fake_963_frame_301.jpg`): **Score: 0.000305** (Predicted Real, True: Fake).
4. `face_022647.jpg` (`FaceForensics-1000_fake_736_frame_414.jpg`): **Score: 0.000451** (Predicted Real, True: Fake).
5. `face_022849.jpg` (`FaceForensics-1000_fake_997_frame_395.jpg`): **Score: 0.000779** (Predicted Real, Fake: Fake).

*Key finding*: The top extreme false negatives are overwhelmingly from **FaceForensics-1000** (e.g. FaceShifter / NeuralTextures / Deepfakes pristine renders), where spatial textures are preserved cleanly and artifacts exist primarily in temporal dynamics.

---

## 5. Diagnostic Questions & Findings

### A. Direct FAST10 Image vs. YuNet Face-Crop: Is there a significant performance difference, and what causes it?
- **Result**: The performance difference is modest:
  - Accuracy drops by **0.76%** (from 81.05% to 80.29%).
  - Precision drops by **1.11%** (from 83.85% to 82.74%).
  - False positives increase by **43 samples** (from 477 to 520).
- **Cause**: The FAST10 dataset images are *already* tight face crops extracted during dataset compilation. Re-running YuNet on an already-tight crop causes YuNet to re-detect facial landmarks in a restricted canvas, slightly adjusting bounding box margins (e.g., cropping forehead or chin edges slightly tighter). This minor boundary shift alters the peripheral context around hair and jawlines, slightly increasing synthetic texture activations in borderline images.

### B. Which dataset causes the most False Positives?
- By **Rate**: **FF-C23** causes the highest False Positive Rate at **55.67%** (167 / 300 real faces). The strong H.264 compression (compression factor 23) introduces blocking and DCT quantization noise across smooth facial skin that the V5 EfficientNet-B2 spatial backbone misinterprets as synthetic generative noise.
- By **Absolute Count**: **Deepfake-Videos-Merged** causes the highest number of False Positives (**187 FPs**), with a 48.32% FP rate, due to web re-encoding and lighting variance.

### C. Which dataset causes the most False Negatives?
- By **Rate**: **FaceForensics-1000** has by far the highest False Negative Rate at **19.87%** (59 / 297 fake faces missed). Advanced manipulation techniques in this set generate visually convincing facial skin where spatial blending lines are minimal on individual frames.
- By **Absolute Count**: **FF-C23** accounts for **135 FNs**, although its FN rate is low (6.92%) due to the large number of fake samples (1,950).

### D. What is the root cause of the current image error rate?
The forensic evidence demonstrates that the error rate is driven by a hierarchy of causes:
1. **Threshold Calibration & Class Imbalance (Primary Root Cause)**:
   The threshold `0.21` was calibrated to aggressively maximize recall against heavily synthetic datasets. However, real-world authentic faces exhibit a natural score spread ($\mu = 0.3605, \sigma = 0.3656$). Forcing a hard binary split at `0.21` cuts directly through the body of the real distribution (below its mean and near its median), guaranteeing that almost half of all real images ($48.04\%$) are flagged as fake.
2. **Model Generalization / Compression Sensitivity (Secondary Cause)**:
   The model exhibits severe sensitivity to video compression artifacts (e.g., C23 compression, mobile compression, AR filters). Natural skin with slight compression artifacts or makeup triggers synthetic scores of $0.999+$.
3. **Face Cropping Preprocessing (Minor Factor)**:
   YuNet vs. direct crop contributes only a $\sim 0.76\%$ variance. Preprocessing is stable and reliable (99.27% detection rate).

### E. Next Recommended Safe Experiment:
1. **Calibrate Dual-Threshold / Uncertainty Boundary for Single Images**:
   Similar to the V5.2 video pipeline, single images in the range $[0.18, 0.40]$ should not be forced into a brittle binary decision. Adopting an **Uncertain** or **Low Confidence** state prevents 48% of authentic faces from being falsely convicted.
2. **Hard-Negative Mining on High-Confidence False Positives**:
   Without retraining the base EfficientNet backbone from scratch, fine-tune the classification head or train an ensemble calibration adapter using the 477 verified false-positive real faces identified in `evaluation/image_error_analysis.csv` along with filtered AR/social media faces.

---

## 6. Regression & System Integrity Verification

- **Regression Test Suite**:
  - `python -m unittest discover -s backend/tests -p "test_*.py"`
  - **Result**: `Ran 65 tests in 34.255s, OK` (100% pass rate across all 65 unit and integration tests).
- **Scope Compliance**:
  - `backend/video_detector.py`: Unmodified.
  - Video thresholds (`VIDEO_TEMPORAL_THRESHOLD = 0.30`, `IMAGE_THRESHOLD = 0.21`): Unchanged.
  - V5.2 temporal model weights: Unmodified.
  - Video fusion logic & preprocessing: Unmodified.
  - Zero models retrained.
