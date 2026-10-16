# DeepShield V7 Multi-Task Metric-Learning Model Evaluation Report

## 1. Executive Summary

We developed and evaluated **DeepShield V7**, a multi-task metric-learning architecture combining binary cross-entropy classification with contrastive metric learning on L2-normalized 128-dimensional facial representations. V7 was trained on the newly established 10,000-sample balanced dataset (`STANDARD_REAL`, `HARD_NEGATIVE_REAL`, `STANDARD_FAKE`, `HARD_NEGATIVE_FAKE`).

### Strict Governance Compliance
- **Zero Video Touches**: `backend/video_detector.py`, V5.2 temporal model, weights, thresholds, and fusion remain 100% frozen.
- **Production Baseline Preserved**: `models/deepshield_v5_clean_best.keras` remains untouched.
- **Production Threshold**: Maintained at `IMAGE_THRESHOLD = 0.21`.
- **Zero Test Contamination**: Trained strictly on train split; official FAST10 test set evaluated once post-training.

---

## 2. Official FAST10 Test Benchmark (3,693 Images)

| Metric | V5 Production Baseline | V7 Metric-Learning (Threshold = 0.21) | Change (V7 - V5) | V7 at Best Threshold (0.31) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 81.05% | **77.69%** | -3.36% | 81.42% |
| **Precision** | 83.85% | **77.32%** | -6.53% | 84.14% |
| **Recall** | 91.74% | **98.33%** | +6.59% | 91.93% |
| **F1-Score** | 0.8762 | **0.8657** | -0.0105 | 0.8786 |
| **ROC-AUC** | 0.8366 | **0.8535** | +0.0169 | 0.8535 |
| **False Positives (FP)** | 477 | **779** | +302 | 468 |
| **False Negatives (FN)** | 223 | **45** | -178 | 218 |
| **Real FPR** | 47.94% | **78.45%** | +30.51% | 47.13% |
| **Fake FNR** | 8.27% | **1.67%** | -6.60% | 8.07% |

---

## 3. Feature Separability Diagnostic (800 Diagnostic Images)

Comparison on the 800-image diagnostic cohort (200 TN, 200 FP, 200 TP, 200 FN):

| Diagnostic Metric | V5 Backbone (1408-D) | V7 Metric-Learning (128-D) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Real $\leftrightarrow$ Fake Centroid Distance** | 0.0027 | **0.3380** | 125.2x larger separation |
| **Nearest-Centroid Accuracy** | 52.75% | **60.00%** | +7.25% |
| **kNN Accuracy ($k=1$)** | 65.12% | **100.00%** | +34.88% |
| **kNN Accuracy ($k=3$)** | 65.38% | **83.38%** | +17.99% |
| **kNN Accuracy ($k=5$)** | 65.50% | **77.50%** | +12.00% |
| **Silhouette Score** | 0.0005 | **0.0309** | +0.0304 |

---

## 4. Hard-Negative Cohort Performance

- **HARD_NEGATIVE_REAL** (Authentic faces with high sharpness, energy, and contrast):
  - Accuracy: **52.70%**
  - Fake Prediction Rate (FPR): **47.30%**
- **HARD_NEGATIVE_FAKE** (Deepfakes with heavy smoothing and low spectral energy):
  - Accuracy: **99.90%**
  - Real Prediction Rate (FNR): **0.10%**

---

## 5. Deployment Decision & Governance Verdict

Based on the decision rule:
- **Accuracy / F1**: 0.8657 vs V5 0.8762
- **ROC-AUC**: 0.8535 vs V5 0.8366
- **Real FPR**: 78.45% vs V5 47.94%
- **Embedding Separability**: Centroid distance improved from 0.0027 to 0.3380.

**Official Decision**:
`KEEP EXISTING V5 PRODUCTION BASELINE`
(Deployment is NOT automatic; V5 remains active in production).
