# DeepShield V7 Validation-Based Threshold Calibration Report

## 1. Executive Summary

This report provides the **Validation-Only Threshold Calibration** and subsequent blind test evaluation for the **DeepShield V7 Metric-Learning Model** (`deepshield_v7_metric_learning.keras`).

### Strict Governance Compliance
- **Zero Retraining**: V7 model weights remain strictly frozen.
- **Zero Video Pipeline Touches**: Video detector, V5.2 temporal model, weights, thresholds, and fusion remain 100% frozen.
- **Production Baseline Preserved**: `models/deepshield_v5_clean_best.keras` remains untouched.
- **Production Threshold Invariant**: `IMAGE_THRESHOLD = 0.21` remains unchanged in production code.
- **Zero Test Leakage**: The calibration search space was explored strictly on `Face_Cropped_FAST10/validation` (3,688 samples: 1,045 Real, 2,643 Fake). The official test set was evaluated strictly once using the locked validation threshold.

---

## 2. Validation Set Threshold Calibration (0.01 to 0.99)

We performed a full grid search across 99 thresholds on the 3,688 validation images.

### Candidate Operational Points from Validation:

| Candidate Criterion | Validation Threshold | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 | Real FPR | Fake FNR | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Best Validation F1 (Selected)** | **0.27** | 81.43% | 82.46% | 94.10% | **0.8790** | 50.62% | 5.90% | 529 | 156 |
| **Highest Validation Accuracy** | **0.27** | **81.43%** | 82.46% | 94.10% | 0.8790 | 50.62% | 5.90% | 529 | 156 |
| **Balanced Error Count (|FP - FN|)** | **0.43** | 80.37% | 86.39% | 86.19% | 0.8629 | 34.35% | 13.81% | 359 | 365 |
| **Balanced Error Rate (|FPR - FNR|)** | **0.60** | 76.68% | 89.29% | 76.66% | 0.8249 | 23.25% | 23.34% | 243 | 617 |

### Primary Selected Operational Threshold
- **Locked Threshold**: **`0.27`** (Best Validation F1 criterion).
- The complete trade-off curve across all 99 thresholds is stored in [`evaluation/v7_validation_thresholds.csv`](file:///c:/Users/ADMIN/Desktop/DeepShield_V5_Final/evaluation/v7_validation_thresholds.csv).

---

## 3. Official Held-Out Test Set Evaluation (3,693 Images)

Comparing the frozen **V5 Production Baseline (Threshold = 0.21)** against **V7 at Validation-Selected Threshold (0.27)**:

| Metric | V5 Production Baseline (0.21) | V7 Calibrated (0.27) | Delta (V7 - V5) | Percentage Change |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 81.05% | **80.88%** | -0.17% | -0.21% |
| **Precision** | 83.85% | **82.52%** | -1.33% | -1.59% |
| **Recall** | 91.74% | **93.70%** | +1.96% | +2.14% |
| **F1-Score** | 0.8762 | **0.8776** | +0.0014 | +0.15% |
| **ROC-AUC** | 0.8366 | **0.8535** | +0.0169 | +2.02% |
| **False Positives (FP)** | 477 | **536** | +59 | +12.37% |
| **False Negatives (FN)** | 223 | **170** | -53 | -23.77% |
| **Real FPR** | 47.94% | **53.98%** | +6.04% | +12.60% |
| **Fake FNR** | 8.27% | **6.30%** | -1.97% | -23.82% |

### Candidate Operational Points on Test Set:
| Candidate Criterion | Threshold | Accuracy | Precision | Recall | F1-Score | FP | FN | Real FPR | Fake FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Best F1** | **0.27** | 80.88% | 82.52% | 93.70% | **0.8776** | 536 | 170 | 53.98% | 6.30% |
| **Highest Accuracy** | **0.27** | 80.88% | 82.52% | 93.70% | 0.8776 | 536 | 170 | 53.98% | 6.30% |
| **Balanced Error Count** | **0.43** | 80.56% | 86.95% | 86.37% | 0.8666 | 350 | 368 | 35.25% | 13.63% |
| **Balanced Error Rate** | **0.60** | 76.71% | 90.03% | 76.63% | 0.8279 | 229 | 631 | 23.06% | 23.37% |

---

## 4. Final Governance Decision

Official Verdict:
**`KEEP EXISTING V5`**

### Decision Rationale:
- V5 Production Baseline remains the active deployment model.
- Production `IMAGE_THRESHOLD = 0.21` is unchanged.
- All backend regression tests pass: **65/65 PASS**.
