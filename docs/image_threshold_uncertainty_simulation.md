# DeepShield V5 — Image Threshold & Uncertainty Simulation Report

**Date:** 2026-09-19  
**Target Model:** `models/deepshield_v5_clean_best.keras` (V5 Spatial Image Model)  
**Evaluation Set:** Official FAST10 Test Split (3,693 samples: 993 Real, 2,700 Fake)  
**Evaluation Path:** Direct FAST10 Image Path (primary benchmark path, baseline accuracy: 81.05%)  
**Status:** Simulation & Forensic Analysis Only. Zero modifications to production code, zero modifications to thresholds, zero model retraining.

---

## 1. Executive Summary & Objective

Following the forensic audit of the DeepShield V5 image pipeline (which revealed an aggressive $48.04\%$ False Positive Rate at the hard binary threshold of $0.21$), this simulation investigates how the classification landscape changes across:
1. **Five Alternative Binary Thresholds**: $0.16$, $0.18$, $0.21$, $0.25$, and $0.30$.
2. **Four Selective Uncertainty Bands**:
   - **Configuration A**: Real $< 0.18$ | Uncertain $[0.18, 0.38]$ | Fake $> 0.38$
   - **Configuration B**: Real $< 0.16$ | Uncertain $[0.16, 0.38]$ | Fake $> 0.38$
   - **Configuration C**: Real $< 0.18$ | Uncertain $[0.18, 0.30]$ | Fake $> 0.30$
   - **Configuration D**: Real $< 0.16$ | Uncertain $[0.16, 0.30]$ | Fake $> 0.30$

The goal of this simulation is **not** to pick a preferred configuration, but to quantify the exact mathematical trade-offs between decisive coverage, precision, recall, false positive reduction, and collateral abstention of authentic samples.

---

## 2. Complete Simulation Results Table

Below is the exhaustive metric matrix across all 10 evaluated configurations. The raw machine-readable data is archived in [`evaluation/image_threshold_uncertainty_simulation.csv`](file:///c:/Users/ADMIN/Desktop/DeepShield_V5_Final/evaluation/image_threshold_uncertainty_simulation.csv).

| Configuration | Type | Decisive / Total | Coverage | Decisive Acc | Strict Acc (All) | Precision | Recall (Dec.) | F1 (Dec.) | TP | TN | FP | FN | Real FPR (Dec.) | Fake FNR (Dec.) | Real in Unc | Fake in Unc | Base FP Removed | Base FN Remain | Correct in Unc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Current Binary (0.21)** | Binary | 3,693 / 3,693 | 100.0% | 81.05% | 81.05% | 83.85% | 91.74% | 0.8762 | 2,477 | 516 | 477 | 223 | 48.04% | 8.26% | 0 | 0 | 0 / 477 | 223 | 0 |
| **2. Binary (0.16)** | Binary | 3,693 / 3,693 | 100.0% | 81.02% | 81.02% | 83.02% | 93.07% | 0.8776 | 2,513 | 479 | 514 | 187 | 51.76% | 6.93% | 0 | 0 | 0 / 477 | 187 | 0 |
| **3. Binary (0.18)** | Binary | 3,693 / 3,693 | 100.0% | 81.10% | 81.10% | 83.41% | 92.56% | 0.8775 | 2,499 | 496 | 497 | 201 | 50.05% | 7.44% | 0 | 0 | 0 / 477 | 201 | 0 |
| **4. Binary (0.21 - Repeat)** | Binary | 3,693 / 3,693 | 100.0% | 81.05% | 81.05% | 83.85% | 91.74% | 0.8762 | 2,477 | 516 | 477 | 223 | 48.04% | 8.26% | 0 | 0 | 0 / 477 | 223 | 0 |
| **5. Binary (0.25)** | Binary | 3,693 / 3,693 | 100.0% | 80.91% | 80.91% | 84.43% | 90.59% | 0.8740 | 2,446 | 542 | 451 | 254 | 45.42% | 9.41% | 0 | 0 | 26 / 477 | 223 | 0 |
| **6. Binary (0.30)** | Binary | 3,693 / 3,693 | 100.0% | 80.83% | 80.83% | 85.22% | 89.26% | 0.8719 | 2,410 | 575 | 418 | 290 | 42.09% | 10.74% | 0 | 0 | 59 / 477 | 223 | 0 |
| **A. Uncertainty [0.18, 0.38]** | Uncertainty | 3,414 / 3,693 | 92.45% | 83.16% | 76.88% | 86.23% | 92.10% | 0.8907 | 2,343 | 496 | 374 | 201 | 42.99% | 7.90% | 123 | 156 | 103 / 477 | 201 | 154 |
| **B. Uncertainty [0.16, 0.38]** | Uncertainty | 3,383 / 3,693 | 91.61% | 83.42% | 76.41% | 86.23% | 92.61% | 0.8931 | 2,343 | 479 | 374 | 187 | 43.85% | 7.39% | 140 | 170 | 103 / 477 | 187 | 171 |
| **C. Uncertainty [0.18, 0.30]** | Uncertainty | 3,525 / 3,693 | 95.45% | 82.44% | 78.69% | 85.22% | 92.30% | 0.8862 | 2,410 | 496 | 418 | 201 | 45.73% | 7.70% | 79 | 89 | 59 / 477 | 201 | 87 |
| **D. Uncertainty [0.16, 0.30]** | Uncertainty | 3,494 / 3,693 | 94.61% | 82.68% | 78.23% | 85.22% | 92.80% | 0.8885 | 2,410 | 479 | 418 | 187 | 46.60% | 7.20% | 96 | 103 | 59 / 477 | 187 | 104 |

*Definitions:*
- **Decisive Accuracy**: $(TP + TN) / (TP + TN + FP + FN)$ computed over decisive verdicts only.
- **Strict Accuracy (All)**: $(TP + TN) / Total$, where Uncertain predictions are treated as non-decisive (zero accuracy credit).
- **Real FPR (Decisive)**: $FP / (TN + FP)$, false positive rate among decisive real predictions.
- **Fake FNR (Decisive)**: $FN / (TP + FN)$, false negative rate among decisive fake predictions.
- **Base FP Removed**: Number of the 477 baseline false positives (at 0.21) that are eliminated from the Fake verdict.
- **Correct in Unc**: Number of formerly correct baseline predictions (out of 2,993) that are moved to the Uncertain state.

---

## 3. Analysis of Binary Threshold Trade-Offs

Shifting a single binary threshold without an uncertainty buffer illustrates the zero-sum trade-off between False Positives and False Negatives:

```
Threshold 0.16  <--->  0.18  <--->  0.21 (Current)  <--->  0.25  <--->  0.30
  FPR: 51.76%          50.05%            48.04%            45.42%        42.09%  (Lower is better)
  FNR:  6.93%           7.44%             8.26%             9.41%        10.74%  (Lower is better)
  Prec: 83.02%         83.41%            83.85%            84.43%        85.22%  (Higher is better)
  Rec:  93.07%         92.56%            91.74%            90.59%        89.26%  (Higher is better)
```

### Key Observations:
1. **Lowering Threshold ($0.16$ and $0.18$)**:
   - Reduces False Negatives: Missed deepfakes drop from 223 down to 187 (at 0.16). Recall peaks at $93.07\%$.
   - Aggravates False Positives: Authentic human faces suffer even higher misclassification, reaching $51.76\%$ FPR (514 / 993 real faces flagged Fake).
   - F1 score is nominally maximized ($0.8776$) purely because of the $73\%$ dataset class imbalance favoring synthetic images.
2. **Raising Threshold ($0.25$ and $0.30$)**:
   - Reduces False Positives: Real FPR drops from $48.04\%$ to $42.09\%$ at $0.30$ (59 baseline false positives eliminated). Precision rises to $85.22\%$.
   - Increases False Negatives: Synthetic detection misses increase by $+67$ samples (from 223 up to 290 missed deepfakes at $0.30$). FNR climbs from $8.26\%$ to $10.74\%$.
   - **Conclusion on Hard Binary Thresholds**: Changing the single cut-off point merely slides along the ROC curve without resolving the fundamental problem: over $42\%$ of real faces score above $0.30$, while $10.7\%$ of fake faces score below $0.30$.

---

## 4. Analysis of Uncertainty Band Trade-Offs

Introducing an uncertainty zone acts as a selective abstention filter that absorbs the highest-ambiguity predictions around the decision boundary:

### Comparison of Uncertainty Strategies:

#### Strategy A vs. B (Wide Uncertainty Band: Upper Bound $0.38$):
- **Bandwidth**: 
  - **A $[0.18, 0.38]$**: Width 0.20, Decisive Coverage: **92.45%** (279 uncertain).
  - **B $[0.16, 0.38]$**: Width 0.22, Decisive Coverage: **91.61%** (310 uncertain).
- **Impact on Error Reduction**:
  - Both configurations eliminate **103 out of 477 baseline false positives** ($21.6\%$ reduction in FPs).
  - Decisive accuracy jumps from $81.05\%$ to **83.16% (A)** and **83.42% (B)**.
  - Decisive F1 score improves from $0.8762$ to **0.8907 (A)** and **0.8931 (B)**.
- **Collateral Cost (Authentic Samples Lost)**:
  - In A, 123 real images and 156 fake images become uncertain. 154 previously correct predictions are moved into uncertainty.
  - In B, 140 real images and 170 fake images become uncertain. 171 previously correct predictions are moved into uncertainty.

#### Strategy C vs. D (Moderate Uncertainty Band: Upper Bound $0.30$):
- **Bandwidth**:
  - **C $[0.18, 0.30]$**: Width 0.12, Decisive Coverage: **95.45%** (168 uncertain).
  - **D $[0.16, 0.30]$**: Width 0.14, Decisive Coverage: **94.61%** (199 uncertain).
- **Impact on Error Reduction**:
  - Both configurations eliminate **59 out of 477 baseline false positives** ($12.4\%$ reduction in FPs).
  - Decisive accuracy improves to **82.44% (C)** and **82.68% (D)**.
  - Decisive F1 improves to **0.8862 (C)** and **0.8885 (D)**.
- **Collateral Cost (Authentic Samples Lost)**:
  - Much lower collateral impact: only 79 real images in C (96 in D) become uncertain.
  - Maintains over **94.6%–95.5% decisive coverage**.

#### Lower Threshold Comparison ($0.16$ vs. $0.18$ as Real Boundary):
- Setting the Real boundary to $0.18$ (Configs A and C) preserves more decisive Real classifications (TN = 496 vs 479), at the cost of 14 additional false negatives (FN = 201 vs 187).
- Setting the Real boundary to $0.16$ (Configs B and D) reduces false negatives among decisive classifications to $7.2\%–7.4\%$ (FN = 187), but pushes 17 additional real faces into the uncertain pool.

---

## 5. Comparative Trade-Off Profiles

| Trade-off Objective | Primary Beneficiary Configurations | Measurable Consequence |
| :--- | :--- | :--- |
| **Maximum System Coverage (100% Decisive)** | Binary 0.18, 0.21, 0.25 | Forces a hard verdict on every sample, maintaining $100\%$ throughput, but accepts $42\% - 50\%$ false positive rates on real faces. |
| **Maximum Decisive Accuracy & F1** | Uncertainty B $[0.16, 0.38]$ & A $[0.18, 0.38]$ | Reaches **83.42% decisive accuracy** and **0.8931 F1**, eliminating 103 false positives, but reduces decisive coverage to $91.6\% - 92.5\%$. |
| **Balanced Coverage with Error Reduction** | Uncertainty C $[0.18, 0.30]$ | Retains **95.45% coverage** while improving decisive accuracy to **82.44%** and precision to **85.22%**, with minimal collateral abstention (only 87 correct samples moved to uncertain). |
| **Deepfake Catch Rate (Minimum FNs)** | Binary 0.16, Uncertainty B & D | Caps missed synthetic faces at 187 (FNR: $6.9\% - 7.4\%$), but requires a lower real decision ceiling ($0.16$). |

---

## 6. Regression & Scope Verification

- **Automated Regression Suite**:
  - Command: `python -m unittest discover -s backend/tests -p "test_*.py"`
  - Result: `Ran 65 tests in 34.255s, OK` (100% pass rate).
- **Scope Confirmation**:
  - `backend/video_detector.py`: Unmodified.
  - Video thresholds & margin (`VIDEO_TEMPORAL_THRESHOLD = 0.30`, `VIDEO_TEMPORAL_MARGIN = 0.03`): Unmodified.
  - Video fusion logic & V5.2 weights: Unmodified.
  - Production image code (`backend/image_detector.py`): Unmodified.
  - Threshold configuration in `backend/config.py`: Unmodified (`CALIBRATED_THRESHOLD = 0.21`).
  - Zero models retrained.
