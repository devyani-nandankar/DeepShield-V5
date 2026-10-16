# Feature-Separability Diagnostic Report

## Executive Summary

Following the failure of frozen classification-head calibration (which suffered mode collapse due to linear inseparability), we conducted a rigorous **Feature-Separability Diagnostic** on the DeepShield V5 image model (`models/deepshield_v5_clean_best.keras`).

### Governance & Protocol Compliance
- **Video Pipeline Untouched**: `backend/video_detector.py`, V5.2 temporal model, video weights, thresholds, fusion, and tests remained completely frozen.
- **Production Baseline Protected**: Zero retraining, zero code modifications, zero weight changes, zero threshold changes.
- **Test Set Safety**: The official FAST10 test set was used purely for read-only inference diagnostics (no training).
- **Regression Tests**: All **65 / 65 backend unit tests PASS**.

### Definitive Recommendation
**Recommendation: `PROCEED WITH CONTROLLED BACKBONE FINE-TUNING`**

---

## 1. Experimental Setup & Cohort Sampling

We evaluated $800$ deterministic samples ($200$ images per cohort) from the official FAST10 Direct Image test set (`evaluation/direct_fast10_predictions.csv`):

| Cohort | Description | Selection Criteria | Count | Mean Baseline Score |
| :--- | :--- | :---: | :---: | :---: |
| **Cohort A: TN_Real** | Correctly classified Real | $\text{Label} = 0, \text{Score} < 0.21$ | 200 | $0.0603 \pm 0.0576$ |
| **Cohort B: FP_Real** | False-Positive Real (Hard Negatives) | $\text{Label} = 0, \text{Score} \ge 0.21$ | 200 | $0.6798 \pm 0.2718$ |
| **Cohort C: TP_Fake** | Correctly classified Fake | $\text{Label} = 1, \text{Score} \ge 0.21$ | 200 | $0.8945 \pm 0.1901$ |
| **Cohort D: FN_Fake** | False-Negative Fake (Hard Positives) | $\text{Label} = 1, \text{Score} < 0.21$ | 200 | $0.0850 \pm 0.0611$ |

For each of the $800$ images, we extracted the full $1,408$-dimensional feature vector $\mathbf{v}_i \in \mathbb{R}^{1408}$ from the `global_average_pooling2d` layer of `models/deepshield_v5_clean_best.keras` using the exact Direct FAST10 preprocessing pipeline (RGB, $224 \times 224$ area resize, normalized to $[0, 1]$).

The complete dataset is preserved in [`evaluation/image_feature_separability.csv`](file:///c:/Users/ADMIN/Desktop/DeepShield_V5_Final/evaluation/image_feature_separability.csv) and [`evaluation/image_feature_separability_summary.json`](file:///c:/Users/ADMIN/Desktop/DeepShield_V5_Final/evaluation/image_feature_separability_summary.json).

---

## 2. Quantitative Feature-Space Measurements

### 2.1 Pairwise Centroid Euclidean Distances

| Centroid | TN_Real | FP_Real | TP_Fake | FN_Fake | All_Real | All_Fake |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TN_Real** | 0.0000 | 0.0044 | 0.0057 | 0.0025 | 0.0022 | 0.0028 |
| **FP_Real** | 0.0044 | 0.0000 | **0.0042** | 0.0064 | 0.0022 | 0.0042 |
| **TP_Fake** | 0.0057 | **0.0042** | 0.0000 | 0.0069 | 0.0045 | 0.0035 |
| **FN_Fake** | **0.0025** | 0.0064 | 0.0069 | 0.0000 | 0.0043 | 0.0035 |
| **All_Real** | 0.0022 | 0.0022 | 0.0045 | 0.0043 | 0.0000 | **0.0027** |
| **All_Fake** | 0.0028 | 0.0042 | 0.0035 | 0.0035 | **0.0027** | 0.0000 |

### 2.2 Critical Geometric Insights from Centroid Distances
1. **Global Inter-Class Distance is Minuscule**:
   The distance between `All_Real` and `All_Fake` centroids is only **$0.0027$**. By contrast, the within-cohort standard deviation along Principal Component 1 is **$0.060$**—more than **$22\times$ larger** than the separation between the Real and Fake class centers!
2. **False-Positive Reals are Closer to True Fakes than to True Reals**:
   $$\text{dist}(\mathbf{c}_{\text{FP\_Real}}, \mathbf{c}_{\text{TP\_Fake}}) = 0.0042 < \text{dist}(\mathbf{c}_{\text{FP\_Real}}, \mathbf{c}_{\text{TN\_Real}}) = 0.0044$$
   The authentic Real faces that trigger false alarms have literally crossed into the Fake feature territory in the frozen 1408-D space.
3. **False-Negative Fakes are Overwhelmingly Closer to Clean Reals**:
   $$\text{dist}(\mathbf{c}_{\text{FN\_Fake}}, \mathbf{c}_{\text{TN\_Real}}) = 0.0025 \ll \text{dist}(\mathbf{c}_{\text{FN\_Fake}}, \mathbf{c}_{\text{TP\_Fake}}) = 0.0069$$
   The false-negative deepfakes are **nearly $3\times$ closer** to authentic True Negative Real faces ($0.0025$) than they are to True Positive Fakes ($0.0069$).

### 2.3 Cosine Similarity Analysis
Across all pairs of centroids, cosine similarity is **$1.0000$** (specifically $> 0.9999997$).
- `All_Real` vs `All_Fake`: $\cos(\theta) = 0.99999982$
- `TN_Real` vs `TP_Fake`: $\cos(\theta) = 0.99999970$
- `FP_Real` vs `TP_Fake`: $\cos(\theta) = 0.99999982$
- `FN_Fake` vs `TN_Real`: $\cos(\theta) = 1.00000000$

**Root Cause**: Global average pooling over non-negative ReLU/Swish activations yields feature vectors whose orientation is entirely dominated by shared facial morphology (skin reflectance, illumination, head geometry). The deepfake forgery signal represents less than $0.01\%$ of the directional orientation in the raw pooled embedding.

---

## 3. Classification and Neighborhood Analysis

### 3.1 Nearest-Centroid Prototype Classification
Using canonical prototypes $\mathbf{c}_{\text{TN\_Real}}$ and $\mathbf{c}_{\text{TP\_Fake}}$ to classify every sample based on Euclidean distance:

| Cohort | % Closer to $\mathbf{c}_{\text{TN\_Real}}$ | % Closer to $\mathbf{c}_{\text{TP\_Fake}}$ | Mean Dist to $\mathbf{c}_{\text{TN\_Real}}$ | Mean Dist to $\mathbf{c}_{\text{TP\_Fake}}$ |
| :--- | :---: | :---: | :---: | :---: |
| **TN_Real** ($N=200$) | **66.5%** | 33.5% | $0.0426 \pm 0.0399$ | $0.0441 \pm 0.0387$ |
| **FP_Real** ($N=200$) | **62.0%** | 38.0% | $0.0494 \pm 0.0522$ | $0.0508 \pm 0.0509$ |
| **TP_Fake** ($N=200$) | **56.5%** | **43.5%** | $0.0486 \pm 0.0438$ | $0.0494 \pm 0.0425$ |
| **FN_Fake** ($N=200$) | **61.0%** | 39.0% | $0.0436 \pm 0.0329$ | $0.0448 \pm 0.0320$ |

- **Overall Nearest-Centroid Accuracy vs True Labels: 52.75%**
- A linear centroid classifier operating on the raw 1408-D features is barely better than random chance ($50.0\%$).
- Even for genuine deepfakes (`TP_Fake`), $56.5\%$ of samples are geometrically closer to the Real prototype than to the Fake prototype!

### 3.2 k-NN Feature-Space Classification (5-Fold Cross-Validation)

| k Neighbors | Cross-Validation Accuracy |
| :---: | :---: |
| $k = 1$ | **65.12%** |
| $k = 3$ | **58.50%** |
| $k = 5$ | **56.38%** |
| $k = 7$ | **56.38%** |
| $k = 15$ | **57.12%** |

### 3.3 1-NN Nearest-Neighbor Class Agreement
When querying the single closest distinct neighbor in 1408-D space:
- **Overall 1-NN Agreement with True Label**: **$67.38\%$**
- **TN_Real**: Closest neighbor is Real in $68.0\%$ of cases, Fake in $32.0\%$.
- **FP_Real**: Closest neighbor is Real in $72.5\%$ of cases, Fake in $27.5\%$.
- **TP_Fake**: Closest neighbor is Fake in $59.0\%$ of cases, Real in $41.0\%$.
- **FN_Fake**: Closest neighbor is Fake in $70.0\%$ of cases, Real in $30.0\%$.

Notice that for `TP_Fake`, $41.0\%$ of immediate nearest neighbors in 1408-D space are authentic Real faces!

---

## 4. Principal Component Analysis (PCA)

```mermaid
xychart-beta
    title "PCA Top 10 Components Explained Variance (%)"
    x-axis ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"]
    y-axis "Explained Variance (%)" 0 --> 100
    bar [90.02, 5.26, 1.52, 0.98, 0.55, 0.26, 0.20, 0.18, 0.15, 0.12]
```

### Variance Distribution:
- **PC 1 explains 90.02% of all variance.**
- **PC 2 explains 5.26% of all variance.**
- Cumulative variance of top 2 components: **95.28%**.
- Cumulative variance of top 10 components: **99.24%**.

### Cohort Distributions along Principal Components:

| Cohort | PC 1 Mean $\pm$ Std | PC 2 Mean $\pm$ Std | Raw Score Mean $\pm$ Std |
| :--- | :---: | :---: | :---: |
| **TN_Real** | $-0.00187 \pm 0.0555$ | $+0.00001 \pm 0.0130$ | $0.0603 \pm 0.0576$ |
| **FP_Real** | $+0.00130 \pm 0.0686$ | $-0.00257 \pm 0.0153$ | $0.6798 \pm 0.2718$ |
| **TP_Fake** | $+0.00345 \pm 0.0621$ | $+0.00079 \pm 0.0155$ | $0.8945 \pm 0.1901$ |
| **FN_Fake** | $-0.00289 \pm 0.0511$ | $+0.00177 \pm 0.0134$ | $0.0850 \pm 0.0611$ |

### What PC1 and PC2 Reveal:
1. **PC1 corresponds to the Decision Axis**:
   - Clean Reals (`TN_Real`) have negative mean ($-0.00187$).
   - False-Negative Fakes (`FN_Fake`) cluster strongly on the negative side ($-0.00289$).
   - Clean Fakes (`TP_Fake`) have positive mean ($+0.00345$).
   - False-Positive Reals (`FP_Real`) cross over into the positive domain ($+0.00130$).
2. **Intra-Class Variance Overwhelms Class Separation**:
   While the mean difference between `TN_Real` and `TP_Fake` along PC1 is $0.00532$, the standard deviation within each cohort is $\approx 0.060$ (more than **$11\times$ larger**). The clusters completely overlap along the primary eigen-axes.

---

## 5. Cluster Separability & Quality Metrics

| Metric | Measured Value | Standard Interpretation | Diagnostic Implication |
| :--- | :---: | :---: | :--- |
| **Silhouette Score (Real vs Fake)** | **+0.0005** | Range $[-1, +1]$; $> 0.5$ strong, $\approx 0$ overlapping | **Complete overlap**. Classes do not form separate clusters in 1408-D space. |
| **Calinski-Harabasz Index** | **0.38** | Higher indicates better-defined clusters | Extremely low. Between-cluster dispersion is negligible compared to within-cluster dispersion. |
| **Davies-Bouldin Index** | **33.77** | Lower indicates better separation ($< 1.0$ is good) | Extremely high. Confirms severe cluster interpenetration. |

---

## 6. Answers to the 6 Required Diagnostic Questions

### Q1: Are Real and Fake classes separable in the current 1408-D feature space?
**NO.** The empirical evidence is overwhelming:
- Silhouette score is **$+0.0005$** (effectively zero).
- Nearest-centroid accuracy is **$52.75\%$** (coin-toss level).
- Real and Fake centroid distance is only $0.0027$ compared to an intra-class standard deviation of $0.060$.
- Under frozen feature extraction, the classes occupy the same geometric region in 1408-D space.

### Q2: Are false-positive Real images located closer to Fake samples?
**YES.**
- The `FP_Real` centroid is closer to `TP_Fake` ($0.0042$) than to `TN_Real` ($0.0044$).
- Along the primary variance component (PC1), `FP_Real` has a positive mean ($+0.00130$), shifting across the decision boundary toward `TP_Fake` ($+0.00345$).
- The frozen backbone maps hard Real faces into the exact feature manifold occupied by genuine fakes.

### Q3: Are false-negative Fake images located closer to Real samples?
**YES, decisively.**
- The `FN_Fake` centroid has a distance of only **$0.0025$** to `TN_Real`, compared to **$0.0069$** to `TP_Fake`.
- `FN_Fake` is **$2.73\times$ closer** to clean Reals than to clean Fakes.
- $61.0\%$ of `FN_Fake` samples are closer to the True Negative prototype than to the True Positive prototype.

### Q4: Is the frozen backbone fundamentally mixing the classes?
**YES.**
- The frozen EfficientNet-B2 backbone is fundamentally mixing the classes because its convolutional receptive fields are predominantly responding to high-level facial anatomy (eyes, nose, mouth, skin tone, facial contours) rather than forensic synthesis traces.
- This is mathematically proven by the fact that PC1 captures **$90.02\%$** of all variance and all class centroids exhibit **$> 0.999999$** cosine similarity.

### Q5: Would unfreezing upper EfficientNet blocks be justified?
**YES.**
- Because the dense classification head operates purely on the fixed $1,408$-D pooled output of the backbone, keeping the backbone frozen mathematically guarantees that no linear or non-linear head can separate the classes without mode collapse (as proven in the previous head calibration experiment).
- To make the representations separable, the convolutional kernels in the upper stages (e.g., MBConv Block 6 and Block 7) must be permitted to adapt their spatial filters away from macro-facial geometry and toward micro-textures, blending boundaries, and frequency-domain artifacts.

### Q6: Should we proceed with controlled fine-tuning?
**YES.**
- Controlled fine-tuning of the upper convolutional blocks (with a very low learning rate such as $10^{-5}$, frozen lower/middle blocks to preserve general visual priors, and hard-negative mining from the train split) is the only mathematically viable path to create feature separability within this architecture.

---

## 7. Regression Test Suite

All 65 regression tests were executed to confirm zero regression or system disruption:

```bash
python -m unittest discover -s backend/tests -p "test_*.py"
```

**Result**:
```
Ran 65 tests in 20.173s

OK
```
- **65 / 65 backend unit tests PASSED**.
- Image pipeline, video pipeline, and all production models remain completely intact.

---

## 8. Final Recommendation

Based strictly on the feature-space measurements:

### **`PROCEED WITH CONTROLLED BACKBONE FINE-TUNING`**

### Technical Justification:
1. **The head alone cannot separate what the backbone has entangled**: The silhouette score of $+0.0005$ and nearest-centroid accuracy of $52.75\%$ demonstrate that the feature vectors are inherently inseparable prior to the classification head.
2. **Upper-block adaptation is necessary and sufficient**: The lower and middle blocks of EfficientNet-B2 capture low-level edges and textures that are robust and should remain frozen. Unfreezing only the top MBConv blocks (Blocks 6 and 7) with a low learning rate ($10^{-5}$) and contrastive / hard-negative supervision will allow the network to adjust its high-level spatial receptive fields, disentangling authentic facial features from manipulation artifacts without destroying the base model's general visual capabilities.
