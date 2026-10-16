# Dual-Stream Spatial + Forensic Image Model Experiment Report (DeepShield V6)

## Executive Summary

Following the diagnostic findings that deepfake classification errors stem from entangled spatial representations and high-frequency energy imbalances, we designed, implemented, trained, and evaluated a **Dual-Stream Spatial + Forensic Image Architecture (DeepShield V6)**.

The objective was to determine whether feeding explicit high-frequency forensic residuals ($\mathbf{r} = \mathbf{x} - \text{GaussianBlur}(\mathbf{x})$) through a dedicated convolutional stream alongside a frozen EfficientNet-B2 spatial backbone could decouple artifact detection from macro-facial geometry, thereby reducing the high Real $\to$ Fake false positive rate ($48.04\%$ baseline FPR) without degrading genuine deepfake detection.

### Strict Governance & Safety Compliance
- **Video Pipeline Frozen**: `backend/video_detector.py`, V5.2 temporal model, video weights, thresholds, fusion/arbitration, and video tests remained 100% frozen and untouched.
- **Production Baseline Protected**: Zero modifications to `models/deepshield_v5_clean_best.keras`, `backend/image_detector.py`, or `backend/config.py`. Production threshold remains strictly $0.21$.
- **Test Integrity Preserved**: The official FAST10 test set ($3,693$ images) remained **strictly read-only and held out**. No test images were used for training or threshold tuning.
- **Model Isolation**: Saved independently as [`models/v6_dual_stream/deepshield_v6_dual_stream.keras`](file:///c:/Users/ADMIN/Desktop/DeepShield_V5_Final/models/v6_dual_stream/deepshield_v6_dual_stream.keras).
- **Regression Verification**: All **65 / 65 backend unit tests PASS**.

### Definitive Conclusion
**Conclusion: `KEEP EXISTING V5`**

Although the forensic stream successfully diluted single-axis PCA dominance (reducing PC1 explained variance from $90.02\%$ down to $54.49\%$) and increased centroid separation, the dual-stream model suffered from **severe classification head mode collapse**. On the held-out FAST10 test set ($N = 3,693$), test accuracy dropped from **$81.05\%$ to $73.11\%$**, ROC-AUC dropped from **$0.8366$ to $0.6241$** ($-0.2125$), and the Real False Positive Rate escalated from **$48.04\%$ to $100.00\%$** ($993/993$ Real faces misclassified as Fake at $th=0.21$; $98.99\%$ at optimal $th=0.40$). In accordance with the decision rule, the V6 dual-stream model is rejected and the baseline V5 production model remains active.

---

## 1. Architectural Specification

The V6 model fuses two distinct visual pathways:

```
[Spatial Stream]
  Input: (224, 224, 3) RGB
    │
    ▼
  EfficientNet-B2 (Frozen)
    │
    ▼
  GlobalAveragePooling2D ────────┐ (1,408 dims)
                                 │
[Forensic Stream]               ▼
  Input: (224, 224, 3) Residual  Concatenate (1,536 dims)
    │                              │
    ▼                              ▼
  Conv2D(32, 3x3, relu) + BN + MaxPool   Dense(256, relu)
    │                              │
    ▼                              ▼
  Conv2D(64, 3x3, relu) + BN + MaxPool   Dropout(0.4)
    │                              │
    ▼                              ▼
  Conv2D(128, 3x3, relu) + BN + GAP     Dense(1, sigmoid)
    │ (128 dims)
    └─────────────────────────────┘
```

### Parameter Breakdown

| Component | Layer Description | Trainable Status | Parameters |
| :--- | :--- | :---: | :---: |
| **Spatial Stream** | EfficientNet-B2 Backbone | **FROZEN** | 7,768,569 |
| **Forensic Stream** | 3-Stage Conv-BN-Pool Block | **TRAINABLE** | 93,824 |
| **Fusion Head** | Concatenate + Dense(256) + Dropout(0.4) + Dense(1) | **TRAINABLE** | 393,729 |
| **Batch Normalization (Forensic)** | Non-trainable BN running stats | **NON-TRAINABLE** | 448 |
| **TOTALS** | Entire V6 Dual-Stream Network | - | **8,256,442** |
| **Trainable Parameters** | Forensic CNN + Dense Fusion Head | **TRAINABLE** | **487,425 (5.9%)** |
| **Non-Trainable Parameters** | Frozen Backbone + BN parameters | **FROZEN** | **7,769,017 (94.1%)** |

---

## 2. Training Configuration & Trajectory

- **Training Cohort**: $2,000$ balanced images ($1,000$ Real, $1,000$ Fake) sampled proportionally across `FF-C23`, `Deepfake-Videos-Merged`, and `FaceForensics-1000` strictly from `Face_Cropped_FAST10/train`.
- **Validation Cohort**: $400$ balanced images ($200$ Real, $200$ Fake) strictly from `Face_Cropped_FAST10/validation`.
- **Optimization**: Adam with $lr = 1 \times 10^{-4}$, Binary Cross-Entropy loss.
- **Callbacks**: `ModelCheckpoint` (monitoring `val_auc`, max), `EarlyStopping` (patience 4, restore best), `ReduceLROnPlateau` (factor 0.5, patience 2).
- **Execution Time**: $2,670.12\text{ s}$ ($44.50\text{ min}$) on native CPU.
- **Epoch Trajectory**:
  - Epoch 1: `loss = 0.7042`, `val_auc = 0.5000`
  - Epoch 4: `val_auc = 0.5340`
  - Epoch 8: `val_auc = 0.5780`
  - Epoch 12: `loss = 0.6704`, `val_loss = 0.6890`, `val_accuracy = 0.5100`, `val_auc = 0.5976` (Best checkpoint restored).

---

## 3. Official Test Benchmark Comparison ($N = 3,693$)

Evaluated on the exact, untouched official FAST10 direct image test set ($993$ Real, $2,700$ Fake):

| Metric | V5 Baseline (`th=0.21`) | V6 Dual-Stream (`th=0.21`) | Delta | V6 Dual-Stream (Optimal `th=0.40`) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **81.05%** | **73.11%** | **-7.94%** | 73.30% |
| **Precision** | **83.85%** | **73.11%** | **-10.74%** | 73.29% |
| **Recall (Fake)** | 91.74% | **100.00%** | $+8.26\%$ *(collapse)* | 99.89% |
| **F1 Score** | **0.8762** | **0.8447** | **-0.0315** | 0.8455 |
| **ROC-AUC** | **0.8366** | **0.6241** | **-0.2125** | 0.6241 |
| **Real FPR** | **48.04%** ($477/993$) | **100.00%** ($993/993$) | **+51.96%** | 98.99% ($983/993$) |
| **Fake FNR** | 8.26% ($223/2,700$) | **0.00%** ($0/2,700$) | $-8.26\%$ | 0.11% ($3/2,700$) |
| **True Positives (TP)** | 2,477 | 2,700 | +223 | 2,697 |
| **True Negatives (TN)** | **516** | 0 | -516 *(collapsed)* | 10 |
| **False Positives (FP)** | **477** | 993 | +516 | 983 |
| **False Negatives (FN)** | 223 | 0 | -223 | 3 |

---

## 4. Detailed Metric Impact Analysis

### 4.1 False Positive (FP) / Real FPR Analysis
- **Baseline FPs**: $477$ Real images predicted Fake ($48.04\%$ FPR).
- **V6 Dual-Stream FPs**: $993$ Real images predicted Fake ($100.00\%$ FPR at $0.21$; $983$ at optimal $0.40$, $98.99\%$ FPR).
- **Impact**: **Severe Degeneration**. Rather than suppressing false alarms on authentic faces, the dual-stream model pushed all predictions above the decision boundary. Virtually every authentic face in the test set was flagged as a deepfake.

### 4.2 False Negative (FN) / Fake FNR Analysis
- **Baseline FNs**: $223$ Fake images predicted Real ($8.26\%$ FNR).
- **V6 Dual-Stream FNs**: $0$ Fake images predicted Real ($0.00\%$ FNR at $0.21$; $3$ at $0.40$, $0.11\%$ FNR).
- **Impact**: The apparent "perfection" on fake recall is entirely an artifact of majority-class mode collapse, wherein the classification head learned a strong positive bias ($\mu_{\text{score}} > 0.45$).

### 4.3 ROC-AUC & Discriminative Power
- **Baseline ROC-AUC**: **$0.8366$**
- **V6 Dual-Stream ROC-AUC**: **$0.6241$**
- **Impact**: **Net drop of $-0.2125$**. This drop confirms that the model's failure is not simply a threshold calibration issue, but an intrinsic collapse in ranking ability across the feature space.

---

## 5. Feature Separability Analysis ($N = 800$ Diagnostic Cohort)

Extracted from the 256-dimensional fused representation (`fusion_dense`) on the identical 800-image diagnostic cohort (200 TN, 200 FP, 200 TP, 200 FN):

| Separability Metric | V5 Baseline (`GAP-1408`) | V6 Dual-Stream (`Dense-256`) | Assessment |
| :--- | :---: | :---: | :--- |
| **Real vs. Fake Centroid Distance** | $0.0027$ | **$0.1829$** | $+0.1802$ (Substantial geometric expansion) |
| **FP vs. TN Centroid Distance** | $0.0044$ | **$0.1162$** | $+0.1118$ (FP cluster drifted further from TN) |
| **FN vs. TP Centroid Distance** | $0.0069$ | **$0.1493$** | $+0.1424$ |
| **Nearest-Centroid Accuracy** | 52.75% | **57.88%** | $+5.13\%$ (Modest separability gain) |
| **$k$-NN ($k=1$) Accuracy** | **65.12%** | 55.50% | $-9.62\%$ (Local neighborhood structure degraded) |
| **$k$-NN ($k=3$) Accuracy** | **58.50%** | 55.75% | $-2.75\%$ |
| **$k$-NN ($k=5$) Accuracy** | **56.38%** | 55.00% | $-1.38\%$ |
| **Silhouette Score** | $+0.0005$ | **$+0.0136$** | Slight increase, but clusters remain highly overlapping |
| **PCA PC1 Explained Variance** | 90.02% | **54.49%** | **-35.53%** (Diluted dominant macro-facial axis) |

### Interpretation of Feature Diagnostics
The forensic stream did exert a measurable architectural effect:
1. **PCA variance dilution**: By injecting high-frequency residuals, the first principal component's variance dropped from $90.02\%$ to $54.49\%$. The model is no longer operating solely on the single dominant facial-geometry vector.
2. **Cluster dispersion without class separation**: However, the increased centroid distance ($0.0027 \to 0.1829$) did not translate into distinct, separable Real and Fake clusters. Instead, the intra-class variance exploded: $k$-NN accuracy dropped from $65.12\%$ to $55.50\%$, proving that nearest neighbors in the latent space are now more entropic than in V5.

---

## 6. Read-Only Perturbation Robustness Evaluation

Evaluated on a balanced 200-image test subset (100 Real, 100 Fake) under in-memory non-destructive transformations:

| Perturbation Condition | V5 Baseline AUC | V6 Dual-Stream AUC | V5 Accuracy | V6 Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 0.4879 | **0.6265** | 50.0% | 50.0% |
| **JPEG Compression ($Q=50$)** | 0.4890 | **0.6407** | 50.0% | 50.0% |
| **Downsample + Upsample ($0.5\times$)** | 0.4873 | **0.5847** | 50.0% | 50.0% |
| **Mild Gaussian Blur ($3\times 3, \sigma=0.8$)** | 0.4894 | **0.5921** | 50.0% | 50.0% |

*Note: On this diagnostic cohort (composed of extreme false positives and hard negatives), both models predict all samples as positive at $th=0.21$ (giving 50% accuracy on balanced 1:1 data). While V6 exhibits slightly higher ranking stability under JPEG compression on this subset, its overall test performance ($0.6241$ AUC across 3,693 images) remains far inferior to V5's $0.8366$ test AUC.*

---

## 7. Forensic Branch Analysis: Did the High-Frequency Stream Help?

### What Worked
- High-frequency subtraction removed low-frequency identity and illumination signals.
- The 3-stage CNN successfully extracted independent texture representations, preventing EfficientNet-B2's $1,408$-d representation from monopolizing the entire variance profile.

### Why It Failed to Solve the Core Real $\to$ Fake Problem
1. **Real-World High-Frequency Asymmetry**:
   As established in our prior frequency diagnostic ([`docs/image_compression_diagnostic.md`](file:///c:/Users/ADMIN/Desktop/DeepShield_V5_Final/docs/image_compression_diagnostic.md)), authentic Real faces from uncompressed or sharp sources possess **higher** high-frequency energy than heavily compressed fakes. Consequently, a high-frequency residual stream inevitably associates strong edge energy with deepfakes unless paired with synthetic compression augmentation.
2. **Dense Head Gradients Overpowering Residuals**:
   With the spatial backbone frozen and the spatial vector carrying $1,408$ dimensions versus only $128$ dimensions from the forensic branch, the concatenation layer ($1,536$ dims) remained overwhelmingly dominated by spatial features. Binary cross-entropy optimization minimized training error by simply learning a positive bias in the dense classification head.

---

## 8. Final Recommendation

| Evaluated Architecture | Test Accuracy | Test F1 | Test ROC-AUC | Real FPR | Fake FNR | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **V5 Baseline (`clean_best`)** | **81.05%** | **0.8762** | **0.8366** | **48.04%** | **8.26%** | **ACTIVE PRODUCTION** |
| **V5 Fine-Tuned (Blocks 6-7)** | 73.11% | 0.8447 | 0.4697 | 100.00% | 0.00% | REJECTED |
| **V6 Dual-Stream (Spatial + Residual)** | 73.11% | 0.8447 | 0.6241 | 100.00% | 0.00% | **REJECTED** |

**Final Recommendation**:
**`KEEP EXISTING V5`**

Neither CNN backbone fine-tuning nor frozen spatial + forensic residual dual-stream fusion improves upon the V5 baseline. DeepShield V5 (`models/deepshield_v5_clean_best.keras`) remains the superior, protected production image model.
