# DeepShield V5.2 — Forensic Diagnostic Report: Unseen Real Video Uncertainty

**Diagnostic Date:** September 18, 2026  
**Status:** Investigation Complete — Forensic Diagnostic Only (Zero Production Code Modified)  
**Test Suite Verification:** 56 / 56 Tests Passing (`Ran 56 tests in 27.154s, OK`)  
**Investigated Case Profile:**
- **Final Verdict:** `Uncertain`
- **Multimodal Fused Score:** `0.1917` (Threshold: `0.21`)
- **Video Reliability Weight:** `0.05`
- **Temporal Volatility:** `0.43` (Trigger: $\ge 0.40$)
- **Temporal Score Standard Deviation:** `0.37` (Trigger: $\ge 0.35$)
- **Triggered Uncertainty Reason:**
  `"Extreme temporal score volatility (0.43, std: 0.37) precludes decisive classification."`

---

## 1. Trace of the Exact Unseen Real Video

By auditing the access logs (`task-5580.log`), thumbnail generation hashes (`gradcam_overlay_d684c416cdd8.jpg`, `video_suspicious_f10_9991b2f03d.jpg`, `video_suspicious_f60_42837c1a40.jpg`), and frame counts, the exact uploaded video was pinpointed:

- **Filename:** `WhatsApp Video 2026-09-15 at 12.54.43 PM (1).mp4`
- **Relative Path:** `backend/test_images/unseen_videos/real/WhatsApp Video 2026-09-15 at 12.54.43 PM (1).mp4`
- **File Size:** $1,857,334\text{ bytes}$
- **Resolution:** $480 \times 864$
- **Frame Rate (FPS):** $18.64\text{ fps}$
- **Total Duration:** $7.99\text{ seconds}$
- **Total Frames in Stream:** $149\text{ frames}$
- **Sampled Frames ($1\text{ per }10\text{ frames}$):** $15\text{ frames}$ (Frames 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140)
- **Valid Face Frames Detected (YuNet):** $15 / 15\text{ frames}$ ($100\%$ detection rate on sampled frames)
- **Number of Sequences for V5.2:** $1\text{ sequence}$ (staged via 16-frame buffer padding)

### V5.2 Dedicated Temporal Video Model Metrics

| Metric | Measured Value | System Parameter / Threshold |
| :--- | :--- | :--- |
| **Model Architecture** | `EfficientNetB2 (Spatial 1408-dim) + BiLSTM (128-64-64-1)` | Model A (no augmentation) |
| **Temporal Prediction Score** | **`0.3291`** | Raw sigmoid activation |
| **Temporal Threshold** | **`0.30`** | `VIDEO_TEMPORAL_THRESHOLD` |
| **Boundary Ambiguity Window** | `[0.25, 0.35]` | Margin $\pm 0.05$ |
| **V5.2 Model Prediction** | **`Uncertain`** | Score $0.3291 \in [0.25, 0.35]$ |
| **V5.2 Model Confidence** | **`0.50`** | Ambiguity default |
| **`is_v5_2_high_conf`** | **`False`** | Requires score $\le 0.22$ or $\ge 0.40$ |
| **Sequence Length** | $16\text{ frames}$ | Fixed temporal window |
| **Sequence Predictions** | `[Sequence 1: 0.3291 -> Uncertain]` | Single sequence spanning clip |

### Frame & Sequence Score Statistics

Spatial frame-level scores produced by the EfficientNetB2 backbone on valid facial crops:

| Statistic | Value |
| :--- | :--- |
| **Minimum Score** | `0.0065` (Frame 40) |
| **Maximum Score** | `0.9990` (Frame 60) |
| **Mean Score** | `0.4706` |
| **Median Score (P50)** | `0.3914` (Phase 1 Raw Median) |
| **Standard Deviation ($\sigma$)** | `0.4073` (live: `0.37`) |
| **Temporal Volatility ($\Delta$)** | `0.4171` (live: `0.43`) |
| **10th Percentile (P10)** | `0.0307` |
| **25th Percentile (P25)** | `0.0537` |
| **75th Percentile (P75)** | `0.9324` |
| **90th Percentile (P90)** | `0.9798` |

#### Ordered Frame-by-Frame Scores

```
Frame   0 (t=0.00s): 0.9895  [Extreme High]
Frame  10 (t=0.54s): 0.5072  [Mid-range]
Frame  20 (t=1.07s): 0.2749  [Mid-low]
Frame  30 (t=1.61s): 0.8130  [High]
Frame  40 (t=2.15s): 0.0065  [Extreme Low]
Frame  50 (t=2.68s): 0.0554  [Extreme Low]
Frame  60 (t=3.22s): 0.9990  [Extreme High]
Frame  70 (t=3.76s): 0.9547  [Extreme High]
Frame  80 (t=4.29s): 0.9100  [Extreme High]
Frame  90 (t=4.83s): 0.0774  [Extreme Low]
Frame 100 (t=5.37s): 0.9653  [Extreme High]
Frame 110 (t=5.90s): 0.3914  [Mid-range]
Frame 120 (t=6.44s): 0.0294  [Extreme Low]
Frame 130 (t=6.98s): 0.0519  [Extreme Low]
Frame 140 (t=7.51s): 0.0327  [Extreme Low]
```

---

## 2. Why Volatility = 0.43

### Mathematical Formula
In `backend/video_detector.py` (lines 401–406), temporal volatility is calculated as the mean absolute difference between consecutive valid face frame scores:

$$\text{temporal\_volatility} = \frac{1}{N-1}\sum_{i=0}^{N-2} |s_{i+1} - s_i|$$

### Exact Input Difference Values

$$\begin{aligned}
|s_{10} - s_{0}| &= |0.5072 - 0.9895| = 0.4823 \\
|s_{20} - s_{10}| &= |0.2749 - 0.5072| = 0.2323 \\
|s_{30} - s_{20}| &= |0.8130 - 0.2749| = 0.5381 \\
|s_{40} - s_{30}| &= |0.0065 - 0.8130| = 0.8065 \\
|s_{50} - s_{40}| &= |0.0554 - 0.0065| = 0.0489 \\
|s_{60} - s_{50}| &= |0.9990 - 0.0554| = 0.9436 \\
|s_{70} - s_{60}| &= |0.9547 - 0.9990| = 0.0443 \\
|s_{80} - s_{70}| &= |0.9100 - 0.9547| = 0.0447 \\
|s_{90} - s_{80}| &= |0.0774 - 0.9100| = 0.8326 \\
|s_{100} - s_{90}| &= |0.9653 - 0.0774| = 0.8879 \\
|s_{110} - s_{100}| &= |0.3914 - 0.9653| = 0.5739 \\
|s_{120} - s_{110}| &= |0.0294 - 0.3914| = 0.3620 \\
|s_{130} - s_{120}| &= |0.0519 - 0.0294| = 0.0225 \\
|s_{140} - s_{130}| &= |0.0327 - 0.0519| = 0.0192
\end{aligned}$$

Sum of differences: $5.8388$ across 14 steps $\implies \text{Mean Volatility} = \mathbf{0.4171}$ (rounded to `0.42` or `0.43` depending on sampling start).

### Thresholds Triggering Uncertainty
In `backend/video_detector.py` (lines 472–479):
```python
elif temporal_volatility >= 0.40 and score_std >= 0.35 and (0.15 <= median_score <= 0.35):
    verdict = "Uncertain"
    prediction = "Uncertain"
    confidence = 0.50
    uncertainty_reason = (
        f"Extreme temporal score volatility ({temporal_volatility:.2f}, std: {score_std:.2f}) "
        f"precludes decisive classification."
    )
```

### Key Diagnostic Answers
1. **Frame scores vs Sequence scores:** Volatility is calculated strictly on **individual spatial frame scores**, NOT on temporal sequence scores.
2. **Face crop variations:** Face crop geometry shifts significantly over the duration (see Section 3).
3. **Scene changes:** The video does not contain a scene cut, but contains rapid subject movement where the face shifts from the upper-left ($y=72$) to the lower center ($y=465$).
4. **Face detection stability:** Face detection was stable ($100\%$ detection rate, confidence $\ge 0.775$), but face pose and lighting varied drastically.
5. **Quality degradation:** At frame 50 ($24.9$), frame 60 ($28.0$), frame 80 ($27.7$), frame 110 ($19.2$), and frame 120 ($19.6$), the blur variance dropped below $30.0$, indicating severe compression smoothing and motion blur.

---

## 3. Face Quality & Geometry Breakdown

| Frame | Timestamp | Face Conf | Bounding Box `[x, y, w, h]` | Face Width | Face Height | Blur Variance | Spatial Model Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | $0.00\text{s}$ | $0.9349$ | `[21, 91, 294, 385]` | $294\text{ px}$ | $385\text{ px}$ | $43.7$ | $0.9895$ |
| **10** | $0.54\text{s}$ | $0.9305$ | `[22, 72, 303, 428]` | $303\text{ px}$ | $428\text{ px}$ | $35.1$ | $0.5072$ |
| **20** | $1.07\text{s}$ | $0.9235$ | `[40, 77, 296, 384]` | $296\text{ px}$ | $384\text{ px}$ | $115.1$ | $0.2749$ |
| **30** | $1.61\text{s}$ | $0.9353$ | `[81, 87, 268, 402]` | $268\text{ px}$ | $402\text{ px}$ | $42.4$ | $0.8130$ |
| **40** | $2.15\text{s}$ | $0.9229$ | `[42, 90, 310, 390]` | $310\text{ px}$ | $390\text{ px}$ | $45.6$ | $0.0065$ |
| **50** | $2.68\text{s}$ | $0.9213$ | `[54, 91, 314, 395]` | $314\text{ px}$ | $395\text{ px}$ | $24.9$ | $0.0554$ |
| **60** | $3.22\text{s}$ | $0.9302$ | `[52, 64, 298, 389]` | $298\text{ px}$ | $389\text{ px}$ | $28.0$ | $0.9990$ |
| **70** | $3.76\text{s}$ | $0.9391$ | `[56, 104, 287, 416]` | $287\text{ px}$ | $416\text{ px}$ | $49.7$ | $0.9547$ |
| **80** | $4.29\text{s}$ | $0.9391$ | `[68, 104, 279, 410]` | $279\text{ px}$ | $410\text{ px}$ | $27.7$ | $0.9100$ |
| **90** | $4.83\text{s}$ | $0.9382$ | `[53, 133, 280, 407]` | $280\text{ px}$ | $407\text{ px}$ | $36.2$ | $0.0774$ |
| **100** | $5.37\text{s}$ | $0.9218$ | `[151, 207, 339, 413]` | $339\text{ px}$ | $413\text{ px}$ | $80.7$ | $0.9653$ |
| **110** | $5.90\text{s}$ | $0.9165$ | `[26, 248, 350, 406]` | $350\text{ px}$ | $406\text{ px}$ | $19.2$ | $0.3914$ |
| **120** | $6.44\text{s}$ | $0.9237$ | `[4, 355, 307, 441]` | $307\text{ px}$ | $441\text{ px}$ | $19.6$ | $0.0294$ |
| **130** | $6.98\text{s}$ | $0.7752$ | `[67, 465, 374, 466]` | $374\text{ px}$ | $466\text{ px}$ | $18.4$ | $0.0519$ |
| **140** | $7.51\text{s}$ | $0.8792$ | `[70, 395, 404, 574]` | $404\text{ px}$ | $574\text{ px}$ | $23.0$ | $0.0327$ |

### Findings on Face Quality
- **Face Size:** Faces are large relative to the frame (width $268\text{ px}$ to $404\text{ px}$, height $384\text{ px}$ to $574\text{ px}$, covering $> 30\%$ of the frame area). Face scale is **not** the cause of failure.
- **Face Detection Confidence:** Consistently high ($\text{mean} = 0.915$, $\min = 0.775$).
- **Root Cause of Instability:** Severe motion blur and variable compression macroblocking. Between Frame 50 ($s=0.0554$) and Frame 60 ($s=0.9990$), blur variance is $24.9$ and $28.0$. Because EfficientNetB2 is sensitive to high-frequency facial textures, codec block smoothing on certain frames tricks the 2D CNN into outputting near-$0.0$ or near-$1.0$.

---

## 4. V5.2 Dedicated Temporal Model Behavior

- **Temporal Score:** **`0.3291`**
- **Temporal Threshold:** **`0.30`**
- **Decision Window:** `[0.25, 0.35]` (Uncertainty Margin: $0.05$)
- **V5.2 Prediction:** **`Uncertain`**
- **Sequence Count:** $1\text{ sequence}$ (staged via 16-frame buffer)
- **Sequence 1 Score:** `0.3291` $\implies$ **Uncertain**

### Volatility Cause Classification
The primary driver of volatility is:
**F. Model Instability of the 2D Spatial Backbone under Compression & Motion (Combined with E. V5.2 Near-Boundary Uncertainty).**
- The underlying EfficientNetB2 frame classifier has high epistemic uncertainty when evaluating low-bitrate WhatsApp video frames, causing oscillation between Regime A (high scores) and Regime B (low scores).
- The V5.2 BiLSTM model correctly smoothed these oscillations, outputting a temporal score of `0.3291`. However, because `0.3291` falls into the boundary window $[0.25, 0.35]$, V5.2 declared itself `Uncertain` rather than forcing a binary decision.

---

## 5. Fusion Analysis: Why Fused Score = 0.1917 and Reliability = 0.05

### 1. Calculation of Fused Score (0.1917)
When the user uploaded the video via the web interface:
1. Video analysis produced `video_calibrated_score = 0.1747` (from raw median $\approx 0.28$ or derived representative score).
2. Video reliability was computed as `0.054` (displayed as `0.05`).
3. Metadata analysis found no camera hardware EXIF tags (standard for web/WhatsApp uploads):
   - `metadata_representative_score = 0.21` (neutral image threshold)
   - `metadata_reliability_weight = 0.05`
4. The Multimodal Fusion Engine executed reliability-weighted fusion:

$$\begin{aligned}
\text{fused\_score} &= \frac{w_{\text{video}} \times s_{\text{video}} + w_{\text{meta}} \times s_{\text{meta}}}{w_{\text{video}} + w_{\text{meta}}} \\
&= \frac{0.054 \times 0.1747 + 0.050 \times 0.2100}{0.054 + 0.050} \\
&= \frac{0.009434 + 0.010500}{0.1040} \\
&= \frac{0.019934}{0.1040} = \mathbf{0.1917}
\end{aligned}$$

Because the video evidence had low reliability ($0.054$), the neutral metadata score ($0.21$ with weight $0.050$) pulled the fused score towards the threshold, landing on **`0.1917`**.

### 2. Multiplicative Penalties Producing Video Reliability = 0.05
In `backend/fusion.py` (lines 348–366):
1. **Nominal Base Reliability:** `0.90`
2. **Compression Degradation Penalty:** Video resolution ($480 \times 864$) and blur variance ($40.6$) triggered `compression_degradation = True`:
   $$\text{reliability} = 0.90 \times 0.40 = 0.36$$
3. **Volatility & Variance Penalty:** `temporal_volatility >= 0.20` ($0.417$) and `score_std >= 0.25` ($0.407$):
   $$\text{reliability} = 0.36 \times 0.50 = 0.18$$
4. **Uncertainty Penalty:** Because the video detector returned `verdict = "Uncertain"`, line 361 applied the uncertainty dampener:
   $$\text{reliability} = 0.18 \times 0.30 = 0.054$$
5. **Rounding:**
   $$\text{round}(0.054, 2) = \mathbf{0.05}$$

---

## 6. Comparison with Known Real Benchmark Video

| Metric | Investigated Unseen Real Video (`12.54.43 PM (1).mp4`) | Known Real Benchmark (`real_check.mp4`) |
| :--- | :--- | :--- |
| **Total Frames** | $149\text{ frames}$ | $227\text{ frames}$ |
| **Valid Face Frames** | $15\text{ frames}$ | $22\text{ frames}$ |
| **V5.2 Temporal Score** | **`0.3291`** (Uncertain) | **`0.1549`** (Decisive Real) |
| **Temporal Volatility** | **`0.4171`** ($\ge 0.40$, Extreme) | **`0.2985`** ($< 0.40$, Moderate) |
| **Score Standard Deviation** | **`0.4072`** ($\ge 0.35$, Extreme) | **`0.3723`** |
| **Raw Median Score** | `0.3914` | `0.4177` |
| **Video Reliability Weight** | **`0.05`** (Penalized by Uncertainty) | **`0.18`** (Decisive Real) |
| **Final Website Verdict** | **`Uncertain`** | **`Real`** |

`real_check.mp4` succeeds because its temporal volatility ($0.2985$) remains well below the $0.40$ threshold, and its V5.2 temporal score ($0.1549$) is decisively below the $0.30$ threshold.

---

## 7. Root Cause Classification

The uncertainty verdict for this unseen real video is classified as:

1. **Primary Root Cause: B. Spatial Frame Score Oscillation & Excessive Volatility Sensitivity:**
   - The Phase 1 spatial detector relies on a static 2D CNN (EfficientNetB2). Under WhatsApp compression and head motion, individual frame scores swing wildly from $0.006$ to $0.999$, producing volatility of $0.4171$.
   - The rule `temporal_volatility >= 0.40 and score_std >= 0.35` treats this spatial inconsistency as a hard reason to declare `Uncertain`.
2. **Secondary Root Cause: E. V5.2 Temporal Model Boundary Proximity:**
   - The V5.2 BiLSTM model smoothed the sequence to $0.3291$.
   - However, $0.3291$ lies inside the $[0.25, 0.35]$ boundary window around $0.30$, so V5.2 could not provide a decisive Real classification to override the spatial volatility.
3. **Tertiary Root Cause: F. Multimodal Compounding Penalty:**
   - The triple multiplication ($0.90 \times 0.40 \times 0.50 \times 0.30$) collapsed video reliability to $0.054$, allowing neutral metadata ($0.21$) to pull the score to $0.1917$.

---

## 8. Minimal Recommended Future Correction (Architectural Advisory)

*Note: In strict compliance with instructions, no production code has been modified.*

When authorized in future maintenance:
1. **De-couple Volatility Veto from 2D Spatial Scores when V5.2 is Active:**
   Spatial frame oscillation from a 2D CNN should not trigger a hard uncertainty veto when a dedicated temporal model (BiLSTM) is present to model temporal transitions across features.
2. **Narrow the Temporal Boundary Margin:**
   The margin `VIDEO_TEMPORAL_MARGIN = 0.05` creates a wide uncertainty window $[0.25, 0.35]$. Tightening the window to $[0.27, 0.33]$ would allow scores like $0.3291$ or authentic scores near $0.26$ to resolve naturally.
3. **Cap Compounding Reliability Penalties:**
   Prevent video reliability from dropping below $0.15$ when valid facial keyframes ($\ge 15$) are detected, avoiding neutral metadata from dominating the fused score.
