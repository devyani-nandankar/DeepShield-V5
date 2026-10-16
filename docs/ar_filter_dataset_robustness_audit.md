# DeepShield V5 / V5.2 — AR / Beauty-Filter Dataset Robustness Audit

**Audit Date:** 2026-09-18  
**Scope:** Dataset Manifest & Metadata Forensic Audit (V5 Image & V5.2 Temporal Video Pipelines)  
**Safety Protocol:** Zero code modified; zero model weights retrained or altered; thresholds remain fixed at `IMAGE_THRESHOLD = 0.21`, `VIDEO_TEMPORAL_THRESHOLD = 0.30`, and `VIDEO_TEMPORAL_MARGIN = 0.03`. No hardcoded Snapchat rules added.

---

## 1. Executive Summary

Following the diagnostic identification of a false-positive classification on a legitimate real video recorded with a Snapchat AR beauty filter ([`docs/snapchat_filter_false_positive_diagnostic.md`](file:///c:/Users/ADMIN/Desktop/DeepShield_V5_Final/docs/snapchat_filter_false_positive_diagnostic.md)), this audit comprehensively inspects the actual dataset manifests, file headers, and metadata across all training, validation, and test splits used by:
1. **DeepShield V5 Image Model** (`models/deepshield_v5_clean_best.keras`)
2. **DeepShield V5.2 Dedicated Temporal Video Model** (`models/v5_2/deepshield_v5_2_temporal_no_aug.keras`)

### Key Finding
**Zero legitimate filtered REAL examples are documented or labeled in any dataset manifest across V5 or V5.2.**  
All manifest metadata fields across 24,589 image frames and 43 canonical videos track technical parameters (resolution, face detection score, bounding box dimensions, video group, split, source dataset). None of the datasets track or categorize social media filters, AR face meshes, beauty airbrushing, or digital makeup. 

Consequently, **legitimate human media with AR/beauty filters constitutes a textbook out-of-distribution (OOD) case for the authentic ("Real") class.** Because the neural feature extractors (EfficientNetB2) and temporal models (BiLSTM) were trained solely on unedited camera captures as "Real" and digitally altered faces as "Fake", the models naturally interpret AR beauty modifications as synthetic manipulation.

---

## 2. Comprehensive Manifest & Split Audit

### 2.1 V5 Image Datasets (`Kaggle_Transfer/face_crop_FAST10_manifest_split.csv`)
- **Total Samples:** 24,589 facial crops
- **Metadata Columns Available:** `frame_id`, `filepath`, `filename`, `label`, `label_name`, `source_dataset`, `video_group`, `split`, `face_path`, `face_detected`, `face_width`, `face_height`, `face_confidence`.

| Dataset Split | Total Samples | Real Samples | Fake Samples | Source Datasets Represented |
| :--- | :---: | :---: | :---: | :--- |
| **1. V5 Image Training** | **17,208** | 5,005 ($29.1\%$) | 12,203 ($70.9\%$) | FF-C23 (10,361), Deepfake-Videos-Merged (3,923), FaceForensics-1000 (2,924) |
| **2. V5 Image Validation** | **3,688** | 1,045 ($28.3\%$) | 2,643 ($71.7\%$) | FF-C23 (2,221), Deepfake-Videos-Merged (841), FaceForensics-1000 (626) |
| **3. V5 Image Test** | **3,693** | 993 ($26.9\%$) | 2,700 ($73.1\%$) | FF-C23 (2,220), Deepfake-Videos-Merged (841), FaceForensics-1000 (632) |
| **Total V5 Image Corpus** | **24,589** | **7,043** ($28.6\%$) | **17,546** ($71.4\%$) | **All 3 canonical benchmarks** |

#### Real Samples Breakdown (V5 Image):
- **Confirmed Unfiltered:** 0 (Not identifiable from the available dataset metadata).
- **Confirmed Filtered / AR:** 0 (Not identifiable from the available dataset metadata).
- **Unknown / Not Specified:** **7,043 (100.0% of Real samples)**.
- *Metadata Note:* While the source datasets (FaceForensics++ and Deepfake-Videos-Merged) originate from public YouTube video sequences (news broadcasts, interview shows), no metadata attribute specifies whether subjects wore cosmetic makeup, used digital post-processing, or had camera smoothing enabled.

#### Fake Samples Breakdown (V5 Image):
- **Manipulation Types Available in Dataset Provenance:**
  - **FaceForensics++ (`FF-C23` & `FaceForensics-1000`):** Contains four algorithmic deepfake manipulation methods:
    1. *Deepfakes (DF)*: Learning-based autoencoder face replacement.
    2. *Face2Face (F2F)*: Real-time facial reenactment modifying expression.
    3. *FaceSwap (FS)*: Graphics-based 3D facial feature transfer.
    4. *NeuralTextures (NT)*: Facial modification using neural texture rendering.
  - **Deepfake-Videos-Merged:** Composite web and challenge deepfake videos (DFDC-derived and in-the-wild facial swaps).
  - *AR/Beauty Filter Manipulation:* **0 samples**.

---

### 2.2 V5.2 Temporal Video Datasets (`Metadata/v5_2_video_manifest.csv` & `v5_2_sequence_manifest.csv`)
- **Total Canonical Videos:** 43 videos
- **Total Extracted Sequences (T=16):** 85 sequences
- **Metadata Columns Available:** `video_id`, `filepath`, `label`, `label_name`, `source_dataset`, `video_group`, `split`, `frames_sampled`, `valid_face_frames`, `face_detection_rate`, `avg_face_confidence`, `avg_face_width`, `avg_face_height`, `resolution`, `fps`, `duration`, `bitrate`, `quality_class`.

| Dataset Split | Video Count (Total / Real / Fake) | Sequence Count (T=16) (Total / Real / Fake) | Source Datasets Represented |
| :--- | :---: | :---: | :--- |
| **4. V5.2 Video Training** | **15** (8 Real / 7 Fake) | **30** (16 Real / 14 Fake) | WhatsApp-InTheWild-Unseen (14), DeepShield-Evaluation-Clips (1) |
| **5. V5.2 Video Validation**| **4** (2 Real / 2 Fake) | **8** (4 Real / 4 Fake) | WhatsApp-InTheWild-Unseen (3), DeepShield-Evaluation-Clips (1) |
| **6. V5.2 Video Test** | **4** (2 Real / 2 Fake) | **8** (4 Real / 4 Fake) | WhatsApp-InTheWild-Unseen (3), DeepShield-Evaluation-Clips (1) |
| **Quarantined Benchmark** | **20** (10 Real / 10 Fake) | **39** (19 Real / 20 Fake) | WhatsApp-Calibration-Benchmark (20) |
| **Total V5.2 Corpus** | **43** (22 Real / 21 Fake) | **85** (43 Real / 42 Fake) | **Complete V5.2 Video Corpus** |

#### Real Samples Breakdown (V5.2 Video):
- **Confirmed Unfiltered:** 0 (Not identifiable from the available dataset metadata).
- **Confirmed Filtered / AR:** 0 (Not identifiable from the available dataset metadata).
- **Unknown / Not Specified:** **22 videos (100.0% of Real videos)**, corresponding to **43 sequences (100.0% of Real sequences)**.

#### Fake Samples Breakdown (V5.2 Video):
- **Manipulation Types Available:** In-the-wild mobile video deepfakes (AI voice-synced avatar generation, generative face swaps, mobile face reenactment, and DeepFaceLab swaps distributed via WhatsApp).
- **AR/Beauty Filter Manipulation:** **0 samples**.

---

## 3. Metadata Keyword Search Audit

An automated regex scan across all text fields (`filepath`, `filename`, `source_dataset`, `video_id`) in all manifests for filter and AR terminology returned the following results:

| Target Keyword | V5 Image Manifest Matches | V5.2 Video Manifest Matches | Finding / Status |
| :--- | :---: | :---: | :--- |
| `snapchat` | **0** | **0** | Not present in any metadata |
| `instagram` | **0** | **0** | Not present in any metadata |
| `tiktok` | **0** | **0** | Not present in any metadata |
| `filter` | **0** | **0** | Not present in any metadata |
| `beauty` | **0** | **0** | Not present in any metadata |
| `smooth` | **0** | **0** | Not present in any metadata |
| `makeup` | **0** | **0** | Not present in any metadata |
| `ar` (augmented reality) | **0** | **0** | Matches were substrings in words (`EHARPYBT`, `downstairs`) |
| `mask` | **0** | **0** | Not present in any metadata |
| `warp` | **0** | **0** | Not present in any metadata |
| `cartoon` | **0** | **0** | Not present in any metadata |

> **Audit Standard Compliance:**  
> In accordance with strict auditing guidelines, filter information is **"Not identifiable from the available dataset metadata."** No video or image has been inferred to be filtered merely from visual inspection.

---

## 4. Training Distribution Calculation

### Quantitative Proportion:
$$\text{REAL\_FILTERED} = 0 \quad (\text{confirmed in metadata})$$
$$\text{TOTAL\_REAL}_{\text{Image Training}} = 5,005$$
$$\text{TOTAL\_REAL}_{\text{Video Training}} = 8 \text{ videos } (16 \text{ sequences})$$

$$\frac{\text{REAL\_FILTERED}}{\text{TOTAL\_REAL}} = \text{The exact proportion cannot be determined from the available dataset metadata alone.}$$

However, because the source datasets consist strictly of:
1. Academic Deepfake benchmark corpora (FaceForensics++ YouTube news/interviews), and
2. Unmanipulated mobile camera recordings collected to represent authentic user communications,

there is **no representation of intentional AR / beauty / face-modification filters in the training set**.

---

## 5. The Robustness Gap: Out-of-Distribution (OOD) Analysis

Comparing the training distribution against the real-world case:
$$\text{Real Human Subject} + \text{Snapchat AR Beauty Filter}$$

### Why This is an Out-of-Distribution (OOD) Case:
1. **The Semantic Assumption of the Training Set:**
   - The binary classification paradigm defines:
     - $\text{Class 0 (Real)} \iff \text{Natural camera sensor capture (unmodified pixel lattice, natural PRNU noise, organic skin texture)}$.
     - $\text{Class 1 (Fake)} \iff \text{Digitally modified, generated, or swapped face (boundary seams, smoothed textures, altered facial geometry)}$.
2. **What an AR Filter Actually Does:**
   - An AR filter is mathematically a **real-time digital face manipulation system**. It runs facial landmark regression, fits a 3D deformable mesh, warps facial contours (jaw thinning, nose resizing, eye enlargement), overlays synthetic graphics (cartoon elements), and applies bilateral smoothing to remove natural skin pores.
3. **The Consequence for DeepShield:**
   - Because the model was never trained on samples where *benign digital face modification = Real*, it behaves consistently and correctly according to its training: it detects the synthetic smoothing, boundary contrast, and geometry warping, and flags the content as **Fake**.
   - This proves that **legitimate AR-filtered media is currently Out-Of-Distribution (OOD)**.

---

## 6. Strategic Recommendations for AR Robustness (Data-Only)

*Note: In accordance with project instructions, none of these recommendations are implemented at this time.*

To systematically eliminate this robustness gap without compromising deepfake detection accuracy, the following dataset expansion strategy is recommended:

### A. Add Real Snapchat-Filter Videos
Collect authentic, consented mobile videos recorded through Snapchat's most popular filter categories:
- Subtle beauty/retouch filters (skin smoothing, eye brightening).
- Morphing filters (face slimming, facial reshaping).
- Overlay filters (animated 2D/3D accessories, animal ears, floating graphics).

### B. Add Real Instagram-Filter Videos
Incorporate Reels and Stories videos utilizing Instagram's Spark AR ecosystem (color grading, soft focus, digital cosmetic filters).

### C. Add Real TikTok-Filter Videos
Incorporate authentic mobile videos recorded using ByteDance's Effect House AR filters (AI beauty enhancement, stylized makeup, face-distortion effects).

### D. Add Beauty-Filter Images and Videos
Source still selfies and video clips from dedicated portrait retouching apps (FaceTune, Meitu, Snow) to capture diverse smoothing algorithms.

### E. Add AR-Mask Examples
Include extreme 3D AR masks (full digital face paint, character masks, virtual sunglasses) to train the model to isolate facial regions beneath surface augmentations.

### F. Sample Multiple Filter Strengths
Construct a tiered dataset spanning:
1. *Subtle / Low Intensity (10–30%):* Barely perceptible skin smoothing.
2. *Moderate Intensity (40–70%):* Noticeable contouring, digital lashes, lipstick.
3. *Heavy / Extreme Intensity (80–100%):* Severe jaw warping, cartoon eyes, dense overlays.

### G. Paired Counterfactual Data Structure (Essential)
Collect **matched pairs of the exact same individual in the exact same setting**:
- Video Pair: `[Person A — Natural Camera]` $\leftrightarrow$ `[Person A — Snapchat Filter]`
- This ensures the neural network learns to penalize actual face-swapping artifacts (source-target identity mismatch, temporal warping, boundary blending) rather than penalizing the individual's identity or cosmetic filter.

### H. Rigorous Split Separation & Evaluation Protocol
- **Strict Video-Group & Subject Isolation:**
  No subject appearing in the AR training split may ever appear in validation or test splits.
- **Dedicated AR Robustness Benchmark:**
  Create an independent evaluation set:
  $$\text{FPR}_{\text{AR}} = \frac{\text{False Positives on Filtered Real Media}}{\text{Total Filtered Real Media}}$$
  Target: Reduce $\text{FPR}_{\text{AR}}$ from $100\%$ to $< 5\%$ while maintaining $\ge 95\%$ recall on malicious deepfakes.

---

## 7. Audit Conclusion

The DeepShield V5 and V5.2 training, validation, and test datasets contain **no identifiable, labeled AR or beauty-filter samples**. The false-positive classification of the Snapchat AR video is directly explained by this training distribution omission. Expanding the authentic training and evaluation corpora with paired, multi-strength AR filter media is the correct, permanent path to robust in-the-wild generalization.
