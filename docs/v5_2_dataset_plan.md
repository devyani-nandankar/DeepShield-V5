# DeepShield V5.2 — Large-Scale Temporal Video Dataset Plan

**Document Version:** 1.0  
**Status:** Approved Specification  
**Architecture Target:** DeepShield V5.2 Temporal Video Pipeline  

---

## 1. Executive Summary & Objective

In DeepShield V5.1, a dedicated spatial-temporal model (EfficientNetB2 CNN feature extractor + Bidirectional LSTM) was designed and benchmarked. While demonstrating high temporal discriminative potential (ROC-AUC 0.80 on held-out test and 0.92 on the WhatsApp benchmark), V5.1 was constrained by an initial training set of only 10 videos (29 temporal sequences).

**DeepShield V5.2** establishes a large-scale, multi-source video dataset pipeline designed to scale temporal training across thousands of video sequences while maintaining strict forensic integrity:
- **Zero Data Leakage:** Video-group isolation prevents frame leakage across Train, Validation, and Test splits.
- **Benchmark Preservation:** The 20-video WhatsApp calibration benchmark remains strictly isolated as an external generalization evaluation corpus.
- **Robustness to Compression:** Incorporates multi-compression sources (c23, c40, WhatsApp transcoding) to counter domain shift and score inversion across compression regimes.
- **Reproducible Sequence Extraction:** Structured manifests define uniform temporal sampling ($T=16$ and $T=24$) with YuNet face filtering and probabilistic compression augmentation.

---

## 2. Dataset Sources & Inventory

The V5.2 pipeline integrates four established deepfake video sources:

```
+---------------------------------------------------------------------------------------------------+
| DeepShield V5.2 Dataset Portfolio                                                                  |
+------------------------------------+--------------------------------+-----------------------------+
| High-Fidelity & Synthesis (Lab)    | Multi-Compression Benchmarks   | In-The-Wild Social Media    |
| - Deepfake-Videos-Merged           | - FaceForensics++ (FF-C23)     | - WhatsApp Calibration (20) |
|   (Celeb-DF v2 / DFDC subsets)     | - FaceForensics-1000 (c40)     | - WhatsApp In-The-Wild (20) |
+------------------------------------+--------------------------------+-----------------------------+
```

### Source 1: FaceForensics++ (FF-C23)
- **Dataset Name:** FaceForensics++ (FF-C23)
- **Source:** Technical University of Munich (TUM) & Google Research.
- **Labels:** Real and Fake (Manipulated).
- **Approximate Video Count:** 1,000 real original YouTube video sequences; manipulated into 4,000 deepfake videos across multiple methods. In the DeepShield FAST10 manifest: **4,936 video groups** (14,802 keyframes: 2,097 Real, 12,705 Fake).
- **Format & Codec:** MP4 container, H.264 video codec, CRF 23 compression parameter.
- **Expected Quality:** High to moderate resolution (typically 720p/1080p). Visual artifacts are subtle and representative of standard web streaming video compression.
- **Manipulation Types Documented:** 
  - *Deepfakes (DF)*: Autoencoder-based face replacement.
  - *Face2Face (F2F)*: Real-time facial reenactment.
  - *FaceSwap (FS)*: Classical 3D graphics face swapping.
  - *NeuralTextures (NT)*: Neural rendering for photorealistic facial reenactment.
- **License / Access Notes:** Academic non-commercial research license. Terms require direct registration with the TUM Computer Vision group.
- **Suitability for DeepShield:** **High.** Crucial for training fine-grained facial boundary artifacts, expression inconsistencies, and temporal jitter at moderate compression.

---

### Source 2: FaceForensics-1000 (c40 / Heavily Compressed)
- **Dataset Name:** FaceForensics-1000
- **Source:** FaceForensics benchmark subset (heavily compressed c40 standard).
- **Labels:** Real and Fake (Balanced 1:1 in curated subset).
- **Approximate Video Count:** 1,000 source video sequences. In the DeepShield FAST10 manifest: **1,394 video groups** (4,182 keyframes: 2,082 Real, 2,100 Fake).
- **Format & Codec:** MP4 container, H.264 / MJPEG, CRF 40 compression parameter.
- **Expected Quality:** Low to moderate visual fidelity. Heavy compression introduces block discrete cosine transform (DCT) artifacts, mosquito noise, and edge softening.
- **Manipulation Types Documented:** Reenactment and face swapping subjected to heavy post-compression.
- **License / Access Notes:** Academic research license (TUM).
- **Suitability for DeepShield:** **High.** Directly addresses "Regime A" (compression-smoothed videos). Trains the temporal model to remain invariant to quantization noise and blockiness without mistaking compression artifacts for synthetic manipulation.

---

### Source 3: Deepfake-Videos-Merged (Celeb-DF v2 + DFDC Challenge)
- **Dataset Name:** Deepfake-Videos-Merged
- **Source:** Merged high-quality open benchmarks including Celeb-DF v2 (State University of New York at Buffalo) and the Deepfake Detection Challenge (DFDC) preview corpus.
- **Labels:** Real and Fake (Balanced).
- **Approximate Video Count:** In the DeepShield FAST10 manifest: **1,873 video groups** (5,605 keyframes: 2,864 Real, 2,741 Fake).
- **Format & Codec:** MP4 container, H.264/MPEG-4 AVC, variable bitrates (typically 2–8 Mbps).
- **Expected Quality:** High quality. Minimal initial compression, high facial resolution, diverse celebrity subjects, natural lighting, and multi-angle poses.
- **Manipulation Types Documented:**
  - Modern GAN-based face swapping with color matching and boundary blending.
  - Variable face-swap masks with post-processing edge refinement.
- **License / Access Notes:** Celeb-DF research agreement (non-commercial research); DFDC dataset challenge license (Meta / Kaggle).
- **Suitability for DeepShield:** **Essential.** Represents "Regime B" (high-fidelity synthesis). Prevents false negatives on modern high-resolution deepfakes where spatial blending is photorealistic.

---

### Source 4: WhatsApp In-The-Wild Video Corpus (Local Benchmark & Evaluation)
- **Dataset Name:** WhatsApp Real-World Mobile Corpus
- **Source:** Locally collected in-the-wild video recordings transmitted through WhatsApp's media compression and transcoding pipeline.
- **Subsets:**
  1. **20-Video WhatsApp Calibration Benchmark:** 10 Real (`backend/test_images/videos_real/`), 10 Fake (`backend/test_images/videos_fake/`). Kept strictly isolated as the primary external benchmark.
  2. **Unseen Real-World In-the-Wild Videos:** 10 Real (`backend/test_images/unseen_videos/real/`), 10 Fake (`backend/test_images/unseen_videos/fake/`). Used for unseen held-out temporal evaluation.
  3. **Verification Clips:** `test.mp4`, `evaluation/test_videos/test_real.mp4`, `evaluation/test_videos/test_fake.mp4`.
- **Labels:** Ground truth verified Real and Fake.
- **Approximate Video Count:** 46 local files across subsets (40 unique videos after deduplication).
- **Format & Codec:** MP4 container, AVC1 / H.264, AAC audio, variable framerates ($7.60$ to $63.14$ FPS), resolutions ($480 \times 848$, $480 \times 864$, $720 \times 1280$, $848 \times 480$).
- **Expected Quality:** Heavily transcoded by social media messaging algorithms (downscaled, compressed, low bitrate $500\text{ kbps} - 2.5\text{ Mbps}$, temporal motion blur).
- **Manipulation Types Documented:** Authentic user phone captures vs. mobile-circulated synthetic deepfakes.
- **License / Access Notes:** Internal proprietary benchmark & evaluation corpus.
- **Suitability for DeepShield:** **Gold Standard for Generalization.** Evaluates operational resilience in real-world messaging scenarios.

---

## 3. Dataset Portfolio Summary Table

| Dataset Name | Source | Real Videos / Groups | Fake Videos / Groups | Total Video Groups | Primary Format | Compression Level | Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FF-C23** | TUM / Google | ~1,000 (2,097 frames) | ~4,000 (12,705 frames) | 4,936 | MP4 (H.264) | Moderate (CRF 23) | High (Core temporal learning) |
| **FaceForensics-1000** | TUM | 500 (2,082 frames) | 500 (2,100 frames) | 1,394 | MP4 / JPEG | Heavy (CRF 40) | High (Compression robustness) |
| **Deepfake-Videos-Merged** | Celeb-DF v2 / DFDC | 950 (2,864 frames) | 923 (2,741 frames) | 1,873 | MP4 (H.264) | Low to Moderate | Essential (High-fidelity deepfakes) |
| **WhatsApp In-The-Wild (Unseen)** | Local Mobile Transcoding | 10 videos | 10 videos | 20 | MP4 (AVC1) | Variable Mobile | High (Generalization testing) |
| **WhatsApp Calibration (Benchmark)** | Local Mobile Transcoding | 10 videos | 10 videos | 20 | MP4 (AVC1) | Variable Mobile | Primary External Benchmark |
| **Total Ecosystem** | Multi-Source | **> 2,470 groups** | **> 5,430 groups** | **> 8,240 groups** | Mixed | Broad Spectrum | Comprehensive |

---

## 4. Architectural Integration & Constraints

1. **Storage Optimization**:
   - The large-scale extracted dataset resides securely in Google Drive / Kaggle storage (`/content/drive/MyDrive/DeepShield_Datasets/`).
   - Local repositories store manifests, deduplication indices, evaluation benchmarks, and preprocessing generators to maintain zero bloat in local git repositories.
2. **Video-Level Isolation**:
   - Every video is assigned a unique `video_id` and `video_group`.
   - All temporal sequences derived from a video remain within one split: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
3. **YuNet Face Filtering**:
   - Every frame must be localized by YuNet (`models/yunet/face_detection_yunet_2023mar.onnx`).
   - Non-face frames are discarded. Center-crop fallback is strictly banned.
4. **Sequence Representation**:
   - Sequences use uniform temporal subsampling with $T=16$ (primary) and $T=24$ (optional extended).
   - Sequence tensors are pre-extracted to $(T, 1408)$ spatial feature vectors via frozen EfficientNetB2 GlobalAveragePooling2D for fast BiLSTM training.
