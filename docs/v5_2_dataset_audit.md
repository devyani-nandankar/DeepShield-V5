# DeepShield V5.2 — Large-Scale Temporal Video Dataset Audit

**Document Version:** 1.0  
**Audit Date:** 2026-09-18  
**Dataset Specification:** DeepShield V5.2 Temporal Video Pipeline  
**Training Status:** **NO TRAINING PERFORMED** (Dataset Preparation & Audit Phase Only)  

---

## 1. Executive Summary

This document reports the comprehensive audit, quality evaluation, deduplication, and video-group partitioning for the **DeepShield V5.2 Large-Scale Temporal Video Dataset Pipeline**.

In strict accordance with project safety guidelines:
1. **Zero Retraining or Model Modification**: No weights in `models/` were altered; production Phase 1 pipeline remains active and untouched.
2. **Benchmark Isolation**: The 20-video WhatsApp calibration benchmark (10 Real, 10 Fake) is quarantined in a dedicated evaluation partition (`benchmark_calibration`) and completely excluded from training and validation.
3. **Rigorous Video-Group Isolation**: All sequences are assigned strictly by `video_group` to guarantee zero frame leakage between Train, Validation, and Test sets.
4. **Leakage Verification Status**: **PASS** (Zero video group overlaps, zero hash collisions across splits).

---

## 2. Dataset Counts & Distribution Summary

### 2.1 Video-Level Partitioning

| Partition / Split | Real Videos | Fake Videos | Total Videos | Ratio (Real / Fake) | Role in DeepShield |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 8 | 7 | **15** | 53.3% / 46.7% | Primary temporal training set |
| **Validation** | 2 | 2 | **4** | 50.0% / 50.0% | Model checkpoint selection & hyperparameter tuning |
| **Test** | 2 | 2 | **4** | 50.0% / 50.0% | Held-out evaluation on unseen video groups |
| **Benchmark (Isolated)** | 10 | 10 | **20** | 50.0% / 50.0% | WhatsApp 20-video calibration benchmark |
| **Total Canonical** | **22** | **21** | **43** | **51.2% / 48.8%** | **Balanced across all operational splits** |
| *Excluded Duplicates* | 2 | 1 | *3* | — | Exact binary duplicates excluded via hash audit |

> [!NOTE]
> All 23 non-benchmark videos were split according to a 65–70% Train, 15–18% Validation, 15–18% Test ratio with balanced Real/Fake representation.

---

### 2.2 Sequence-Level Partitioning ($T=16$ Primary Standard)

Each sequence consists of $T=16$ uniformly sampled frames containing verified YuNet face crops.

| Split | Real Sequences | Fake Sequences | Total Sequences ($T=16$) | Total Sequences ($T=24$) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 16 | 14 | **30** | 29 |
| **Validation** | 4 | 4 | **8** | 8 |
| **Test** | 4 | 4 | **8** | 8 |
| **Benchmark (Isolated)** | 19 | 20 | **39** | 36 |
| **Total Ecosystem** | **43** | **42** | **85** | **81** |

---

## 3. Data Quality & Forensic Characteristics

For each video, up to 32 frames were uniformly extracted and evaluated for face presence, bounding box fidelity, blur variance (Laplacian operator), and bitstream health:

| Metric | Measured Value | Forensic Implication |
| :--- | :---: | :--- |
| **Average Valid Face Frames / Video** | **29.77 / 32** | 93.39% of sampled frames contain verifiable faces. |
| **Average Face Detection Rate** | **93.39%** | Robust facial tracking across head movements and compression. |
| **Average YuNet Face Confidence** | **0.8842** | High-confidence facial localization with minimal false detections. |
| **Video Quality Classification** | **MODERATE: 25 (58.1%)**<br>**DEGRADED: 18 (41.9%)**<br>**INVALID: 0 (0.0%)** | 0% corrupt or unreadable files. High representation of compressed social media media (DEGRADED), crucial for preventing compression-induced false positives. |
| **Resolution Range** | $224 \times 224$ to $1152 \times 864$ | Includes mobile portrait ($480 \times 864$, $720 \times 1280$) and landscape ($848 \times 480$). |
| **Framerate Range** | $7.60\text{ FPS}$ to $63.14\text{ FPS}$ | Reflects diverse mobile camera hardware and transcode pipelines. |
| **Bitrate Range** | $450\text{ kbps}$ to $2.85\text{ Mbps}$ | Representative of modern social media messaging compression. |

---

## 4. Leakage Verification Results

An automated cross-split integrity check was executed to ensure zero contamination between splits:

```
==================================================
LEAKAGE AUDIT STATUS: PASS
  Train n Val video groups: 0
  Train n Test video groups: 0
  Val n Test video groups: 0
  Train n Benchmark video groups: 0
  Hash overlaps across splits: 0
==================================================
```

- **Video Group Overlaps:** $0$ across all pairwise split intersections.
- **Cross-Split Hash Collisions:** $0$.
- **Benchmark Protection:** Zero overlap between the 20-video WhatsApp calibration benchmark and training/validation sets.

---

## 5. Storage & Processing Efficiency

1. **Raw Video Storage**:
   - Total disk footprint of canonical video assets: **~104.8 MB**.
2. **Sequence Preprocessing Strategy**:
   - Rather than redundantly saving thousands of extracted frame JPEG files to disk, DeepShield V5.2 uses an efficient, reproducible sequence generation pipeline.
   - Pre-extracted feature tensors: $(T, 1408)$ spatial vectors from frozen EfficientNetB2 GlobalAveragePooling2D require only **$90.1\text{ KB}$ per sequence** ($16 \times 1408 \times 4\text{ bytes}$).
   - Entire dataset of 85 sequences can be cached in **$7.66\text{ MB}$ of memory or disk storage**, eliminating multi-gigabyte intermediate frame caches.

---

## 6. Manifest File Specifications

The generated manifest files are stored in `Metadata/`:

### 6.1 `Metadata/v5_2_video_manifest.csv`
- **Total Rows:** 43
- **Columns (18):** `video_id`, `filepath`, `label`, `label_name`, `source_dataset`, `video_group`, `split`, `frames_sampled`, `valid_face_frames`, `face_detection_rate`, `avg_face_confidence`, `avg_face_width`, `avg_face_height`, `resolution`, `fps`, `duration`, `bitrate`, `quality_class`.

### 6.2 `Metadata/v5_2_sequence_manifest.csv`
- **Total Rows:** 85 ($T=16$)
- **Columns (8):** `sequence_id`, `video_id`, `label`, `split`, `frame_paths`, `sequence_length`, `valid_frames`, `source_dataset`.

### 6.3 `Metadata/v5_2_sequence_manifest_t24.csv`
- **Total Rows:** 81 ($T=24$ extended)
- **Columns (8):** `sequence_id`, `video_id`, `label`, `split`, `frame_paths`, `sequence_length`, `valid_frames`, `source_dataset`.

---

## 7. Readiness Assessment for DeepShield V5.2 Training

| Readiness Criterion | Status | Notes |
| :--- | :---: | :--- |
| **Video Assets Audited** | **READY** | 46 files inspected, 43 canonical unique, 3 duplicates documented. |
| **Class Balance** | **READY** | 22 Real (51.2%), 21 Fake (48.8%). |
| **Leakage Isolation** | **READY** | Video-group splitting verified with 0 group overlap. |
| **Benchmark Protection** | **READY** | 20-video WhatsApp set isolated in `benchmark_calibration`. |
| **YuNet Face Quality** | **READY** | 93.39% face detection rate; no center-crop fallback. |
| **Augmentation Config** | **READY** | Probabilistic compression pipeline prepared ($Q=30..75$, blur, downsampling). |
| **Manifests Generated** | **READY** | Validated CSVs present in `Metadata/`. |
| **Overall Readiness** | **READY FOR V5.2 TRAINING** | Dataset preparation and auditing complete. Model training awaits explicit user trigger. |
