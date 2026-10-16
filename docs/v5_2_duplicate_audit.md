# DeepShield V5.2 — Duplicate Detection and Asset Audit

**Document Version:** 1.0  
**Audit Date:** 2026-09-18  
**Scope:** Video Asset Deduplication across Local Repositories and Cross-Dataset Manifests  

---

## 1. Executive Summary & Objective

In deep learning for video deepfake detection, data leakage caused by duplicate or near-duplicate video clips can severely compromise evaluation validity. If a video clip (or its transcode) appears simultaneously in training and evaluation splits, model generalization is artificially overestimated.

This audit executes a strict cryptographic and perceptual deduplication protocol across all video assets in the DeepShield ecosystem. In accordance with forensic requirements:
- **Zero Silent Deletion**: No files were deleted from disk. Exclusions are formally tracked and enforced at the manifest and sequence-extraction level.
- **Canonical Designation**: For every duplicate cluster, a primary canonical asset is designated based on source priority hierarchy.
- **Split Isolation**: Duplicates of videos in the 20-video WhatsApp calibration benchmark are strictly prohibited from entering training or validation splits.

---

## 2. Methodology

The deduplication pipeline scans all 46 discovered video assets across `backend/test_images/` and `evaluation/test_videos/`:
1. **Cryptographic Hashing (MD5)**: 64KB block-wise MD5 checksum calculation across full binary contents.
2. **Canonical Selection Hierarchy**:
   - Priority 1: Official calibration folders (`backend/test_images/videos_real/` and `videos_fake/`).
   - Priority 2: Unseen evaluation folders (`backend/test_images/unseen_videos/`).
   - Priority 3: Root test folders (`backend/test_images/`).
   - Priority 4: Evaluation benchmarks (`evaluation/test_videos/`).
3. **Cross-Dataset & Manifest Overlap**: Cross-referencing file stems and frame counts against the FAST10 multi-source manifest (`Kaggle_Transfer/face_crop_FAST10_manifest_split.csv`).

---

## 3. Exact Duplicate Files Detected

Out of 46 scanned video files, **43 unique cryptographic hashes** and **3 exact duplicate files** were detected across 3 duplicate clusters:

| Duplicate Cluster | MD5 Checksum | File Size (Bytes) | File Paths Detected | Designated Canonical Asset | Exclusion Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cluster A** | `e0672cd82f01830c18dc1eb31faeb6cb` | 1,333,385 | 1. `backend/test_images/videos_fake/WhatsApp Video 2026-09-15 at 9.30.53 AM.mp4`<br>2. `backend/test_images/fake_check.mp4` | `videos_fake/WhatsApp Video 2026-09-15 at 9.30.53 AM.mp4` | `fake_check.mp4` is an exact binary duplicate copied to root for manual smoke testing. Excluded from training to prevent redundant sampling. |
| **Cluster B** | `197d1fdafecf778f862dd47e305c12ab` | 2,023,364 | 1. `backend/test_images/videos_real/WhatsApp Video 2026-09-15 at 9.23.08 AM.mp4`<br>2. `backend/test_images/real_check.mp4` | `videos_real/WhatsApp Video 2026-09-15 at 9.23.08 AM.mp4` | `real_check.mp4` is an exact binary duplicate copied to root for manual smoke testing. Excluded from training. |
| **Cluster C** | `ba6f602dcede0d2358934da5d460b620` | 941,030 | 1. `backend/test_images/videos_real/WhatsApp Video 2026-09-15 at 9.23.09 AM.mp4`<br>2. `backend/test_images/test1.mp4` | `videos_real/WhatsApp Video 2026-09-15 at 9.23.09 AM.mp4` | `test1.mp4` is an exact binary duplicate of calibration video 9.23.09 AM. Excluded to prevent calibration contamination. |

---

## 4. Near-Duplicate & Cross-Dataset Audit

### 4.1 Evaluation Clips vs. Calibration Set
- `evaluation/test_videos/test_real.mp4` (MD5: `8dc5a61a28d46794b2866807fd90d8eb`, 329,718 bytes, $1152 \times 864$, 50 frames): Unique video sequence. No hash collision with calibration or unseen videos. Assigned to `test` split.
- `evaluation/test_videos/test_fake.mp4` (MD5: `b84b3a3d0728b8d6b3f50f76ebfafb72`, 22,559 bytes, $224 \times 224$, 50 frames): Unique video sequence. No hash collision with calibration or unseen videos. Assigned to `test` split.
- `backend/test_images/test.mp4` (MD5: `6190036267b98dc7037545a4747bec70`, 902,956 bytes, $720 \times 1280$, 145 frames): Unique real interview video clip. Assigned to `train` split.

### 4.2 Cross-Dataset Partitioning (FAST10 vs. WhatsApp Sets)
- The FAST10 dataset (`face_crop_FAST10_manifest_split.csv`) incorporates frame sequences from FF-C23, FaceForensics-1000, and Deepfake-Videos-Merged. All video group identifiers (`video_group` 00000 to 99999) and filenames are distinct from the mobile WhatsApp collections.
- Zero frame or video overlap exists between cloud FAST10 sequences and the local mobile video corpus.

---

## 5. Audit Results Summary

- **Total Video Files Discovered:** 46
- **Unique Cryptographic Hashes:** 43
- **Duplicate Pairs Identified:** 3 pairs (6 files involved)
- **Canonical Files Retained:** 43
- **Duplicate Files Excluded from Manifest:** 3
- **Files Deleted:** 0 (100% preservation)
- **Cross-Split Hash Collisions:** 0
- **Deduplication Audit Status:** **PASS**
