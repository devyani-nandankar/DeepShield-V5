# DeepShield V5.2: Volatility Decoupling & Margin Refinements Report
**Date:** 2026-09-18  
**Status:** Completed & Validated  
**Module:** DeepShield V5.2 Multimodal Temporal Video Fusion  
**Test Suite:** 65 / 65 PASS  

---

## 1. Executive Summary

Following the forensic diagnostic in `docs/v5_2_unseen_real_video_diagnostic.md`, the minimal volatility and margin refinements have been implemented, tested, and validated.

### Key Objectives & Compliance:
1. **Threshold Preservation:**
   - Image Decision Threshold: `CALIBRATED_THRESHOLD = 0.21` strictly preserved.
   - Temporal Decision Threshold: `VIDEO_TEMPORAL_THRESHOLD = 0.30` strictly preserved.
2. **Model Integrity:** Zero model files were retrained or modified. Existing V4, V5, V5.1, and V5.2 weights remain intact.
3. **Decoupled Spatial Volatility Veto:** When dedicated V5.2 temporal evidence is available and has sufficient facial evidence ($\ge 15$ valid face frames), legacy Phase 1 spatial frame jumping does not force an automatic `Uncertain` verdict. Volatility is preserved as an audit and reliability dampener.
4. **Refined Temporal Margin:** The uncertainty margin `VIDEO_TEMPORAL_MARGIN` was narrowed from $\pm 0.05$ ($[0.25, 0.35]$) to $\pm 0.03$ ($[0.27, 0.33]$). Scores $< 0.27$ indicate Real evidence, and scores $> 0.33$ indicate Fake evidence.
5. **Video Reliability Floor:** When valid face frames $\ge 15$, video evidence reliability has a lower bound floor of $0.15$ (`max(calculated, 0.15)`), preventing secondary metadata from dominating the final multimodal score.
6. **Preservation of Genuine Uncertainty:** Videos with genuine ambiguity ($0.27 \le s \le 0.33$) or strong inter-modal contradictions continue to report `Uncertain`.

---

## 2. Before vs. After Logic Specification

| Component | Before Refinement | After Refinement | Rationale |
| :--- | :--- | :--- | :--- |
| **Temporal Margin** | $[0.25, 0.35]$ (width: $0.10$) around $0.30$ | **$[0.27, 0.33]$** (width: $0.06$) around $0.30$ | Tightens uncertainty window while maintaining $0.30$ calibrated threshold. |
| **Spatial Volatility Veto** | Volatility $\ge 0.40$ + std $\ge 0.35$ unconditionally forced Phase 1 to `Uncertain`, blocking decisive temporal verdicts. | When V5.2 has $\ge 15$ face frames and decisive evidence (`Real`/`Fake`), spatial volatility is **decoupled** from the veto. | A 2D CNN's frame score oscillation must not veto a dedicated sequence BiLSTM model. |
| **Volatility Audit Note** | None. Explanation cited ambiguous keyframes. | **"Phase 1 spatial score volatility was treated as a reliability factor rather than an automatic final-decision veto because dedicated V5.2 temporal evidence was available."** | Full forensic transparency when decoupling triggers. |
| **Video Reliability** | Multiplicative degradation ($0.90 \times 0.40 \times 0.50 \times 0.30$) allowed reliability to drop to $0.05$. | **Floor of $0.15$** applied when `valid_face_frames >= 15`: `max(reliability, 0.15)`. | Prevents neutral metadata ($0.21$, weight $0.05$) from overriding visual evidence. |
| **Representative Score** | Phase 1 noisy median was blended $50/50$ with V5.2 even when Phase 1 was volatile. | When spatial volatility is decoupled, `representative_score = calibrated_temporal_score`. | Eliminates score distortion caused by volatile 2D frames. |

---

## 3. Automated Test Suite Results

- **Command:** `python -m unittest discover -s backend/tests -p "test_*.py"`
- **Total Tests:** **65 tests** (56 pre-existing regression tests + 9 new targeted refinement tests)
- **Status:** **65 / 65 PASSED (100%)** in 22.814s.

### New Test Suite (`test_v5_2_volatility_margin_refinements.py`):
- `test_01_threshold_and_margin_invariants`: Verified thresholds ($0.21, 0.30$) and margin ($0.03$).
- `test_02_v5_2_boundary_scores`: Verified scores $0.26$ (Real), $0.27$ (Uncertain), $0.30$ (Uncertain), $0.33$ (Uncertain), $0.34$ (Fake), $0.40$ (Fake).
- `test_03_high_volatility_plus_strong_v5_2_real`: High spatial volatility + strong V5.2 Real -> `Real` with exact audit note.
- `test_04_high_volatility_plus_strong_v5_2_fake`: High spatial volatility + strong V5.2 Fake -> `Likely Fake` with exact audit note.
- `test_05_volatility_with_v5_2_uncertain_remains_uncertain`: Boundary score ($0.30$) preserves `Uncertain` verdict.
- `test_06_face_frames_boundaries_14_vs_15_vs_16`:
  - 14 frames: No floor, legacy volatility veto maintained ($< 15$).
  - 15 frames: Reliability floor ($0.15$) applied, volatility decoupled.
  - 16 frames: Eligible for high-confidence override ($\le 0.22$ or $\ge 0.40$).
- `test_07_reliability_floor_exact`: Verified exact lower bound calculation ($0.054 \to 0.15$).
- `test_08_genuine_strong_contradiction`: Image Fake ($0.85$) vs Video Real ($0.10$) correctly triggers inter-modal conflict (`Uncertain`).
- `test_09_existing_phase1_v5_2_agreement`: Concordant Real and concordant Fake behaviors fully preserved.

---

## 4. Live 4-Video Benchmark Validation Results

The updated pipeline was evaluated on the four benchmark videos:

| Metric | Video A: `real_check.mp4` | Video B: `fake_check.mp4` | Video C: Unseen Real (`12.54.43 PM (1).mp4`) | Video D: `test.mp4` |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1 Prediction** | `Real` | `Fake` | `Real` | `Real` |
| **Phase 1 Score** | $0.1548$ | $0.9992$ | $0.1618$ | $0.0194$ |
| **Phase 1 Volatility** | $0.2985$ | $0.0989$ | $0.4171$ | $0.2123$ |
| **V5.2 Prediction** | `Real` | `Fake` | `Uncertain` | `Fake` |
| **V5.2 Score** | $0.2214$ | $0.6193$ | **$0.3291$** | $0.4602$ |
| **V5.2 Confidence** | $0.6311$ | $0.7281$ | $0.5000$ | $0.6145$ |
| **Valid Face Frames** | $22$ | $21$ | $15$ | $15$ |
| **Video Reliability** | $0.18$ | $0.36$ | **$0.15$ (floor applied)** | $0.18$ |
| **Fused Multimodal Score** | $0.1549$ | $0.7419$ | **$0.2023$** | $0.3908$ |
| **Inter-Modal Conflict** | False | False | False | False |
| **Final Verdict** | **`Real`** | **`Likely Fake`** | **`Uncertain`** | **`Likely Fake`** |
| **Uncertainty Reason** | None | None | `Temporal prediction score (0.3291) is in the boundary window [0.27, 0.33].` | None |

---

## 5. Analysis of Remaining Uncertainty Cases

### Video C Analysis:
- The unseen real video produced `final_verdict = Uncertain`.
- **Reason:** V5.2 temporal model evaluated the sequence and assigned a score of **$0.3291$**.
- Because $0.27 \le 0.3291 \le 0.33$, the score lies squarely within the calibrated temporal uncertainty margin window.
- **Key Improvement over Pre-Refinement State:**
  - Previously, the uncertainty reason was an artifact of Phase 1 2D spatial jitter: `"Extreme temporal score volatility (0.43, std: 0.37) precludes decisive classification."`.
  - Now, spatial volatility is decoupled from the veto, and the system honestly identifies the true scientific source of ambiguity: `"Temporal prediction score (0.3291) is in the boundary window [0.27, 0.33]."`.
  - Video reliability was lifted by the floor from $0.05$ to $0.15$, bringing the fused score to $0.2023$ (below the $0.21$ image threshold).
  - DeepShield honors scientific integrity: uncertainty is not artificially masked when a temporal model score falls on the decision boundary.

---

## 6. False Positives / False Negatives Audit

- **False Positives:** Zero ($0$) new false positives observed. Video A (`real_check.mp4`) remains classified as `Real` ($0.1549$).
- **False Negatives:** Zero ($0$) new false negatives observed. Video B (`fake_check.mp4`) and Video D (`test.mp4`) are classified as `Likely Fake` ($0.7419$ and $0.3908$).
- **System Stability:** All 65 unit tests pass, Flask backend daemon is operating cleanly on port 5000, and model weights remain bit-for-bit identical to baseline.
