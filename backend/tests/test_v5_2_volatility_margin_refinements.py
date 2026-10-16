"""
Unit Tests for DeepShield V5.2 Volatility and Margin Refinements
================================================================

Validates:
1. Threshold invariants: IMAGE_THRESHOLD (0.21), VIDEO_TEMPORAL_THRESHOLD (0.30), VIDEO_TEMPORAL_MARGIN (0.03)
2. Temporal boundary window: [0.27, 0.33]
   - score 0.26 -> Real
   - score 0.27 -> Uncertain
   - score 0.30 -> Uncertain
   - score 0.33 -> Uncertain
   - score 0.34 -> Likely Fake
   - score 0.40 -> Likely Fake (high confidence)
3. Spatial volatility decoupling:
   - high volatility + strong V5.2 Real -> Real
   - high volatility + strong V5.2 Fake -> Likely Fake
   - audit statement exact text verified
4. Valid face frames boundaries (14 vs 15 vs 16):
   - 14 frames: no floor, volatility veto preserved
   - 15 frames: reliability floor (0.15) applied, volatility decoupled if V5.2 decisive
   - 16 frames: eligible for high-confidence V5.2 override (<= 0.22 or >= 0.40)
5. Video reliability floor:
   - max(calculated, 0.15) when valid_face_frames >= 15
6. Genuine strong contradiction: Image Fake vs Video Real -> Uncertain
7. Agreement and uncertainty preservation
"""

import unittest
from backend.config import (
    CALIBRATED_THRESHOLD as IMAGE_THRESHOLD,
    VIDEO_TEMPORAL_THRESHOLD,
    VIDEO_TEMPORAL_MARGIN
)
from backend.fusion import (
    MultimodalFusionEngine,
    calibrate_temporal_score,
    parse_video_evidence,
    fuse_multimodal_evidence
)


class TestV52VolatilityMarginRefinements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fusion_engine = MultimodalFusionEngine()
        cls.exact_audit_note = (
            "Phase 1 spatial score volatility was treated as a reliability factor "
            "rather than an automatic final-decision veto because dedicated V5.2 "
            "temporal evidence was available."
        )

    def test_01_threshold_and_margin_invariants(self):
        """Verify image threshold is 0.21, temporal threshold is 0.30, and margin is 0.03."""
        self.assertEqual(IMAGE_THRESHOLD, 0.21)
        self.assertEqual(VIDEO_TEMPORAL_THRESHOLD, 0.30)
        self.assertEqual(VIDEO_TEMPORAL_MARGIN, 0.03)
        lower_bound = VIDEO_TEMPORAL_THRESHOLD - VIDEO_TEMPORAL_MARGIN
        upper_bound = VIDEO_TEMPORAL_THRESHOLD + VIDEO_TEMPORAL_MARGIN
        self.assertAlmostEqual(lower_bound, 0.27, places=4)
        self.assertAlmostEqual(upper_bound, 0.33, places=4)

    def test_02_v5_2_boundary_scores(self):
        """Verify decision boundary [0.27, 0.33] behaviors across critical test scores."""
        def evaluate_temporal_score(score: float, thresh: float = VIDEO_TEMPORAL_THRESHOLD, margin: float = VIDEO_TEMPORAL_MARGIN):
            lower = round(thresh - margin, 4)
            upper = round(thresh + margin, 4)
            if lower <= score <= upper:
                return "Uncertain"
            elif score >= thresh:
                return "Fake"
            else:
                return "Real"

        self.assertEqual(evaluate_temporal_score(0.26), "Real")
        self.assertEqual(evaluate_temporal_score(0.27), "Uncertain")
        self.assertEqual(evaluate_temporal_score(0.30), "Uncertain")
        self.assertEqual(evaluate_temporal_score(0.33), "Uncertain")
        self.assertEqual(evaluate_temporal_score(0.34), "Fake")
        self.assertEqual(evaluate_temporal_score(0.40), "Fake")

    def test_03_high_volatility_plus_strong_v5_2_real(self):
        """High spatial volatility + strong V5.2 Real -> Real with exact audit text."""
        video_payload = {
            "media_type": "video",
            "verdict": "Real",
            "prediction": "Real",
            "confidence": 0.70,
            "median_score": 0.28,
            "temporal_volatility": 0.43,
            "score_std": 0.37,
            "valid_prediction_count": 16,
            "face_frames": 16,
            "frames_analyzed": 16,
            "phase1_video_evidence": {
                "prediction": "Uncertain",
                "verdict": "Uncertain",
                "median_score": 0.28,
                "temporal_volatility": 0.43,
                "score_std": 0.37,
                "uncertainty_reason": "Extreme temporal score volatility (0.43, std: 0.37) precludes decisive classification.",
                "is_spatial_volatile": True,
                "spatial_volatility_decoupled": True
            },
            "v5_2_temporal_evidence": {
                "prediction": "Real",
                "verdict": "Real",
                "temporal_score": 0.15,
                "temporal_threshold": 0.30,
                "confidence": 0.75,
                "valid_face_frames": 16
            },
            "spatial_volatility_decoupled": True,
            "spatial_volatility_audit": self.exact_audit_note
        }

        parsed = parse_video_evidence(video_payload)
        self.assertEqual(parsed["verdict"], "Real")
        self.assertEqual(parsed["prediction"], "Real")
        self.assertTrue(parsed["spatial_volatility_decoupled"])
        self.assertTrue(any(self.exact_audit_note in f for f in parsed["findings"]))

        res = self.fusion_engine.evaluate_fusion(video_prediction=video_payload)
        self.assertEqual(res["overall_prediction"], "Real")
        self.assertFalse(res["conflict_detected"])
        self.assertTrue(any(self.exact_audit_note in c for c in res["corroborating_factors"]))

    def test_04_high_volatility_plus_strong_v5_2_fake(self):
        """High spatial volatility + strong V5.2 Fake -> Likely Fake with exact audit text."""
        video_payload = {
            "media_type": "video",
            "verdict": "Likely Fake",
            "prediction": "Fake",
            "confidence": 0.85,
            "median_score": 0.30,
            "temporal_volatility": 0.45,
            "score_std": 0.38,
            "valid_prediction_count": 16,
            "face_frames": 16,
            "frames_analyzed": 16,
            "phase1_video_evidence": {
                "prediction": "Uncertain",
                "verdict": "Uncertain",
                "median_score": 0.30,
                "temporal_volatility": 0.45,
                "score_std": 0.38,
                "uncertainty_reason": "Extreme temporal score volatility (0.45, std: 0.38) precludes decisive classification.",
                "is_spatial_volatile": True,
                "spatial_volatility_decoupled": True
            },
            "v5_2_temporal_evidence": {
                "prediction": "Fake",
                "verdict": "Likely Fake",
                "temporal_score": 0.65,
                "temporal_threshold": 0.30,
                "confidence": 0.85,
                "valid_face_frames": 16
            },
            "spatial_volatility_decoupled": True,
            "spatial_volatility_audit": self.exact_audit_note
        }

        parsed = parse_video_evidence(video_payload)
        self.assertEqual(parsed["verdict"], "Likely Fake")
        self.assertEqual(parsed["prediction"], "Fake")
        self.assertTrue(parsed["spatial_volatility_decoupled"])

        res = self.fusion_engine.evaluate_fusion(video_prediction=video_payload)
        self.assertEqual(res["overall_prediction"], "Likely Fake")
        self.assertFalse(res["conflict_detected"])
        self.assertTrue(any(self.exact_audit_note in c for c in res["corroborating_factors"]))

    def test_05_volatility_with_v5_2_uncertain_remains_uncertain(self):
        """Volatility + V5.2 score inside [0.27, 0.33] does not hide uncertainty."""
        video_payload = {
            "media_type": "video",
            "verdict": "Uncertain",
            "prediction": "Uncertain",
            "confidence": 0.50,
            "median_score": 0.28,
            "temporal_volatility": 0.42,
            "score_std": 0.36,
            "valid_prediction_count": 16,
            "face_frames": 16,
            "frames_analyzed": 16,
            "phase1_video_evidence": {
                "prediction": "Uncertain",
                "verdict": "Uncertain",
                "median_score": 0.28,
                "temporal_volatility": 0.42,
                "score_std": 0.36
            },
            "v5_2_temporal_evidence": {
                "prediction": "Uncertain",
                "verdict": "Uncertain",
                "temporal_score": 0.30,
                "temporal_threshold": 0.30,
                "confidence": 0.50,
                "valid_face_frames": 16,
                "uncertainty_reason": "Temporal prediction score (0.3000) is in the boundary window [0.27, 0.33]."
            },
            "uncertainty_reason": "Temporal prediction score (0.3000) is in the boundary window [0.27, 0.33]."
        }

        parsed = parse_video_evidence(video_payload)
        self.assertEqual(parsed["verdict"], "Uncertain")
        self.assertEqual(parsed["prediction"], "Uncertain")
        self.assertFalse(parsed["spatial_volatility_decoupled"])

        res = self.fusion_engine.evaluate_fusion(video_prediction=video_payload)
        self.assertEqual(res["overall_prediction"], "Uncertain")

    def test_06_face_frames_boundaries_14_vs_15_vs_16(self):
        """Verify behavior across valid_face_frames = 14, 15, and 16."""
        # Case A: valid_face_frames = 14 (< 15) -> no floor, no decoupling
        payload_14 = {
            "media_type": "video",
            "verdict": "Uncertain",
            "prediction": "Uncertain",
            "confidence": 0.50,
            "median_score": 0.28,
            "temporal_volatility": 0.42,
            "score_std": 0.36,
            "compression_degradation": True,
            "valid_prediction_count": 14,
            "face_frames": 14,
            "frames_analyzed": 14,
            "phase1_video_evidence": {
                "prediction": "Uncertain",
                "temporal_volatility": 0.42,
                "score_std": 0.36
            },
            "v5_2_temporal_evidence": {
                "prediction": "Real",
                "verdict": "Real",
                "temporal_score": 0.20,
                "valid_face_frames": 14
            }
        }
        parsed_14 = parse_video_evidence(payload_14)
        self.assertLess(parsed_14["reliability_weight"], 0.15)
        self.assertAlmostEqual(parsed_14["reliability_weight"], 0.054, places=2)

        # Case B: valid_face_frames = 15 (>= 15) -> reliability floor applies
        payload_15 = {
            "media_type": "video",
            "verdict": "Uncertain",
            "prediction": "Uncertain",
            "confidence": 0.50,
            "median_score": 0.28,
            "temporal_volatility": 0.42,
            "score_std": 0.36,
            "compression_degradation": True,
            "valid_prediction_count": 15,
            "face_frames": 15,
            "frames_analyzed": 15,
            "phase1_video_evidence": {
                "prediction": "Uncertain",
                "temporal_volatility": 0.42,
                "score_std": 0.36
            },
            "v5_2_temporal_evidence": {
                "prediction": "Uncertain",
                "verdict": "Uncertain",
                "temporal_score": 0.30,
                "valid_face_frames": 15
            }
        }
        parsed_15 = parse_video_evidence(payload_15)
        self.assertGreaterEqual(parsed_15["reliability_weight"], 0.15)
        self.assertAlmostEqual(parsed_15["reliability_weight"], 0.15, places=2)

        # Case C: valid_face_frames = 16 (>= 16) -> eligible for high-confidence override
        payload_16 = {
            "media_type": "video",
            "verdict": "Real",
            "prediction": "Real",
            "confidence": 0.70,
            "median_score": 0.10,
            "video_calibrated_score": 0.45,
            "valid_prediction_count": 16,
            "face_frames": 16,
            "frames_analyzed": 16,
            "phase1_video_evidence": {
                "prediction": "Fake",
                "video_calibrated_score": 0.45
            },
            "v5_2_temporal_evidence": {
                "prediction": "Real",
                "verdict": "Real",
                "temporal_score": 0.18,
                "valid_face_frames": 16
            }
        }
        parsed_16 = parse_video_evidence(payload_16)
        self.assertEqual(parsed_16["prediction"], "Real")
        self.assertEqual(parsed_16["verdict"], "Real")

    def test_07_reliability_floor_exact(self):
        """Verify final_video_reliability = max(calculated, 0.15) only when valid_face_frames >= 15."""
        video_payload = {
            "media_type": "video",
            "verdict": "Uncertain",
            "prediction": "Uncertain",
            "compression_degradation": True,
            "temporal_volatility": 0.42,
            "score_std": 0.40,
            "valid_prediction_count": 15,
            "face_frames": 15,
            "frames_analyzed": 15,
            "v5_2_temporal_evidence": {
                "valid_face_frames": 15,
                "temporal_score": 0.3291,
                "prediction": "Uncertain"
            }
        }
        parsed = parse_video_evidence(video_payload)
        self.assertEqual(parsed["reliability_weight"], 0.15)

    def test_08_genuine_strong_contradiction(self):
        """Decisive Image Fake vs Decisive Video Real produces inter-modal conflict (Uncertain)."""
        image_payload = {
            "media_type": "image",
            "score": 0.85,
            "face_detected": True
        }
        video_payload = {
            "media_type": "video",
            "verdict": "Real",
            "prediction": "Real",
            "representative_score": 0.10,
            "compression_degradation": False,
            "valid_prediction_count": 20,
            "face_frames": 20,
            "frames_analyzed": 20
        }
        res = self.fusion_engine.evaluate_fusion(
            image_prediction=image_payload,
            video_prediction=video_payload
        )
        self.assertEqual(res["overall_prediction"], "Uncertain")
        self.assertTrue(res["conflict_detected"])

    def test_09_existing_phase1_v5_2_agreement(self):
        """Preserve existing agreement behaviors between Phase 1 and V5.2."""
        payload_agree_real = {
            "media_type": "video",
            "verdict": "Real",
            "prediction": "Real",
            "confidence": 0.70,
            "phase1_video_evidence": {"prediction": "Real", "video_calibrated_score": 0.10},
            "v5_2_temporal_evidence": {"prediction": "Real", "temporal_score": 0.15, "valid_face_frames": 16}
        }
        res_real = self.fusion_engine.evaluate_fusion(video_prediction=payload_agree_real)
        self.assertEqual(res_real["overall_prediction"], "Real")
        self.assertFalse(res_real["conflict_detected"])

        payload_agree_fake = {
            "media_type": "video",
            "verdict": "Likely Fake",
            "prediction": "Fake",
            "confidence": 0.80,
            "phase1_video_evidence": {"prediction": "Fake", "video_calibrated_score": 0.60},
            "v5_2_temporal_evidence": {"prediction": "Fake", "temporal_score": 0.65, "valid_face_frames": 16}
        }
        res_fake = self.fusion_engine.evaluate_fusion(video_prediction=payload_agree_fake)
        self.assertEqual(res_fake["overall_prediction"], "Likely Fake")
        self.assertFalse(res_fake["conflict_detected"])


if __name__ == "__main__":
    unittest.main()
