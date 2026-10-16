"""
DeepShield V5 - Multimodal Evidence Fusion Test Suite
=====================================================
Comprehensive tests for Adaptive Multimodal Evidence Fusion in backend/fusion.py:
1. Image only (Real & Fake)
2. Video only (Real, Likely Fake, Uncertain/Degraded)
3. Image + Metadata (Corroborating & Conflicting)
4. Image + Video (Concordant Real & Concordant Fake)
5. Image + Video (Conflicting: Image Fake vs Video Real)
6. Image + Video + Metadata (Tri-modal fusion)
7. Conflicting evidence (yielding Uncertain)
8. Insufficient evidence (yielding Uncertain)
9. Degraded video adaptive downweighting (clean image preserved)
10. Degraded video only (yielding Uncertain)
"""

import unittest
from backend.fusion import MultimodalFusionEngine, fuse_multimodal_evidence
from backend.config import CALIBRATED_THRESHOLD


class TestMultimodalEvidenceFusion(unittest.TestCase):
    """Test suite for adaptive multimodal evidence fusion and conflict detection."""

    def test_01_image_only_real(self):
        """Image-only evaluation with authentic score (< 0.21)."""
        image_data = {
            "score": 0.03,
            "prediction": "Real",
            "confidence": 0.92,
            "face_detected": True,
            "face_count": 1
        }
        res = fuse_multimodal_evidence(image_evidence=image_data)
        self.assertEqual(res["overall_prediction"], "Real")
        self.assertEqual(res["risk_level"], "Low")
        self.assertFalse(res["conflict_detected"])
        self.assertIsNone(res["uncertainty_reason"])
        self.assertTrue(res["image_evidence"]["available"])
        self.assertFalse(res["video_evidence"]["available"])
        self.assertIn("This is an AI-based assessment", res["explanation"])

    def test_02_image_only_fake(self):
        """Image-only evaluation with manipulation score (>= 0.21)."""
        image_data = {
            "score": 0.88,
            "prediction": "Fake",
            "confidence": 0.93,
            "face_detected": True,
            "face_count": 1
        }
        res = fuse_multimodal_evidence(image_evidence=image_data)
        self.assertEqual(res["overall_prediction"], "Likely Fake")
        self.assertEqual(res["risk_level"], "Critical")
        self.assertFalse(res["conflict_detected"])
        self.assertGreater(res["confidence"], 0.80)

    def test_03_video_only_real(self):
        """Video-only evaluation with clean authentic stream."""
        video_data = {
            "verdict": "Real",
            "prediction": "Real",
            "median_score": 0.005,
            "p75_score": 0.010,
            "temporal_volatility": 0.005,
            "persistence_ratio": 0.0,
            "compression_degradation": False,
            "confidence": 0.98,
            "frames_analyzed": 20
        }
        res = fuse_multimodal_evidence(video_evidence=video_data)
        self.assertEqual(res["overall_prediction"], "Real")
        self.assertEqual(res["risk_level"], "Low")
        self.assertFalse(res["conflict_detected"])
        self.assertTrue(res["video_evidence"]["available"])
        self.assertFalse(res["image_evidence"]["available"])

    def test_04_video_only_fake(self):
        """Video-only evaluation with persistent manipulation patterns."""
        video_data = {
            "verdict": "Likely Fake",
            "prediction": "Fake",
            "median_score": 0.92,
            "p75_score": 0.95,
            "temporal_volatility": 0.08,
            "persistence_ratio": 0.95,
            "compression_degradation": False,
            "confidence": 0.96,
            "frames_analyzed": 25
        }
        res = fuse_multimodal_evidence(video_evidence=video_data)
        self.assertEqual(res["overall_prediction"], "Likely Fake")
        self.assertEqual(res["risk_level"], "Critical")
        self.assertFalse(res["conflict_detected"])

    def test_05_video_only_degraded_uncertain(self):
        """Degraded/volatile video alone must yield Uncertain."""
        video_data = {
            "verdict": "Uncertain",
            "prediction": "Uncertain",
            "median_score": 0.02,
            "p75_score": 0.04,
            "temporal_volatility": 0.02,
            "compression_degradation": True,
            "uncertainty_reason": "Video compression degradation detected (bitrate: 650 kbps, blur: 40)",
            "confidence": 0.50,
            "frames_analyzed": 30
        }
        res = fuse_multimodal_evidence(video_evidence=video_data)
        self.assertEqual(res["overall_prediction"], "Uncertain")
        self.assertEqual(res["confidence"], 0.50)
        self.assertIsNotNone(res["uncertainty_reason"])
        self.assertIn("compression degradation", res["uncertainty_reason"])

    def test_06_image_plus_metadata_corroborating(self):
        """Image fake corroborated by software manipulation metadata."""
        image_data = {
            "score": 0.91,
            "prediction": "Fake",
            "confidence": 0.95,
            "face_detected": True
        }
        meta_data = {
            "editing_software_detected": True,
            "editing_software_name": "Midjourney v6",
            "is_ai_generator_indicated": True,
            "has_exif": True
        }
        res = fuse_multimodal_evidence(image_evidence=image_data, metadata_evidence=meta_data)
        self.assertEqual(res["overall_prediction"], "Likely Fake")
        self.assertFalse(res["conflict_detected"])
        self.assertTrue(len(res["corroborating_factors"]) > 0)
        self.assertIn("Midjourney", res["corroborating_factors"][0])

    def test_07_image_plus_metadata_conflict(self):
        """Image Real but metadata indicates AI generative software (conflict noted)."""
        image_data = {
            "score": 0.05,
            "prediction": "Real",
            "confidence": 0.90,
            "face_detected": True
        }
        meta_data = {
            "editing_software_detected": True,
            "editing_software_name": "Stable Diffusion WebUI",
            "is_ai_generator_indicated": True,
            "has_exif": True
        }
        res = fuse_multimodal_evidence(image_evidence=image_data, metadata_evidence=meta_data)
        self.assertTrue(len(res["conflict_factors"]) > 0)
        self.assertIn("Stable Diffusion", res["conflict_factors"][0])

    def test_08_image_plus_video_concordant(self):
        """Concordant multimodal package: both image and video indicate Fake."""
        image_data = {
            "score": 0.85,
            "prediction": "Fake",
            "confidence": 0.91,
            "face_detected": True
        }
        video_data = {
            "verdict": "Likely Fake",
            "prediction": "Fake",
            "median_score": 0.88,
            "p75_score": 0.92,
            "temporal_volatility": 0.05,
            "persistence_ratio": 1.0,
            "compression_degradation": False,
            "confidence": 0.94
        }
        res = fuse_multimodal_evidence(image_evidence=image_data, video_evidence=video_data)
        self.assertEqual(res["overall_prediction"], "Likely Fake")
        self.assertFalse(res["conflict_detected"])
        self.assertTrue(len(res["corroborating_factors"]) > 0)
        self.assertGreater(res["confidence"], 0.85)

    def test_09_image_plus_video_conflict_yields_uncertain(self):
        """Strong inter-modal conflict: Image=Fake vs un-degraded Video=Real must yield Uncertain."""
        image_data = {
            "score": 0.90,
            "prediction": "Fake",
            "confidence": 0.94,
            "face_detected": True
        }
        video_data = {
            "verdict": "Real",
            "prediction": "Real",
            "median_score": 0.01,
            "p75_score": 0.02,
            "temporal_volatility": 0.01,
            "persistence_ratio": 0.0,
            "compression_degradation": False,
            "confidence": 0.95
        }
        res = fuse_multimodal_evidence(image_evidence=image_data, video_evidence=video_data)
        self.assertEqual(res["overall_prediction"], "Uncertain")
        self.assertEqual(res["confidence"], 0.50)
        self.assertTrue(res["conflict_detected"])
        self.assertTrue(len(res["conflict_factors"]) > 0)
        self.assertIn("Significant inter-modal conflict", res["uncertainty_reason"])

    def test_10_insufficient_evidence_yields_uncertain(self):
        """When no valid evidence streams are available, must return Uncertain."""
        res = fuse_multimodal_evidence(image_evidence=None, video_evidence=None, metadata_evidence=None)
        self.assertEqual(res["overall_prediction"], "Uncertain")
        self.assertEqual(res["confidence"], 0.50)
        self.assertIsNotNone(res["uncertainty_reason"])
        self.assertIn("Insufficient forensic evidence", res["uncertainty_reason"])

    def test_11_degraded_video_downweighted_preserving_clean_image(self):
        """
        When video has severe compression degradation, its reliability weight is reduced
        so it does not falsely drag down clean decisive image evidence.
        """
        image_data = {
            "score": 0.92,
            "prediction": "Fake",
            "confidence": 0.95,
            "face_detected": True
        }
        # Video smoothed out by low bitrate compression
        video_data = {
            "verdict": "Uncertain",
            "prediction": "Uncertain",
            "median_score": 0.03,
            "p75_score": 0.05,
            "temporal_volatility": 0.03,
            "compression_degradation": True,
            "uncertainty_reason": "Video compression degradation detected",
            "confidence": 0.50
        }
        res = fuse_multimodal_evidence(image_evidence=image_data, video_evidence=video_data)
        # Video weight is penalised (0.9 * 0.2 * 0.3 = 0.054) vs Image weight (1.0)
        # Fused score remains decisively above threshold (> 0.70)
        self.assertGreater(res["fusion_score"], CALIBRATED_THRESHOLD)
        self.assertEqual(res["overall_prediction"], "Likely Fake")
        self.assertTrue(res["video_evidence"]["compression_degradation"])

    def test_12_trimodal_fusion(self):
        """Full tri-modal fusion combining image, video, and camera hardware metadata."""
        image_data = {
            "score": 0.02,
            "prediction": "Real",
            "confidence": 0.95,
            "face_detected": True
        }
        video_data = {
            "verdict": "Real",
            "prediction": "Real",
            "median_score": 0.015,
            "p75_score": 0.025,
            "temporal_volatility": 0.01,
            "compression_degradation": False,
            "confidence": 0.96
        }
        meta_data = {
            "camera_make": "Sony",
            "camera_model": "ILCE-7RM4",
            "has_exif": True,
            "editing_software_detected": False
        }
        res = MultimodalFusionEngine.evaluate_fusion(
            image_prediction=image_data,
            video_prediction=video_data,
            metadata_analysis=meta_data
        )
        self.assertEqual(res["overall_prediction"], "Real")
        self.assertEqual(res["risk_level"], "Low")
        self.assertFalse(res["conflict_detected"])
        self.assertTrue(len(res["corroborating_factors"]) >= 2)

    def test_13_v5_2_and_phase1_agreement_reinforces(self):
        """When Phase 1 and V5.2 agree, corroboration is recorded and verdict reinforced."""
        video_data = {
            "verdict": "Real",
            "prediction": "Real",
            "confidence": 0.90,
            "phase1_video_evidence": {
                "prediction": "Real",
                "video_calibrated_score": 0.05
            },
            "v5_2_temporal_evidence": {
                "prediction": "Real",
                "temporal_model": "DeepShield V5.2",
                "temporal_score": 0.042,
                "temporal_threshold": 0.30,
                "confidence": 0.94
            }
        }
        res = fuse_multimodal_evidence(video_evidence=video_data)
        self.assertEqual(res["overall_prediction"], "Real")
        self.assertFalse(res["conflict_detected"])
        self.assertTrue(any("Temporal model (V5.2)" in f for f in res["corroborating_factors"]))

    def test_14_v5_2_and_phase1_disagreement_yields_uncertain(self):
        """When Phase 1 and V5.2 disagree (e.g. Real vs Fake), conflict is detected and yields Uncertain."""
        video_data = {
            "verdict": "Uncertain",
            "prediction": "Uncertain",
            "confidence": 0.50,
            "phase1_video_evidence": {
                "prediction": "Real",
                "video_calibrated_score": 0.08
            },
            "v5_2_temporal_evidence": {
                "prediction": "Fake",
                "temporal_model": "DeepShield V5.2",
                "temporal_score": 0.88,
                "temporal_threshold": 0.30,
                "confidence": 0.95
            }
        }
        res = fuse_multimodal_evidence(video_evidence=video_data)
        self.assertEqual(res["overall_prediction"], "Uncertain")
        self.assertEqual(res["confidence"], 0.50)
        self.assertTrue(res["conflict_detected"])
        self.assertTrue(any("Temporal vs Spatial Video Conflict" in f for f in res["conflict_factors"]))

    def test_15_v5_2_temporal_standalone_fusion(self):
        """Dedicated V5.2 temporal evidence alone evaluates correctly through MultimodalFusionEngine."""
        temporal_data = {
            "temporal_model": "DeepShield V5.2",
            "prediction": "Fake",
            "verdict": "Likely Fake",
            "temporal_score": 0.82,
            "temporal_threshold": 0.30,
            "confidence": 0.92,
            "valid_face_frames": 16,
            "total_frames": 120
        }
        res = MultimodalFusionEngine.evaluate_fusion(temporal_prediction=temporal_data)
        self.assertEqual(res["overall_prediction"], "Likely Fake")
        self.assertFalse(res["conflict_detected"])
        self.assertTrue(res["video_evidence"]["available"])
        self.assertEqual(res["video_evidence"]["temporal_model"], "DeepShield V5.2")


if __name__ == "__main__":
    unittest.main()
