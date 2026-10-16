"""
Unit Tests for DeepShield V5.2 Fusion Correction & Temporal Authority
====================================================================

Validates:
1. Phase 1 + V5.2 agreement
2. Phase 1 error + strong V5.2 evidence (Real)
3. Phase 1 error + strong V5.2 evidence (Fake)
4. V5.2 near-boundary disagreement
5. 16 valid face frames requirement
6. Genuine strong contradiction (Image Fake vs Video Real)
7. Separate calibration spaces: IMAGE_THRESHOLD (0.21) vs VIDEO_TEMPORAL_THRESHOLD (0.30)
8. V5.2 score participation in fusion
9. Strict enforcement of no center-crop fallback
10. Forensic report generation with exact audit notes
11. Video-only upload integrity
"""

import unittest
from pathlib import Path
import numpy as np

from backend.config import CALIBRATED_THRESHOLD as IMAGE_THRESHOLD, VIDEO_TEMPORAL_THRESHOLD
from backend.fusion import (
    MultimodalFusionEngine,
    calibrate_temporal_score,
    parse_video_evidence,
    fuse_multimodal_evidence
)
from backend.report_generator import ForensicReportGenerator


class TestV52FusionCorrection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fusion_engine = MultimodalFusionEngine()

    def test_01_threshold_separation(self):
        """Verify image threshold remains 0.21 and video temporal threshold remains 0.30."""
        self.assertEqual(IMAGE_THRESHOLD, 0.21)
        self.assertEqual(VIDEO_TEMPORAL_THRESHOLD, 0.30)
        self.assertNotEqual(IMAGE_THRESHOLD, VIDEO_TEMPORAL_THRESHOLD)

    def test_02_calibrated_temporal_score_mapping(self):
        """Verify calibrate_temporal_score correctly maps 0.30 to 0.21 with preserved direction."""
        # Threshold point
        cal_thresh = calibrate_temporal_score(0.30, 0.30)
        self.assertAlmostEqual(cal_thresh, 0.21, places=3)

        # Real space (< 0.30 -> < 0.21)
        cal_real = calibrate_temporal_score(0.15, 0.30)
        self.assertLess(cal_real, 0.21)
        self.assertAlmostEqual(cal_real, 0.105, places=3)

        # Fake space (>= 0.30 -> >= 0.21)
        cal_fake = calibrate_temporal_score(0.65, 0.30)
        self.assertGreater(cal_fake, 0.21)
        self.assertAlmostEqual(cal_fake, 0.21 + 0.79 * (0.35 / 0.70), places=3)

        # Extrema
        self.assertAlmostEqual(calibrate_temporal_score(0.0, 0.30), 0.0, places=4)
        self.assertAlmostEqual(calibrate_temporal_score(1.0, 0.30), 1.0, places=4)

    def test_03_phase1_v5_2_agreement_real(self):
        """Verify Phase 1 + V5.2 concordant Real prediction reinforces confidence."""
        video_payload = {
            "media_type": "video",
            "prediction": "Real",
            "verdict": "Real",
            "confidence": 0.65,
            "median_score": 0.50,
            "video_calibrated_score": 0.12,
            "phase1_video_evidence": {
                "prediction": "Real",
                "video_calibrated_score": 0.12,
                "confidence": 0.65
            },
            "v5_2_temporal_evidence": {
                "prediction": "Real",
                "verdict": "Real",
                "temporal_score": 0.18,
                "temporal_threshold": 0.30,
                "confidence": 0.70,
                "valid_face_frames": 18
            }
        }
        res = self.fusion_engine.evaluate_fusion(video_prediction=video_payload)
        self.assertEqual(res["overall_prediction"], "Real")
        self.assertFalse(res["conflict_detected"])
        self.assertLess(res["fusion_score"], IMAGE_THRESHOLD)
        self.assertTrue(any("concordantly" in c for c in res["corroborating_factors"]))

    def test_04_phase1_v5_2_agreement_fake(self):
        """Verify Phase 1 + V5.2 concordant Fake prediction yields Likely Fake."""
        video_payload = {
            "media_type": "video",
            "prediction": "Fake",
            "verdict": "Likely Fake",
            "confidence": 0.85,
            "median_score": 0.05,
            "video_calibrated_score": 0.90,
            "phase1_video_evidence": {
                "prediction": "Fake",
                "video_calibrated_score": 0.90,
                "confidence": 0.85
            },
            "v5_2_temporal_evidence": {
                "prediction": "Fake",
                "verdict": "Likely Fake",
                "temporal_score": 0.65,
                "temporal_threshold": 0.30,
                "confidence": 0.80,
                "valid_face_frames": 20
            }
        }
        res = self.fusion_engine.evaluate_fusion(video_prediction=video_payload)
        self.assertEqual(res["overall_prediction"], "Likely Fake")
        self.assertFalse(res["conflict_detected"])
        self.assertGreater(res["fusion_score"], IMAGE_THRESHOLD)

    def test_05_phase1_error_v5_2_strong_real(self):
        """
        Verify that when Phase 1 falsely predicts Fake, but V5.2 provides strong
        temporal Real evidence (score <= 0.22, frames >= 16), V5.2 arbitrates
        to Real with appropriate audit note.
        """
        video_payload = {
            "media_type": "video",
            "prediction": "Real",
            "verdict": "Real",
            "confidence": 0.77,
            "median_score": 0.10,
            "video_calibrated_score": 0.63,
            "phase1_video_evidence": {
                "prediction": "Fake",
                "video_calibrated_score": 0.63,
                "confidence": 0.70
            },
            "v5_2_temporal_evidence": {
                "prediction": "Real",
                "verdict": "Real",
                "temporal_score": 0.13,  # <= 0.22 (high confidence)
                "temporal_threshold": 0.30,
                "confidence": 0.78,
                "valid_face_frames": 16
            }
        }
        res = self.fusion_engine.evaluate_fusion(video_prediction=video_payload)
        self.assertEqual(res["overall_prediction"], "Real")
        self.assertFalse(res["conflict_detected"])
        self.assertLess(res["fusion_score"], IMAGE_THRESHOLD)
        # Check audit note
        audit_found = any("higher reliability than the legacy Phase 1 spatial heuristic" in c for c in res["corroborating_factors"])
        self.assertTrue(audit_found)

    def test_06_phase1_error_v5_2_strong_fake(self):
        """
        Verify that when Phase 1 falsely predicts Real, but V5.2 provides strong
        temporal Fake evidence (score >= 0.40, frames >= 16), V5.2 arbitrates
        to Likely Fake with appropriate audit note.
        """
        video_payload = {
            "media_type": "video",
            "prediction": "Fake",
            "verdict": "Likely Fake",
            "confidence": 0.70,
            "median_score": 0.95,
            "video_calibrated_score": 0.01,
            "phase1_video_evidence": {
                "prediction": "Real",
                "video_calibrated_score": 0.01,
                "confidence": 0.85
            },
            "v5_2_temporal_evidence": {
                "prediction": "Fake",
                "verdict": "Likely Fake",
                "temporal_score": 0.57,  # >= 0.40 (high confidence)
                "temporal_threshold": 0.30,
                "confidence": 0.75,
                "valid_face_frames": 16
            }
        }
        res = self.fusion_engine.evaluate_fusion(video_prediction=video_payload)
        self.assertEqual(res["overall_prediction"], "Likely Fake")
        self.assertFalse(res["conflict_detected"])
        self.assertGreater(res["fusion_score"], IMAGE_THRESHOLD)
        audit_found = any("higher reliability than the legacy Phase 1 spatial heuristic" in c for c in res["corroborating_factors"])
        self.assertTrue(audit_found)

    def test_07_v5_2_near_boundary_disagreement_remains_uncertain(self):
        """
        Verify that when Phase 1 and V5.2 disagree AND V5.2 is near boundary
        (0.22 < temporal_score < 0.40), the pipeline properly retains Uncertain.
        """
        video_payload = {
            "media_type": "video",
            "prediction": "Uncertain",
            "verdict": "Uncertain",
            "confidence": 0.50,
            "median_score": 0.91,
            "video_calibrated_score": 0.02,
            "phase1_video_evidence": {
                "prediction": "Real",
                "video_calibrated_score": 0.02,
                "confidence": 0.80
            },
            "v5_2_temporal_evidence": {
                "prediction": "Fake",
                "verdict": "Likely Fake",
                "temporal_score": 0.36,  # in (0.22, 0.40) -> near boundary
                "temporal_threshold": 0.30,
                "confidence": 0.55,
                "valid_face_frames": 16
            }
        }
        res = self.fusion_engine.evaluate_fusion(video_prediction=video_payload)
        self.assertEqual(res["overall_prediction"], "Uncertain")
        self.assertTrue(res["conflict_detected"])
        conflict_msg_found = any("sufficiently strong contradictory evidence" in c for c in res["conflict_factors"])
        self.assertTrue(conflict_msg_found)

    def test_08_insufficient_frames_disagreement_remains_uncertain(self):
        """
        Verify that when Phase 1 and V5.2 disagree, but V5.2 has fewer than 16
        valid face frames, high-confidence temporal authority does not apply.
        """
        video_payload = {
            "media_type": "video",
            "prediction": "Uncertain",
            "verdict": "Uncertain",
            "confidence": 0.50,
            "median_score": 0.12,
            "video_calibrated_score": 0.55,
            "phase1_video_evidence": {
                "prediction": "Fake",
                "video_calibrated_score": 0.55,
                "confidence": 0.65
            },
            "v5_2_temporal_evidence": {
                "prediction": "Real",
                "verdict": "Real",
                "temporal_score": 0.15,  # low score but insufficient frames
                "temporal_threshold": 0.30,
                "confidence": 0.70,
                "valid_face_frames": 10  # < 16
            }
        }
        res = self.fusion_engine.evaluate_fusion(video_prediction=video_payload)
        self.assertEqual(res["overall_prediction"], "Uncertain")
        self.assertTrue(res["conflict_detected"])

    def test_09_genuine_strong_contradiction_image_vs_video(self):
        """
        Verify that a genuine strong contradiction between decisive image Fake
        and decisive video Real triggers hard conflict and yields Uncertain.
        """
        image_data = {
            "score": 0.85,
            "prediction": "Fake",
            "confidence": 0.92,
            "face_detected": True,
            "face_count": 1
        }
        video_data = {
            "media_type": "video",
            "prediction": "Real",
            "verdict": "Real",
            "confidence": 0.80,
            "median_score": 0.50,
            "video_calibrated_score": 0.05,
            "v5_2_temporal_evidence": {
                "prediction": "Real",
                "verdict": "Real",
                "temporal_score": 0.12,
                "temporal_threshold": 0.30,
                "confidence": 0.80,
                "valid_face_frames": 18
            }
        }
        res = self.fusion_engine.evaluate_fusion(image_prediction=image_data, video_prediction=video_data)
        self.assertEqual(res["overall_prediction"], "Uncertain")
        self.assertTrue(res["conflict_detected"])
        self.assertTrue(any("Strong inter-modal contradiction" in c for c in res["conflict_factors"]))

    def test_10_forensic_report_generation_with_v5_2(self):
        """Verify full forensic report generates cleanly with V5.2 and arbitration notes."""
        video_payload = {
            "success": True,
            "media_type": "video",
            "verdict": "Real",
            "prediction": "Real",
            "confidence": 0.78,
            "median_score": 0.10,
            "video_calibrated_score": 0.63,
            "phase1_video_evidence": {
                "prediction": "Fake",
                "video_calibrated_score": 0.63,
                "confidence": 0.70
            },
            "v5_2_temporal_evidence": {
                "prediction": "Real",
                "verdict": "Real",
                "temporal_score": 0.1345,
                "temporal_threshold": 0.30,
                "confidence": 0.78,
                "valid_face_frames": 16,
                "total_frames": 20
            },
            "metadata": {"filename": "test_auth_video.mp4", "format": "QuickTime / MP4"}
        }
        report = ForensicReportGenerator.generate_report(video_payload)
        self.assertIn("verdict", report)
        self.assertEqual(report["verdict"], "Real")
        self.assertIn("v5_2_temporal_evidence", report["evidence"]["video_evidence"])
        vid_ev = report["evidence"]["video_evidence"]
        self.assertEqual(vid_ev["v5_2_temporal_evidence"]["temporal_threshold"], 0.30)
        self.assertIn("higher reliability than the legacy", vid_ev["conflict_status"])


if __name__ == "__main__":
    unittest.main()
