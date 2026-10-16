"""
DeepShield V5 - Structured Forensic Report Integration Test Suite
==================================================================
Comprehensive test suite verifying forensic report compilation, multimodal evidence synthesis,
uncertainty boundaries, backward compatibility, and serialization across:
1. Authentic media report (Real)
2. Manipulated media report (Likely Fake)
3. Borderline/uncertain media report (Uncertain)
4. Image-only evidence report
5. Video-only evidence report
6. Image + Metadata evidence report
7. Image + Video evidence report
8. Tri-modal (Image + Video + Metadata) evidence report
9. Conflicting evidence detection report
10. Compression degraded video adaptive downweighting report
11. Sparse / missing evidence graceful handling
12. JSON and Markdown serialization integrity
"""

import json
import unittest
from backend.report_generator import ForensicReportGenerator
from backend.fusion import MultimodalFusionEngine, fuse_multimodal_evidence
from backend.config import CALIBRATED_THRESHOLD


class TestForensicReportGenerator(unittest.TestCase):
    """Unit tests for ForensicReportGenerator."""

    def setUp(self):
        self.maxDiff = None

    def test_01_real_report(self):
        """Verifies report generation for authentic/real media with proper qualified wording."""
        pred = {
            "score": 0.035,
            "prediction": "Real",
            "confidence": 0.91,
            "face_detected": True,
            "face_count": 1,
            "metadata": {
                "format": "JPEG",
                "file_size": 250000,
                "dimensions": "1080x1080",
                "has_exif": True,
                "camera_make": "Apple",
                "camera_model": "iPhone 13"
            }
        }
        report = ForensicReportGenerator.generate_report(pred, report_id="DSR-TEST-REAL")

        # Top-level required keys
        expected_keys = [
            "report_id", "timestamp", "verdict", "confidence", "risk_level",
            "summary", "evidence", "uncertainty", "technical_details",
            "report_text", "visualization_paths", "sections"
        ]
        for key in expected_keys:
            self.assertIn(key, report, f"Missing required top-level key: {key}")

        self.assertEqual(report["report_id"], "DSR-TEST-REAL")
        self.assertEqual(report["verdict"], "Real")
        self.assertEqual(report["risk_level"], "Low")
        self.assertGreater(report["confidence"], 0.70)

        # Qualified statements
        summary = report["summary"]
        self.assertIn("The available evidence is more consistent with authentic media.", summary["why_result"])
        self.assertIn("This is an AI-based assessment, not absolute proof of authenticity.", summary["responsible_ai_statement"])

        # Backward compatibility sections
        self.assertIn("1_executive_summary", report["sections"])
        self.assertIn("10_recommended_actions", report["sections"])

    def test_02_fake_report(self):
        """Verifies report generation for manipulated media with proper qualified wording."""
        pred = {
            "score": 0.89,
            "prediction": "Fake",
            "confidence": 0.94,
            "face_detected": True,
            "face_count": 1,
            "gradcam": {
                "available": True,
                "target_layer": "efficientnetb2/top_conv",
                "salient_ratio": 0.28,
                "peak_intensity": 0.85
            }
        }
        report = ForensicReportGenerator.generate_report(pred, report_id="DSR-TEST-FAKE")

        self.assertIn(report["verdict"], ["Likely Fake", "Fake"])
        self.assertIn(report["risk_level"], ["High", "Critical"])
        self.assertGreater(report["confidence"], 0.80)

        summary = report["summary"]
        self.assertIn("The available evidence contains patterns associated with manipulated or synthetic media.", summary["why_result"])
        self.assertIn("This is an AI-based assessment, not absolute proof of authenticity.", summary["responsible_ai_statement"])

        # Grad-CAM explanation must not claim absolute proof
        gradcam_exp = report["evidence"]["image_evidence"]["gradcam"]["explanation"]
        self.assertIn("influenc", gradcam_exp.lower())

    def test_03_uncertain_report(self):
        """Verifies report generation for uncertain prediction within decision boundary."""
        pred = {
            "score": 0.215,  # Within [0.16, 0.26]
            "prediction": "Uncertain",
            "confidence": 0.52,
            "face_detected": True,
            "face_count": 1,
            "uncertainty_reason": "Score 0.2150 falls within forensic decision boundary window [0.16, 0.26]."
        }
        report = ForensicReportGenerator.generate_report(pred, report_id="DSR-TEST-UNCERTAIN")

        self.assertEqual(report["verdict"], "Uncertain")
        self.assertEqual(report["risk_level"], "Moderate")
        self.assertTrue(report["uncertainty"]["is_uncertain"])
        self.assertTrue(report["uncertainty"]["is_within_boundary_window"])
        self.assertIsNotNone(report["uncertainty"]["uncertainty_reason"])

        summary = report["summary"]
        self.assertIn("does not force a binary decision", summary["why_result"])

    def test_04_image_only_report(self):
        """Verifies report with only image evidence present."""
        pred = {
            "score": 0.05,
            "prediction": "Real",
            "confidence": 0.88,
            "face_detected": True,
            "face_count": 1
        }
        report = ForensicReportGenerator.generate_report(pred)

        img_ev = report["evidence"]["image_evidence"]
        vid_ev = report["evidence"]["video_evidence"]
        meta_ev = report["evidence"]["metadata_evidence"]

        self.assertTrue(img_ev["evidence_available"])
        self.assertFalse(vid_ev["evidence_available"])
        self.assertFalse(meta_ev["evidence_available"])
        self.assertIn("image", report["multimodal_fusion"]["modalities_available"])
        self.assertNotIn("video", report["multimodal_fusion"]["modalities_available"])

    def test_05_video_only_report(self):
        """Verifies report with only video evidence present."""
        pred = {
            "prediction": "Real",
            "confidence": 0.85,
            "frames_analyzed": 35,
            "faces_detected": 35,
            "median_score": 0.045,
            "mean_score": 0.051,
            "p75_score": 0.060,
            "temporal_volatility": 0.012,
            "persistence_ratio": 0.0,
            "resolution": "1280x720",
            "fps": 30.0,
            "duration": 12.5,
            "compression_degraded": False
        }
        report = ForensicReportGenerator.generate_report(pred)

        img_ev = report["evidence"]["image_evidence"]
        vid_ev = report["evidence"]["video_evidence"]

        self.assertFalse(img_ev["evidence_available"])
        self.assertTrue(vid_ev["evidence_available"])
        self.assertEqual(vid_ev["frames_analyzed"], 35)
        self.assertEqual(vid_ev["median_score"], 0.045)
        self.assertEqual(vid_ev["resolution"], "1280x720")
        self.assertIn("video", report["multimodal_fusion"]["modalities_available"])

    def test_06_image_plus_metadata_report(self):
        """Verifies report with image and metadata evidence."""
        pred = {
            "score": 0.02,
            "prediction": "Real",
            "confidence": 0.95,
            "face_detected": True,
            "metadata": {
                "format": "PNG",
                "file_size": 1500000,
                "dimensions": "1920x1080",
                "has_exif": True,
                "camera_make": "Sony",
                "camera_model": "A7 IV",
                "editing_software_detected": False
            }
        }
        report = ForensicReportGenerator.generate_report(pred)

        self.assertTrue(report["evidence"]["image_evidence"]["evidence_available"])
        self.assertTrue(report["evidence"]["metadata_evidence"]["evidence_available"])
        self.assertEqual(report["evidence"]["metadata_evidence"]["camera_hardware"]["make"], "Sony")
        self.assertIn("metadata", report["multimodal_fusion"]["modalities_available"])

    def test_07_image_plus_video_report(self):
        """Verifies report with combined image and video evidence."""
        fusion = fuse_multimodal_evidence(
            image_evidence={"score": 0.04, "prediction": "Real", "confidence": 0.90, "face_detected": True},
            video_evidence={"prediction": "Real", "confidence": 0.88, "median_score": 0.03, "frames_analyzed": 20, "temporal_volatility": 0.01}
        )
        report = ForensicReportGenerator.generate_report(fusion)

        self.assertTrue(report["evidence"]["image_evidence"]["evidence_available"])
        self.assertTrue(report["evidence"]["video_evidence"]["evidence_available"])
        self.assertEqual(report["verdict"], "Real")
        self.assertIn("image", report["multimodal_fusion"]["modalities_available"])
        self.assertIn("video", report["multimodal_fusion"]["modalities_available"])

    def test_08_trimodal_report(self):
        """Verifies tri-modal report (image, video, and metadata evidence)."""
        fusion = fuse_multimodal_evidence(
            image_evidence={"score": 0.92, "prediction": "Fake", "confidence": 0.95, "face_detected": True},
            video_evidence={"prediction": "Likely Fake", "confidence": 0.91, "median_score": 0.88, "frames_analyzed": 25},
            metadata_evidence={"software": "Adobe Photoshop 2024", "has_exif": True, "file_format": "MP4"}
        )
        report = ForensicReportGenerator.generate_report(fusion)

        self.assertTrue(report["evidence"]["image_evidence"]["evidence_available"])
        self.assertTrue(report["evidence"]["video_evidence"]["evidence_available"])
        self.assertTrue(report["evidence"]["metadata_evidence"]["evidence_available"])
        self.assertEqual(len(report["multimodal_fusion"]["modalities_available"]), 3)
        self.assertIn(report["verdict"], ["Likely Fake", "Fake"])

    def test_09_conflict_report(self):
        """Verifies report when conflict is detected between modalities."""
        fusion = fuse_multimodal_evidence(
            image_evidence={"score": 0.92, "prediction": "Fake", "confidence": 0.94, "face_detected": True},
            video_evidence={"prediction": "Real", "confidence": 0.89, "median_score": 0.02, "frames_analyzed": 20}
        )
        report = ForensicReportGenerator.generate_report(fusion)

        self.assertTrue(report["multimodal_fusion"]["conflict_detected"])
        self.assertGreater(len(report["multimodal_fusion"]["conflict_factors"]), 0)
        self.assertEqual(report["verdict"], "Uncertain")

    def test_10_degraded_video_report(self):
        """Verifies report when video is flagged as compression degraded."""
        fusion = fuse_multimodal_evidence(
            image_evidence={"score": 0.02, "prediction": "Real", "confidence": 0.92, "face_detected": True},
            video_evidence={
                "prediction": "Uncertain",
                "median_score": 0.22,
                "frames_analyzed": 20,
                "compression_degraded": True,
                "blur_score": 12.4
            }
        )
        report = ForensicReportGenerator.generate_report(fusion)

        vid_ev = report["evidence"]["video_evidence"]
        self.assertTrue(vid_ev["compression_degradation"])
        rel_weights = report["multimodal_fusion"]["reliability_weights"]
        self.assertTrue(rel_weights["video_degraded_downweighted"])
        self.assertLess(rel_weights["video_weight"], rel_weights["image_weight"])

    def test_11_missing_evidence_handling(self):
        """Verifies report generator handles empty/sparse dictionaries without crashing."""
        sparse_pred = {}
        report = ForensicReportGenerator.generate_report(sparse_pred)

        self.assertIn("report_id", report)
        self.assertIn("verdict", report)
        self.assertIn("report_text", report)
        self.assertIsInstance(report["sections"], dict)
        self.assertFalse(report["evidence"]["image_evidence"]["evidence_available"])
        self.assertFalse(report["evidence"]["video_evidence"]["evidence_available"])

    def test_12_json_and_markdown_serialization(self):
        """Verifies report is JSON serializable and markdown contains all required headers."""
        pred = {
            "score": 0.04,
            "prediction": "Real",
            "confidence": 0.90,
            "face_detected": True,
            "face_count": 1,
            "metadata": {"has_exif": True, "format": "JPEG"}
        }
        report = ForensicReportGenerator.generate_report(pred)

        # 1. JSON serialization
        json_str = json.dumps(report, indent=2)
        self.assertIsInstance(json_str, str)
        reloaded = json.loads(json_str)
        self.assertEqual(reloaded["report_id"], report["report_id"])

        # 2. Markdown formatting
        md_text = ForensicReportGenerator.format_markdown(report)
        self.assertIn("# DeepShield Forensic Verification Audit Report", md_text)
        self.assertIn("1. Executive Summary", md_text)
        self.assertIn("2. Multimodal Evidence Synthesis", md_text)
        self.assertIn("10. Recommended Next Actions", md_text)
        self.assertIn("This is an AI-based assessment, not absolute proof of authenticity.", md_text)

    def test_13_v5_2_temporal_evidence_in_report(self):
        """Verifies report generation with DeepShield V5.2 temporal video evidence and concordance status."""
        video_pred = {
            "prediction": "Real",
            "verdict": "Real",
            "confidence": 0.92,
            "media_type": "video",
            "frames_analyzed": 20,
            "phase1_video_evidence": {
                "prediction": "Real",
                "video_calibrated_score": 0.05
            },
            "v5_2_temporal_evidence": {
                "temporal_model": "DeepShield V5.2",
                "temporal_score": 0.038,
                "temporal_threshold": 0.30,
                "sequence_length": 16,
                "valid_face_frames": 16,
                "total_frames": 90,
                "prediction": "Real",
                "confidence": 0.94
            }
        }
        report = ForensicReportGenerator.generate_report(video_pred, report_id="DSR-V5-2-TEST")
        vid_ev = report["evidence"]["video_evidence"]
        self.assertTrue(vid_ev["evidence_available"])
        self.assertIn("v5_2_temporal_evidence", vid_ev)
        self.assertEqual(vid_ev["temporal_model"], "DeepShield V5.2")
        self.assertEqual(vid_ev["temporal_threshold"], 0.30)
        self.assertEqual(vid_ev["v5_2_temporal_evidence"]["temporal_score"], 0.038)
        self.assertIn("Concordant with Phase 1", vid_ev["conflict_status"])

        # Markdown verification
        md_text = ForensicReportGenerator.format_markdown(report)
        self.assertIn("### DeepShield V5.2 Dedicated Temporal Video Evidence", md_text)
        self.assertIn("DeepShield V5.2", md_text)
        self.assertIn("Calibrated Temporal Threshold: `0.30`", md_text)


if __name__ == "__main__":
    unittest.main()
