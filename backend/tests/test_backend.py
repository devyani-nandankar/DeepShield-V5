"""
DeepShield V5 - Backend Verification & Unit Tests
=================================================
Automated verification tests for the DeepShield Flask REST API,
metadata forensics, fusion uncertainty quantification, and report generation.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
import io
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app
from backend.config import CALIBRATED_THRESHOLD
from backend.metadata_forensics import MetadataForensics
from backend.fusion import MultimodalFusionEngine
from backend.report_generator import ForensicReportGenerator


class DeepShieldBackendTestCase(unittest.TestCase):
    """Test suite verifying backend endpoints and forensic modules."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_health_endpoint(self):
        """Verify /api/health returns 200, status online, and accurate V5 metadata."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("status"), "online")
        self.assertEqual(data.get("version"), "5.0.0")

        # Verify benchmark metrics honesty
        model_info = data.get("model_info", {})
        metrics = model_info.get("benchmark_test_metrics", {})
        self.assertEqual(metrics.get("accuracy"), "77.55%")
        self.assertEqual(metrics.get("roc_auc"), "83.30%")
        self.assertEqual(metrics.get("optimal_f1"), 0.8771)
        self.assertEqual(model_info.get("calibrated_threshold"), 0.21)

    def test_02_video_predict_placeholder(self):
        """Verify /api/predict/video returns decoupled status without error."""
        # Create a tiny mock video buffer
        mock_data = {
            "video": (io.BytesIO(b"RIFF\x00\x00\x00\x00AVI LIST"), "sample_video.mp4")
        }
        response = self.client.post(
            "/api/predict/video",
            data=mock_data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("media_type"), "video")
        self.assertEqual(data.get("status"), "pipeline_staged")
        self.assertIn("recommendation", data)

    def test_03_metadata_forensics(self):
        """Verify metadata extractor on an in-memory test image."""
        # Create a temporary image
        img = Image.new("RGB", (300, 300), color=(128, 64, 32))
        tmp_img_path = os.path.join(PROJECT_ROOT, "backend", "uploads", "test_meta.jpg")
        img.save(tmp_img_path, "JPEG")

        try:
            meta = MetadataForensics.extract_metadata(tmp_img_path)
            self.assertTrue(meta.get("success"))
            self.assertEqual(meta.get("format"), "JPEG")
            self.assertEqual(meta.get("dimensions"), {"width": 300, "height": 300})
            self.assertIn("forensic_notes", meta)
        finally:
            if os.path.isfile(tmp_img_path):
                os.remove(tmp_img_path)

    def test_04_fusion_uncertainty_quantification(self):
        """Verify boundary uncertainty quantification around 0.21."""
        # Case A: Score exactly at boundary (0.21)
        pred_boundary = {
            "prediction": "Fake",
            "score": 0.21,
            "confidence": 0.5,
            "face_detected": True,
            "metadata": {}
        }
        fusion_boundary = MultimodalFusionEngine.evaluate_fusion(pred_boundary)
        self.assertTrue(fusion_boundary["boundary_uncertainty"]["is_boundary_uncertain"])
        self.assertEqual(fusion_boundary["risk_level"], "Elevated")

        # Case B: Clear authentic score (0.02)
        pred_real = {
            "prediction": "Real",
            "score": 0.02,
            "confidence": 0.94,
            "face_detected": True,
            "metadata": {}
        }
        fusion_real = MultimodalFusionEngine.evaluate_fusion(pred_real)
        self.assertFalse(fusion_real["boundary_uncertainty"]["is_boundary_uncertain"])
        self.assertEqual(fusion_real["risk_level"], "Low")

        # Case C: High fake score (0.92)
        pred_fake = {
            "prediction": "Fake",
            "score": 0.92,
            "confidence": 0.95,
            "face_detected": True,
            "metadata": {"editing_software_detected": True, "editing_software_name": "Adobe Photoshop"}
        }
        fusion_fake = MultimodalFusionEngine.evaluate_fusion(pred_fake)
        self.assertFalse(fusion_fake["boundary_uncertainty"]["is_boundary_uncertain"])
        self.assertEqual(fusion_fake["risk_level"], "Critical")
        self.assertTrue(len(fusion_fake["corroborating_factors"]) > 0)

    def test_05_forensic_report_generation(self):
        """Verify structured 10-section forensic audit report compilation."""
        simulated_prediction = {
            "prediction": "Fake",
            "score": 0.8842,
            "confidence": 0.9311,
            "threshold": 0.21,
            "face_detected": True,
            "face_count": 1,
            "faces": [{"box": [10, 10, 100, 100], "score": 0.98}],
            "explanation": "Simulated manipulation artifacts detected.",
            "evidence": [
                {"source": "EfficientNetB2", "supports": "Fake", "finding": "Score 0.8842"}
            ],
            "gradcam": {
                "available": True,
                "heatmap_url": "/api/generated/test_h.jpg",
                "overlay_url": "/api/generated/test_o.jpg",
                "peak_intensity": 0.95,
                "mean_intensity": 0.42,
                "salient_ratio": 0.28
            },
            "metadata": {
                "format": "JPEG",
                "file_size_human": "1.2 MB",
                "has_exif": True,
                "editing_software_detected": False
            }
        }

        report = ForensicReportGenerator.generate_report(simulated_prediction, report_id="TEST-001")
        self.assertEqual(report["report_id"], "TEST-001")
        sections = report["sections"]

        # Ensure all 10 sections are present
        expected_sections = [
            "1_executive_summary",
            "2_architecture_specifications",
            "3_benchmark_performance",
            "4_face_localization_audit",
            "5_explainability_attention",
            "6_metadata_forensics",
            "7_decision_boundary_uncertainty",
            "8_evidence_matrix",
            "9_forensic_disclaimer",
            "10_recommended_actions"
        ]
        for sec in expected_sections:
            self.assertIn(sec, sections, f"Section {sec} missing from report.")

        # Test Markdown formatter
        md = ForensicReportGenerator.format_markdown(report)
        self.assertIn("# DeepShield Forensic Verification Audit Report", md)
        self.assertIn("1. Executive Summary", md)
        self.assertIn("10. Recommended Next Actions", md)

    def test_06_report_api_endpoint(self):
        """Verify /api/report with JSON payload returns 200 and markdown option."""
        payload = {
            "prediction_result": {
                "prediction": "Real",
                "score": 0.045,
                "confidence": 0.86,
                "face_detected": True,
                "face_count": 1,
                "faces": [],
                "explanation": "Authentic sample.",
                "evidence": [],
                "gradcam": {"available": False},
                "metadata": {}
            }
        }

        # JSON response
        res_json = self.client.post("/api/report", json=payload)
        self.assertEqual(res_json.status_code, 200)
        self.assertIn("sections", res_json.get_json())

        # Markdown response
        res_md = self.client.post("/api/report?format=markdown", json=payload)
        self.assertEqual(res_md.status_code, 200)
        self.assertIn(b"DeepShield Forensic Verification Audit Report", res_md.data)

    def test_07_predict_image_live_or_fallback(self):
        """Verify /api/predict/image executes end-to-end when model exists, or returns 503 if missing."""
        # Create a valid test image
        img = Image.new("RGB", (320, 320), color=(200, 150, 100))
        img_buf = io.BytesIO()
        img.save(img_buf, format="JPEG")
        img_buf.seek(0)

        mock_data = {
            "image": (img_buf, "test_valid.jpg")
        }
        response = self.client.post(
            "/api/predict/image",
            data=mock_data,
            content_type="multipart/form-data"
        )
        data = response.get_json()

        from backend.image_detector import get_image_detector
        if get_image_detector().is_model_loaded():
            self.assertEqual(response.status_code, 200)
            self.assertTrue(data.get("success"))
            self.assertIn(data.get("prediction"), ["Real", "Fake"])
            self.assertIn("score", data)
            self.assertIn("confidence", data)
            self.assertEqual(data.get("threshold"), 0.21)
        else:
            self.assertEqual(response.status_code, 503)
            self.assertFalse(data.get("success"))

    def test_08_predict_image_invalid_extension_400(self):
        """Verify /api/predict/image returns 400 for unsupported file types."""
        mock_data = {
            "image": (io.BytesIO(b"malicious script content"), "payload.exe")
        }
        response = self.client.post(
            "/api/predict/image",
            data=mock_data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data.get("success"))
        self.assertIn("unsupported file type", data.get("error", {}).get("message", "").lower())


if __name__ == "__main__":
    unittest.main()
