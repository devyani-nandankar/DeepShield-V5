"""
Unit Tests for DeepShield V5.2 Dedicated Temporal Video Detector
===============================================================

Verifies:
- V5.2 Model A loading
- Calibrated temporal threshold (0.30)
- Sequence construction (T=16)
- Face-only frame filtering (strictly no center-crop fallback)
- Sequence padding and masking
- Model input and output shapes
- V5.2 structured evidence output format
- Insufficient frames handling
- Corrupted / invalid video handling
"""

import os
import unittest
import numpy as np
import cv2
from pathlib import Path

from backend.video_temporal_detector import (
    VideoTemporalDetector,
    VIDEO_TEMPORAL_THRESHOLD,
    DEFAULT_T,
    get_temporal_detector
)
from training.video_sequence_dataset import (
    VideoSequenceDataset,
    CompressionAugmenter
)


class TestVideoTemporalDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = VideoTemporalDetector()
        cls.dataset = VideoSequenceDataset(sequence_length=16, use_compression_augmentation=False)
        cls.augmenter = CompressionAugmenter(p=1.0)
        cls.test_real_video = "backend/test_images/real_check.mp4"
        cls.test_fake_video = "backend/test_images/fake_check.mp4"

    def test_01_feature_extractor_shape(self):
        """Verify spatial feature extractor output dimension is (1, 1408)."""
        dummy_crop = np.zeros((1, 224, 224, 3), dtype=np.float32)
        feats = self.detector.feature_extractor(dummy_crop, training=False)
        self.assertEqual(feats.shape, (1, 1408))

    def test_02_temporal_model_input_output_shape(self):
        """Verify temporal BiLSTM model accepts (1, 16, 1408) and outputs (1, 1)."""
        dummy_seq = np.zeros((1, 16, 1408), dtype=np.float32)
        out = self.detector.temporal_model(dummy_seq, training=False)
        self.assertEqual(out.shape, (1, 1))
        val = float(out[0, 0])
        self.assertTrue(0.0 <= val <= 1.0)

    def test_03_calibrated_temporal_threshold(self):
        """Verify calibrated temporal threshold is set to 0.30 for V5.2 Model A."""
        self.assertEqual(self.detector.threshold, 0.30)
        self.assertEqual(VIDEO_TEMPORAL_THRESHOLD, 0.30)

    def test_04_compression_augmenter(self):
        """Verify compression augmenter produces valid BGR frames."""
        dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        aug_frame = self.augmenter.augment_frame(dummy_frame)
        self.assertEqual(aug_frame.shape, (100, 100, 3))
        self.assertEqual(aug_frame.dtype, np.uint8)

    def test_05_face_only_frame_filtering(self):
        """Verify non-face frames are skipped and not padded with background crops."""
        # A synthetic video with pure black frames (no faces)
        tmp_video = "backend/test_images/tmp_no_face_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_video, fourcc, 10.0, (200, 200))
        for _ in range(20):
            out.write(np.zeros((200, 200, 3), dtype=np.uint8))
        out.release()

        try:
            seq, faces, sampled, _ = self.dataset.extract_face_sequence_from_video(tmp_video)
            self.assertIsNone(seq)
            self.assertEqual(faces, 0)
            self.assertGreater(sampled, 0)
        finally:
            if os.path.exists(tmp_video):
                os.remove(tmp_video)

    def test_06_insufficient_frames_handling(self):
        """Verify detector safely flags Uncertain when face count < 3."""
        tmp_video = "backend/test_images/tmp_insufficient_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_video, fourcc, 10.0, (200, 200))
        for _ in range(10):
            out.write(np.zeros((200, 200, 3), dtype=np.uint8))
        out.release()

        try:
            res = self.detector.predict(tmp_video)
            self.assertEqual(res["temporal_model"], "DeepShield V5.2")
            self.assertEqual(res["video_prediction"], "Uncertain")
            self.assertEqual(res["confidence"], 0.50)
            self.assertIn("Insufficient localized face keyframes", res["uncertainty"])
            self.assertEqual(res["sequence_length"], 16)
            self.assertEqual(res["temporal_threshold"], 0.30)
        finally:
            if os.path.exists(tmp_video):
                os.remove(tmp_video)

    def test_07_corrupted_video_handling(self):
        """Verify detector handles non-existent or corrupted video files gracefully."""
        res = self.detector.predict("backend/test_images/non_existent_video.mp4")
        self.assertFalse(res.get("success", True))
        self.assertIn("not found", res.get("error", "").lower())

    def test_08_sequence_padding_masking(self):
        """Verify sequence padding correctly fills up to T=16."""
        if os.path.exists(self.test_real_video):
            seq, faces, sampled, _ = self.dataset.extract_face_sequence_from_video(self.test_real_video)
            if seq is not None:
                self.assertEqual(seq.shape, (16, 224, 224, 3))
                self.assertGreaterEqual(faces, 3)

    def test_09_v5_2_structured_evidence_output_format(self):
        """Verify predict produces all required V5.2 temporal evidence fields."""
        if os.path.exists(self.test_real_video):
            res = self.detector.predict(self.test_real_video)
            self.assertEqual(res["temporal_model"], "DeepShield V5.2")
            self.assertEqual(res["sequence_length"], 16)
            self.assertEqual(res["temporal_threshold"], 0.30)
            self.assertIn("temporal_score", res)
            self.assertIn("number_of_sequences", res)
            self.assertIn("valid_face_frames", res)
            self.assertIn("total_frames", res)
            self.assertIn("prediction", res)
            self.assertIn("confidence", res)
            self.assertIn("uncertainty", res)
            self.assertIn(res["prediction"], ["Real", "Fake", "Uncertain"])
            self.assertTrue(0.0 <= res["video_score"] <= 1.0)
            self.assertTrue(0.50 <= res["confidence"] <= 1.0)
            self.assertIn("v5_2", res["model_file"].lower())

    def test_10_singleton_accessor(self):
        """Verify get_temporal_detector returns a functional singleton instance."""
        detector_instance = get_temporal_detector()
        self.assertIsInstance(detector_instance, VideoTemporalDetector)
        self.assertEqual(detector_instance.threshold, 0.30)


if __name__ == "__main__":
    unittest.main()
