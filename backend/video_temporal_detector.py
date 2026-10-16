"""
DeepShield V5.1 - Dedicated Temporal Video Detector
==================================================

Implements the V5.1 Spatial-Temporal Video Deepfake Detection Pipeline:
- Frame sampling (T = 16 keyframes)
- OpenCV YuNet face detection & 1:1 square crop
- EfficientNetB2 spatial feature extraction (1408 dimensions)
- Bidirectional LSTM temporal classification
- Frame-level temporal contribution and suspicious frame identification
- Independent calibrated threshold (VIDEO_TEMPORAL_THRESHOLD)

Preserves V5 Image Detector and Phase 1 Video Detector as fallbacks.
"""

import os
import cv2
import time
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import tensorflow as tf

from backend.config import (
    IMAGE_SIZE,
    CALIBRATED_THRESHOLD as IMAGE_THRESHOLD,
    GENERATED_FOLDER,
    DEFAULT_V5_2_TEMPORAL_MODEL_PATH,
    V5_2_TEMPORAL_MODEL_PATH,
    VIDEO_TEMPORAL_THRESHOLD as CONFIG_VIDEO_TEMPORAL_THRESHOLD,
    VIDEO_TEMPORAL_MARGIN as CONFIG_VIDEO_TEMPORAL_MARGIN
)
from backend.face_detector import YuNetFaceDetector

# Model Specifications
V5_MODEL_PATH = Path("models/deepshield_v5_clean_best.keras")
V5_2_MODEL_A_PATH = DEFAULT_V5_2_TEMPORAL_MODEL_PATH
V5_1_MODEL_PATH = Path("models/v5_1/deepshield_v5_1_temporal.keras")
V5_1_NO_AUG_PATH = Path("models/v5_1/deepshield_v5_1_temporal_no_aug.keras")

# Independently calibrated temporal threshold (0.30 for V5.2 Model A, NOT 0.21)
VIDEO_TEMPORAL_THRESHOLD = CONFIG_VIDEO_TEMPORAL_THRESHOLD
VIDEO_TEMPORAL_MARGIN = CONFIG_VIDEO_TEMPORAL_MARGIN
DEFAULT_T = 16
MIN_FACE_FRAMES = 3


class VideoTemporalDetector:
    """
    Dedicated DeepShield V5.2 Video Deepfake Detector using EfficientNetB2 + BiLSTM.
    """
    def __init__(
        self,
        temporal_model_path: Optional[str] = None,
        base_v5_model_path: Optional[str] = None,
        sequence_length: int = DEFAULT_T,
        threshold: float = VIDEO_TEMPORAL_THRESHOLD
    ):
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.face_detector = YuNetFaceDetector()

        # Load Spatial Feature Extractor from V5
        v5_path = Path(base_v5_model_path) if base_v5_model_path else V5_MODEL_PATH
        if not v5_path.is_absolute():
            v5_path = Path(__file__).resolve().parent.parent / v5_path

        if not v5_path.exists():
            raise FileNotFoundError(f"V5 base model not found at: {v5_path}")

        v5_base = tf.keras.models.load_model(str(v5_path))
        # Layer 2 is GlobalAveragePooling2D (1408-dim)
        self.feature_extractor = tf.keras.Model(
            inputs=v5_base.input,
            outputs=v5_base.layers[2].output,
            name="V5_Feature_Extractor"
        )
        self.feature_extractor.trainable = False

        # Load Temporal Model (V5.2 Model A preferred, fallback to V5.1)
        t_path = Path(temporal_model_path) if temporal_model_path else V5_2_MODEL_A_PATH
        if not t_path.is_absolute():
            t_path = Path(__file__).resolve().parent.parent / t_path

        if not t_path.exists():
            # Fallback to V5.1 models if V5.2 not found
            fallback_v5_1_no_aug = Path(__file__).resolve().parent.parent / V5_1_NO_AUG_PATH
            fallback_v5_1_aug = Path(__file__).resolve().parent.parent / V5_1_MODEL_PATH
            if fallback_v5_1_no_aug.exists():
                t_path = fallback_v5_1_no_aug
            elif fallback_v5_1_aug.exists():
                t_path = fallback_v5_1_aug
            else:
                raise FileNotFoundError(f"Temporal video model not found at: {t_path}")

        self.temporal_model = tf.keras.models.load_model(str(t_path))
        self.temporal_model_path = str(t_path)

    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Executes end-to-end V5.1 temporal video deepfake detection.

        Args:
            video_path: Absolute or relative path to video file.

        Returns:
            Structured API-compatible dictionary conforming to V5.1 specifications.
        """
        start_time = time.time()
        video_path = str(Path(video_path).resolve())

        if not os.path.exists(video_path):
            return {
                "success": False,
                "error": f"Video file not found: {video_path}"
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"Failed to open video container: {video_path}"
            }

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = float(total_frames / fps) if fps > 0 else 0.0
        file_size = os.path.getsize(video_path)
        bitrate_kbps = round((file_size * 8) / (duration * 1000), 1) if duration > 0 else 0.0

        # Sample T keyframes evenly spaced across the video
        target_T = self.sequence_length
        step = max(1, total_frames // (target_T * 2)) if total_frames >= (target_T * 2) else max(1, total_frames // target_T)

        valid_face_crops = []
        valid_face_tensors = []
        frame_timestamps = []
        face_confidences = []
        face_bboxes = []
        sampled_indices = []
        blur_variances = []
        current_frame = 0

        while cap.isOpened() and len(valid_face_crops) < target_T and current_frame < total_frames:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            if current_frame % step == 0:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                b_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                blur_variances.append(b_var)

                # Face detection with YuNet
                det = self.face_detector.detect_faces(frame_bgr)
                if det.face_detected and det.crop is not None and det.crop.size > 0:
                    crop_bgr = det.crop
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                    crop_tensor = crop_resized.astype(np.float32)

                    valid_face_crops.append(crop_rgb)
                    valid_face_tensors.append(crop_tensor)
                    frame_timestamps.append(round(current_frame / fps, 2) if fps > 0 else 0.0)
                    face_confidences.append(float(det.faces[0]["score"]) if det.faces else 1.0)
                    face_bboxes.append(det.faces[0]["bbox"] if det.faces else None)
                    sampled_indices.append(current_frame)

            current_frame += 1

        cap.release()

        valid_face_count = len(valid_face_crops)
        total_sampled = len(blur_variances)
        avg_blur = float(np.mean(blur_variances)) if blur_variances else 0.0
        compression_degradation = bool(min(width, height) < 720 or bitrate_kbps < 1200 or avg_blur < 100)

        # Insufficient faces check
        if valid_face_count < MIN_FACE_FRAMES:
            reason = f"Insufficient localized face keyframes ({valid_face_count} detected; minimum {MIN_FACE_FRAMES} required for temporal modeling)."
            return {
                "temporal_model": "DeepShield V5.2",
                "model": "DeepShield V5.2",
                "model_file": self.temporal_model_path,
                "model_type": "EfficientNetB2 + BiLSTM",
                "sequence_length": self.sequence_length,
                "temporal_score": None,
                "temporal_threshold": self.threshold,
                "video_temporal_threshold": self.threshold,
                "number_of_sequences": 0,
                "valid_face_frames": valid_face_count,
                "total_frames": total_frames,
                "frames_sampled": total_sampled,
                "video_prediction": "Uncertain",
                "prediction": "Uncertain",
                "verdict": "Uncertain",
                "video_score": 0.50,
                "confidence": 0.50,
                "temporal_features": True,
                "compression_augmentation": False,
                "uncertainty": reason,
                "uncertainty_reason": reason,
                "suspicious_keyframes": [],
                "diagnostics": {
                    "raw_probability": None,
                    "calibrated_video_threshold": self.threshold,
                    "image_threshold": IMAGE_THRESHOLD,
                    "compression_degradation": compression_degradation,
                    "avg_blur_variance": round(avg_blur, 1),
                    "resolution": f"{width}x{height}",
                    "bitrate_kbps": bitrate_kbps,
                    "fps": fps,
                    "duration": round(duration, 2),
                    "temporal_model_path": self.temporal_model_path,
                    "processing_time_seconds": round(time.time() - start_time, 3)
                },
                "technical_details": {
                    "architecture": "EfficientNetB2 (Spatial 1408-dim) + BiLSTM (128-64-64-1)",
                    "sequence_length": self.sequence_length,
                    "video_temporal_threshold": self.threshold,
                    "image_threshold": IMAGE_THRESHOLD,
                    "processing_time_seconds": round(time.time() - start_time, 3)
                }
            }

        # Sequence Padding up to T
        if valid_face_count < target_T:
            padded_tensors = np.zeros((target_T, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
            for i, tensor in enumerate(valid_face_tensors):
                padded_tensors[i] = tensor
            seq_tensors = padded_tensors
        else:
            seq_tensors = np.stack(valid_face_tensors[:target_T], axis=0)

        # Spatial Feature Extraction: (T, 224, 224, 3) -> (T, 1408)
        feature_sequence = self.feature_extractor(seq_tensors, training=False).numpy()

        # Input to BiLSTM: (1, T, 1408)
        bilstm_input = np.expand_dims(feature_sequence, axis=0)
        raw_prob = float(self.temporal_model(bilstm_input, training=False)[0, 0])
        raw_prob = float(np.clip(raw_prob, 0.0, 1.0))

        # Temporal Decision Rule based on calibrated VIDEO_TEMPORAL_THRESHOLD (0.30)
        lower_bound = round(max(0.0, self.threshold - VIDEO_TEMPORAL_MARGIN), 4)
        upper_bound = round(min(1.0, self.threshold + VIDEO_TEMPORAL_MARGIN), 4)

        uncertainty_reason = None
        if lower_bound <= raw_prob <= upper_bound:
            verdict = "Uncertain"
            prediction = "Uncertain"
            confidence = 0.50
            uncertainty_reason = f"Temporal prediction score ({raw_prob:.4f}) is in the boundary window [{lower_bound:.2f}, {upper_bound:.2f}]."
        elif raw_prob >= self.threshold:
            verdict = "Likely Fake"
            prediction = "Fake"
            confidence = float(np.clip(0.50 + 0.50 * ((raw_prob - self.threshold) / (1.0 - self.threshold + 1e-7)), 0.50, 0.9999))
        else:
            verdict = "Real"
            prediction = "Real"
            confidence = float(np.clip(0.50 + 0.50 * ((self.threshold - raw_prob) / (self.threshold + 1e-7)), 0.50, 0.9999))

        # Suspicious Keyframe Extraction (frames with highest feature variance)
        suspicious_frames = []
        if valid_face_crops:
            # Estimate per-frame contribution by feature deviation
            feat_mean = np.mean(feature_sequence, axis=0)
            deviations = [float(np.linalg.norm(feature_sequence[i] - feat_mean)) for i in range(min(valid_face_count, target_T))]
            sorted_idx = np.argsort(deviations)[::-1][:min(3, valid_face_count)]

            for rank, s_i in enumerate(sorted_idx, 1):
                token = uuid.uuid4().hex[:8]
                thumb_name = f"v5_2_suspicious_{rank}_{token}.jpg"
                thumb_path = GENERATED_FOLDER / thumb_name
                cv2.imwrite(str(thumb_path), cv2.cvtColor(valid_face_crops[s_i], cv2.COLOR_RGB2BGR))

                suspicious_frames.append({
                    "rank": rank,
                    "frame_number": sampled_indices[s_i] if s_i < len(sampled_indices) else 0,
                    "timestamp": frame_timestamps[s_i] if s_i < len(frame_timestamps) else 0.0,
                    "temporal_anomaly_score": round(deviations[s_i], 4),
                    "thumbnail_url": f"/generated/{thumb_name}"
                })

        processing_time = round(time.time() - start_time, 3)

        return {
            "temporal_model": "DeepShield V5.2",
            "model": "DeepShield V5.2",
            "model_file": self.temporal_model_path,
            "model_type": "EfficientNetB2 + BiLSTM",
            "sequence_length": self.sequence_length,
            "temporal_score": round(raw_prob, 4),
            "temporal_threshold": self.threshold,
            "video_temporal_threshold": self.threshold,
            "number_of_sequences": 1,
            "valid_face_frames": valid_face_count,
            "total_frames": total_frames,
            "frames_sampled": total_sampled,
            "video_prediction": verdict,
            "prediction": prediction,
            "verdict": verdict,
            "video_score": round(raw_prob, 4),
            "confidence": round(confidence, 4),
            "temporal_features": True,
            "compression_augmentation": False,
            "suspicious_keyframes": suspicious_frames,
            "uncertainty": uncertainty_reason,
            "uncertainty_reason": uncertainty_reason,
            "diagnostics": {
                "raw_probability": round(raw_prob, 4),
                "calibrated_video_threshold": self.threshold,
                "image_threshold": IMAGE_THRESHOLD,
                "compression_degradation": compression_degradation,
                "avg_blur_variance": round(avg_blur, 1),
                "resolution": f"{width}x{height}",
                "bitrate_kbps": bitrate_kbps,
                "fps": fps,
                "duration": round(duration, 2),
                "temporal_model_path": self.temporal_model_path,
                "processing_time_seconds": processing_time
            },
            "technical_details": {
                "architecture": "EfficientNetB2 (Spatial 1408-dim) + BiLSTM (128-64-64-1)",
                "sequence_length": self.sequence_length,
                "video_temporal_threshold": self.threshold,
                "image_threshold": IMAGE_THRESHOLD,
                "processing_time_seconds": processing_time
            }
        }


# Global singleton instance
_temporal_detector_instance: Optional[VideoTemporalDetector] = None


def get_temporal_detector() -> VideoTemporalDetector:
    """Returns singleton instance of VideoTemporalDetector."""
    global _temporal_detector_instance
    if _temporal_detector_instance is None:
        _temporal_detector_instance = VideoTemporalDetector()
    return _temporal_detector_instance
