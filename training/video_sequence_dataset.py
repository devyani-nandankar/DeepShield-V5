"""
DeepShield V5.1 - Temporal Video Sequence Dataset Builder
=========================================================

Handles temporal keyframe sampling, YuNet face detection & 1:1 cropping,
compression data augmentation, and feature sequence preparation.

Key Specifications:
- Sequence length: T = 16 frames
- Face localizer: OpenCV YuNet (Square crop with 20% margin)
- Non-face frames: STRICTLY skipped (no center-crop fallback)
- Spatial resolution: 224 x 224 x 3 (RGB Float32 [0, 255])
- Spatial feature dimension: 1408 (from EfficientNetB2 GlobalAveragePooling2D)
- Splits: By VIDEO GROUP (Zero data leakage across splits)
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from backend.config import YUNET_MODEL_PATH, IMAGE_SIZE
from backend.face_detector import YuNetFaceDetector

class CompressionAugmenter:
    """
    Simulates real-world video compression artifacts:
    - Variable JPEG compression (Q=30..75)
    - Downsampling & upsampling (scale 0.5..0.85)
    - Gaussian / motion blur (kernel 3..5)
    - Additive sensor noise
    """
    def __init__(self, p: float = 0.60):
        self.p = p

    def augment_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return frame_bgr

        aug = frame_bgr.copy()
        choice = random.choice(["jpeg", "downsample", "blur", "noise", "combined"])

        if choice in ["jpeg", "combined"]:
            # JPEG re-compression
            quality = random.randint(30, 75)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc = cv2.imencode(".jpg", aug, encode_param)
            aug = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        if choice in ["downsample", "combined"]:
            # Spatial downsampling + upsampling
            h, w = aug.shape[:2]
            scale = random.uniform(0.50, 0.85)
            small = cv2.resize(aug, (max(16, int(w * scale)), max(16, int(h * scale))), interpolation=cv2.INTER_LINEAR)
            aug = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        if choice in ["blur", "combined"]:
            # Gaussian blur
            ksize = random.choice([3, 5])
            aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

        if choice == "noise":
            # Mild Gaussian sensor noise
            noise = np.random.normal(0, random.uniform(3, 8), aug.shape).astype(np.float32)
            aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return aug


class VideoSequenceDataset:
    """
    Extracts temporal face sequences from video files for DeepShield V5.1.
    """
    def __init__(
        self,
        sequence_length: int = 16,
        use_compression_augmentation: bool = False,
        min_valid_face_frames: int = 3
    ):
        self.T = sequence_length
        self.use_aug = use_compression_augmentation
        self.min_faces = min_valid_face_frames
        self.face_detector = YuNetFaceDetector()
        self.augmenter = CompressionAugmenter(p=0.60) if use_compression_augmentation else None

    def extract_face_sequence_from_video(
        self,
        video_path: str,
        start_frame: int = 0,
        stride: int = 5
    ) -> Tuple[Optional[np.ndarray], int, int, List[float]]:
        """
        Extracts a sequence of T valid face crops from a video stream.

        Returns:
            sequence_tensor: (T, 224, 224, 3) float32 array, or None if insufficient faces.
            valid_face_count: Count of valid face frames extracted.
            total_sampled: Total frames inspected.
            face_confidences: List of face confidence scores.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, 0, 0, []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None, 0, 0, []

        # Determine sampling step
        # If video duration allows, sample across duration
        available_frames = max(1, total_frames - start_frame)
        step = max(1, available_frames // (self.T * 2)) if available_frames >= (self.T * 2) else stride

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_idx = start_frame

        valid_crops = []
        confidences = []
        sampled_count = 0

        while cap.isOpened() and len(valid_crops) < self.T and current_idx < total_frames:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            sampled_count += 1
            if (current_idx - start_frame) % step == 0:
                # Apply compression augmentation if enabled
                frame_to_process = self.augmenter.augment_frame(frame_bgr) if self.use_aug else frame_bgr

                # Detect face with YuNet
                det_res = self.face_detector.detect_faces(frame_to_process)
                if det_res.face_detected and det_res.crop is not None and det_res.crop.size > 0:
                    crop_bgr = det_res.crop
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                    crop_tensor = crop_resized.astype(np.float32)

                    valid_crops.append(crop_tensor)
                    confidences.append(float(det_res.faces[0]["score"]) if det_res.faces else 1.0)

            current_idx += 1

        cap.release()

        valid_face_count = len(valid_crops)
        if valid_face_count < self.min_faces:
            # Insufficient faces to construct a valid temporal sequence
            return None, valid_face_count, sampled_count, confidences

        # Pad sequence up to T if needed (using zero padding for Masking layer)
        if valid_face_count < self.T:
            padded = np.zeros((self.T, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
            # Repeat or place valid crops at beginning
            for i, crop in enumerate(valid_crops):
                padded[i] = crop
            sequence_tensor = padded
        else:
            sequence_tensor = np.stack(valid_crops[:self.T], axis=0)

        return sequence_tensor, valid_face_count, sampled_count, confidences

    def extract_features_for_sequence(
        self,
        sequence_tensor: np.ndarray,
        feature_extractor
    ) -> np.ndarray:
        """
        Extracts compact 1408-dimensional feature vectors for each frame in sequence.
        Input: (T, 224, 224, 3)
        Output: (T, 1408)
        """
        # Batched inference over the T frames
        feats = feature_extractor(sequence_tensor, training=False)
        return feats.numpy()
