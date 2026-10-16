"""
DeepShield V5 - YuNet Face Detection & Cropping Module
======================================================
Implements high-accuracy, lightweight face detection and alignment using
OpenCV's YuNet ONNX model (face_detection_yunet_2023mar.onnx).
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from backend.config import (
    YUNET_MODEL_PATH,
    YUNET_SCORE_THRESHOLD,
    YUNET_NMS_THRESHOLD,
    YUNET_TOP_K,
    FACE_CROP_MARGIN
)


class FaceDetectionResult(tuple):
    """
    Flexible face detection result container supporting tuple unpacking,
    dictionary indexing, and attribute access.
    """
    def __new__(cls, face_detected: bool, face_count: int, faces_info: list, primary_crop_bgr: Optional[np.ndarray], meta: Dict[str, Any]):
        return super().__new__(cls, (face_detected, face_count, faces_info, primary_crop_bgr, meta))

    def __init__(self, face_detected: bool, face_count: int, faces_info: list, primary_crop_bgr: Optional[np.ndarray], meta: Dict[str, Any]):
        self.face_detected = face_detected
        self.face_count = face_count
        self.faces = faces_info
        self.crop = primary_crop_bgr
        self.meta = meta

    def __getitem__(self, item):
        if isinstance(item, str):
            mapping = {
                "face_detected": self.face_detected,
                "face_count": self.face_count,
                "faces": self.faces,
                "faces_info": self.faces,
                "crop": self.crop,
                "meta": self.meta
            }
            if item in mapping:
                return mapping[item]
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class YuNetFaceDetector:
    """
    Encapsulates OpenCV YuNet face detector with dynamic dimension handling,
    face prioritization, and bounded crop extraction.
    """

    def __init__(self, model_path: Path | str = YUNET_MODEL_PATH):
        self.model_path = Path(model_path)
        self.detector = None
        self._initialize_detector()

    def _initialize_detector(self):
        """Initializes the cv2.FaceDetectorYN instance if model weights exist."""
        if not self.model_path.is_file():
            print(f"[!] Warning: YuNet model weights not found at: {self.model_path}")
            print("    Face detection will fallback to centered region if YuNet is unavailable.")
            self.detector = None
            return

        try:
            # Initialize with placeholder input size (320, 320); dynamic resize per image in detect()
            self.detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=(320, 320),
                score_threshold=YUNET_SCORE_THRESHOLD,
                nms_threshold=YUNET_NMS_THRESHOLD,
                top_k=YUNET_TOP_K,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
            print(f"[+] YuNet Face Detector successfully loaded from: {self.model_path}")
        except Exception as exc:
            print(f"[!] Error loading YuNet Face Detector: {exc}")
            self.detector = None

    def is_available(self) -> bool:
        """Returns True if YuNet detector is initialized and ready."""
        return self.detector is not None

    def detect_faces(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        """
        Detects faces in a BGR image array.

        Returns:
            FaceDetectionResult container (face_detected, face_count, faces_info, primary_crop_bgr, meta).
        """
        if image_bgr is None or image_bgr.size == 0:
            return FaceDetectionResult(False, 0, [], None, {"error": "Empty or invalid image array provided."})

        height, width = image_bgr.shape[:2]

        if not self.is_available():
            # Fallback: Center crop if detector weights are not yet downloaded
            center_crop = self._center_crop(image_bgr)
            return FaceDetectionResult(True, 1, [{"bbox": [0, 0, width, height], "score": 1.0, "is_fallback": True}], center_crop, {
                "detector": "FallbackCenterCrop",
                "notice": "YuNet ONNX weights not installed. Used fallback central crop."
            })

        # Set dynamic input size matching current image resolution
        self.detector.setInputSize((width, height))

        _, raw_faces = self.detector.detect(image_bgr)

        if raw_faces is None or len(raw_faces) == 0:
            # Fallback to center crop if no face detected, marking face_detected=False
            center_crop = self._center_crop(image_bgr)
            return FaceDetectionResult(False, 0, [], center_crop, {
                "detector": "YuNet",
                "notice": "Zero faces detected above score threshold. Central region extracted."
            })

        face_count = len(raw_faces)
        faces_info = []

        # Parse detected faces: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
        for f in raw_faces:
            bbox = [int(f[0]), int(f[1]), int(f[2]), int(f[3])]
            score = float(f[14])
            landmarks = {
                "right_eye": [float(f[4]), float(f[5])],
                "left_eye": [float(f[6]), float(f[7])],
                "nose_tip": [float(f[8]), float(f[9])],
                "right_mouth_corner": [float(f[10]), float(f[11])],
                "left_mouth_corner": [float(f[12]), float(f[13])]
            }
            faces_info.append({
                "bbox": bbox,
                "score": score,
                "landmarks": landmarks,
                "area": bbox[2] * bbox[3]
            })

        # Select primary face: prioritize largest bounding box area
        primary_face = max(faces_info, key=lambda item: item["area"])
        x, y, w, h = primary_face["bbox"]

        # Safeguard: If uploaded image is already a tight square face crop (>70% face coverage)
        img_aspect = width / float(height) if height > 0 else 1.0
        face_coverage = (w * h) / float(width * height) if (width > 0 and height > 0) else 0.0

        if (0.85 <= img_aspect <= 1.15) and (face_coverage >= 0.70):
            # Avoid aggressive double-cropping on pre-cropped square faces
            crop = self._center_crop(image_bgr)
            ch, cw = crop.shape[:2]
            cx1 = (width - cw) // 2
            cy1 = (height - ch) // 2
            primary_crop_bbox = [cx1, cy1, cw, ch]
        else:
            # 1:1 SQUARE crop centered on the detected face
            cx = x + w / 2.0
            cy = y + h / 2.0
            base_size = max(w, h)
            square_size = int(round(base_size * (1.0 + 2.0 * FACE_CROP_MARGIN)))
            square_size = min(square_size, min(width, height))

            x1 = int(round(cx - square_size / 2.0))
            y1 = int(round(cy - square_size / 2.0))
            x2 = x1 + square_size
            y2 = y1 + square_size

            # Shift crop into image boundaries while preserving dimensions
            if x1 < 0:
                x2 = min(width, x2 - x1)
                x1 = 0
            if x2 > width:
                x1 = max(0, x1 - (x2 - width))
                x2 = width
            if y1 < 0:
                y2 = min(height, y2 - y1)
                y1 = 0
            if y2 > height:
                y1 = max(0, y1 - (y2 - height))
                y2 = height

            # Enforce exact 1:1 square crop
            side = min(max(0, x2 - x1), max(0, y2 - y1))
            if side <= 0:
                crop = self._center_crop(image_bgr)
                ch, cw = crop.shape[:2]
                primary_crop_bbox = [(width - cw) // 2, (height - ch) // 2, cw, ch]
            else:
                diff_x = (x2 - x1) - side
                diff_y = (y2 - y1) - side
                x1 += diff_x // 2
                x2 = x1 + side
                y1 += diff_y // 2
                y2 = y1 + side
                crop = image_bgr[y1:y2, x1:x2]
                primary_crop_bbox = [x1, y1, x2 - x1, y2 - y1]

        if crop is None or crop.size == 0:
            crop = self._center_crop(image_bgr)
            ch, cw = crop.shape[:2]
            primary_crop_bbox = [(width - cw) // 2, (height - ch) // 2, cw, ch]

        meta = {
            "detector": "YuNet",
            "image_width": width,
            "image_height": height,
            "primary_bbox": primary_crop_bbox,
            "primary_score": primary_face["score"]
        }

        return FaceDetectionResult(True, face_count, faces_info, crop, meta)

    def _center_crop(self, image_bgr: np.ndarray) -> np.ndarray:
        """Fallback square crop around image center if no face is detected."""
        h, w = image_bgr.shape[:2]
        size = min(h, w)
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        return image_bgr[start_y:start_y + size, start_x:start_x + size]


# Global singleton instance for efficient runtime reuse
_detector_instance: Optional[YuNetFaceDetector] = None

# Class alias for convenience
FaceDetector = YuNetFaceDetector


def get_face_detector() -> YuNetFaceDetector:
    """Returns the global singleton YuNetFaceDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YuNetFaceDetector()
    return _detector_instance
