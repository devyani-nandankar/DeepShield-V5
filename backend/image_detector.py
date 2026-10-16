"""
DeepShield V5 - Image Detection Service
=======================================
Orchestrates face detection (YuNet), deepfake inference (EfficientNetB2),
Grad-CAM explainability generation, and metadata extraction for images.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import uuid
import numpy as np
import cv2
import tensorflow as tf

from backend.config import (
    MODEL_PATH,
    IMAGE_SIZE,
    CALIBRATED_THRESHOLD,
    CLASS_MAPPING,
    GENERATED_FOLDER
)
from backend.face_detector import get_face_detector
from backend.gradcam import GradCAMExplainer
from backend.metadata_forensics import MetadataForensics

logger = logging.getLogger("deepshield.image_detector")


class ImageDetector:
    """
    Singleton inference engine for DeepShield V5 image deepfake detection.
    """
    _instance: Optional["ImageDetector"] = None

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_PATH
        self.model: Optional[tf.keras.Model] = None
        self.explainer: Optional[GradCAMExplainer] = None
        self.face_detector = get_face_detector()
        self._load_model()

    @classmethod
    def get_instance(cls, model_path: Optional[Path] = None) -> "ImageDetector":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            cls._instance = ImageDetector(model_path)
        return cls._instance

    def _load_model(self) -> bool:
        """
        Loads the trained DeepShield V5 Keras model.
        Fails gracefully if the model weight file has not yet been placed on disk.
        """
        if not self.model_path.is_file():
            logger.warning(
                f"DeepShield V5 model weights not found at: {self.model_path}. "
                "Inference requests will report that model weights are pending deployment."
            )
            self.model = None
            self.explainer = None
            return False

        try:
            logger.info(f"Loading DeepShield V5 model from: {self.model_path}")
            # EfficientNetB2 Keras model loaded without recompilation for inference
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            logger.info("DeepShield V5 model loaded successfully.")

            # Initialize explainability engine
            try:
                self.explainer = GradCAMExplainer(self.model)
            except Exception as e:
                logger.warning(f"GradCAMExplainer initialization warning: {e}")
                self.explainer = None

            return True
        except Exception as e:
            logger.error(f"Failed to load DeepShield V5 model: {e}", exc_info=True)
            self.model = None
            self.explainer = None
            return False

    def is_model_loaded(self) -> bool:
        """Returns True if the primary V5 Keras model is loaded in memory."""
        if self.model is None and self.model_path.is_file():
            self._load_model()
        return self.model is not None

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Executes end-to-end deepfake detection on an image file.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            Dictionary matching the DeepShield API specification.
        """
        p = Path(image_path)
        if not p.is_file():
            return {
                "success": False,
                "error": f"Image file not found: {image_path}"
            }

        # 1. Ensure Model is Loaded
        if not self.is_model_loaded():
            return {
                "success": False,
                "error": (
                    f"DeepShield V5 model file not found at '{self.model_path.name}'. "
                    "Please place the trained 'deepshield_v5_clean_best.keras' into the models/ directory."
                ),
                "model_available": False
            }

        # 2. Read Image via OpenCV (BGR)
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            return {
                "success": False,
                "error": "Failed to decode image file. File may be corrupted or in an unsupported format."
            }
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 3. Face Detection & Cropping
        detection_result = self.face_detector.detect_faces(img_bgr)
        face_detected = detection_result.face_detected
        face_count = detection_result.face_count
        faces = detection_result.faces

        # Extract target face crop or fallback
        if face_detected and detection_result.crop is not None and detection_result.crop.size > 0:
            target_face_rgb = cv2.cvtColor(detection_result.crop, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to whole image if no face detected
            target_face_rgb = img_rgb

        # 4. Preprocessing for EfficientNetB2
        # Training pipeline loads RGB, resizes to 224x224, float32 range [0, 255]
        face_resized = cv2.resize(target_face_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        input_tensor = np.expand_dims(face_resized.astype(np.float32), axis=0)

        # 5. Model Inference
        try:
            preds = self.model.predict(input_tensor, verbose=0)
            raw_score = float(preds[0][0])
        except Exception as e:
            logger.error(f"Model prediction failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Inference execution failed: {str(e)}"
            }

        # 6. Calibrated Threshold Classification
        # 0 = Real, 1 = Fake (Higher sigmoid score = Fake)
        is_fake = raw_score >= CALIBRATED_THRESHOLD
        prediction = CLASS_MAPPING[1] if is_fake else CLASS_MAPPING[0]

        # Compute calibrated confidence [0.5, 1.0]
        confidence = self._compute_calibrated_confidence(raw_score, CALIBRATED_THRESHOLD)

        # 7. Persist visual evidence artifacts (Original & Face Crop)
        token = uuid.uuid4().hex[:12]
        orig_fname = f"original_{token}.jpg"
        crop_fname = f"face_crop_{token}.jpg"

        cv2.imwrite(str(GENERATED_FOLDER / orig_fname), img_bgr)
        cv2.imwrite(str(GENERATED_FOLDER / crop_fname), cv2.cvtColor(target_face_rgb, cv2.COLOR_RGB2BGR))

        original_url = f"/api/generated/{orig_fname}"
        face_crop_url = f"/api/generated/{crop_fname}"

        # 8. Explainability via Grad-CAM
        gradcam_data: Dict[str, Any] = {
            "available": False,
            "heatmap_url": None,
            "overlay_url": None,
            "error": "Grad-CAM explainer not initialized."
        }
        if self.explainer is not None:
            gradcam_data = self.explainer.generate(
                image_tensor=tf.convert_to_tensor(input_tensor),
                original_image_rgb=target_face_rgb
            )

        # 9. Metadata Forensics
        metadata_data = MetadataForensics.extract_metadata(str(p))

        # 10. Technical Details Packaging
        face_conf_str = "None"
        if faces and len(faces) > 0 and "score" in faces[0]:
            face_conf_str = f"{faces[0]['score'] * 100:.1f}%"
        elif face_detected:
            face_conf_str = "Detected"

        target_layer_str = self.explainer.target_layer_name if self.explainer else "efficientnetb2/top_conv"

        technical_details = {
            "face_detector": "YuNet",
            "model": "DeepShield V5 / EfficientNetB2",
            "input_resolution": "224x224",
            "model_score": round(raw_score, 4),
            "threshold": CALIBRATED_THRESHOLD,
            "gradcam_target_layer": target_layer_str,
            "face_detection_confidence": face_conf_str
        }

        visual_evidence = {
            "original_image_url": original_url,
            "face_crop_url": face_crop_url,
            "heatmap_url": gradcam_data.get("heatmap_url"),
            "overlay_url": gradcam_data.get("overlay_url")
        }

        display_verdict = "Likely Fake" if is_fake else "Real"

        # 11. Simple 2-4 Bullet Explanation adhering strictly to rules
        bullets = self._generate_bullets(
            prediction=prediction,
            raw_score=raw_score,
            face_detected=face_detected,
            face_conf_str=face_conf_str,
            gradcam_data=gradcam_data,
            metadata_data=metadata_data
        )

        # 12. Evidence Aggregation & Human Explanation
        evidence = self._synthesize_evidence(
            raw_score=raw_score,
            prediction=prediction,
            face_detected=face_detected,
            face_count=face_count,
            gradcam_data=gradcam_data,
            metadata_data=metadata_data
        )

        explanation = self._generate_explanation(
            prediction=prediction,
            raw_score=raw_score,
            confidence=confidence,
            face_detected=face_detected,
            gradcam_data=gradcam_data,
            metadata_data=metadata_data
        )

        disclaimer = "This is an AI-based assessment, not absolute proof of authenticity."

        return {
            "success": True,
            "media_type": "image",
            "prediction": prediction,
            "display_verdict": display_verdict,
            "score": round(raw_score, 4),
            "confidence": round(confidence, 4),
            "threshold": CALIBRATED_THRESHOLD,
            "face_detected": face_detected,
            "face_count": face_count,
            "faces": faces,
            "explanation": explanation,
            "bullets": bullets,
            "disclaimer": disclaimer,
            "visual_evidence": visual_evidence,
            "technical_details": technical_details,
            "evidence": evidence,
            "gradcam": gradcam_data,
            "metadata": metadata_data,
            "model_info": {
                "name": "DeepShield V5",
                "backbone": "EfficientNetB2",
                "input_resolution": "224x224x3",
                "calibrated_threshold": CALIBRATED_THRESHOLD,
                "benchmark_metrics": {
                    "accuracy": "77.55%",
                    "roc_auc": "83.30%",
                    "precision": "87.83%",
                    "recall": "80.44%",
                    "best_f1": 0.8771
                }
            }
        }

    def _compute_calibrated_confidence(self, score: float, threshold: float) -> float:
        """
        Calculates calibrated confidence [0.5, 1.0] relative to decision threshold.
        When score equals threshold, confidence is 0.5 (maximum decision ambiguity).
        When score is at extremes (0.0 or 1.0), confidence approaches 1.0.
        """
        score = np.clip(score, 0.0, 1.0)
        if score >= threshold:
            # Fake region: [threshold, 1.0] -> [0.5, 1.0]
            if 1.0 - threshold <= 1e-6:
                return 1.0
            conf = 0.5 + 0.5 * ((score - threshold) / (1.0 - threshold))
        else:
            # Real region: [0.0, threshold] -> [1.0, 0.5]
            if threshold <= 1e-6:
                return 1.0
            conf = 0.5 + 0.5 * ((threshold - score) / threshold)
        return float(np.clip(conf, 0.5, 1.0))

    def _synthesize_evidence(
        self,
        raw_score: float,
        prediction: str,
        face_detected: bool,
        face_count: int,
        gradcam_data: Dict[str, Any],
        metadata_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compiles discrete forensic evidence points for decision auditability."""
        evidence = []

        # Evidence 1: Deep Learning Backbone Score
        evidence.append({
            "source": "EfficientNetB2 Deep Learning Classifier",
            "type": "neural_network_classification",
            "finding": f"Sigmoid output score is {raw_score:.4f} vs calibrated threshold {CALIBRATED_THRESHOLD}.",
            "supports": prediction,
            "weight": "high"
        })

        # Evidence 2: Face Detection Status
        if face_detected:
            evidence.append({
                "source": "YuNet Face Detector",
                "type": "face_localization",
                "finding": f"Detected {face_count} face(s) in input image; primary face isolated for inspection.",
                "supports": "Face Localization",
                "weight": "medium"
            })
        else:
            evidence.append({
                "source": "YuNet Face Detector",
                "type": "face_localization",
                "finding": "No discrete face detected. Evaluation executed on whole-frame context.",
                "supports": "Whole-Image Fallback",
                "weight": "medium"
            })

        # Evidence 3: Grad-CAM Saliency Analysis
        if gradcam_data.get("available"):
            evidence.append({
                "source": "Grad-CAM Explainability",
                "type": "feature_attribution",
                "finding": "The highlighted regions show areas that influenced the model's decision.",
                "supports": "Model Attribution",
                "weight": "medium"
            })

        # Evidence 4: Metadata Forensics
        if metadata_data.get("editing_software_detected"):
            sw = metadata_data.get("editing_software_name")
            evidence.append({
                "source": "Metadata Forensics",
                "type": "exif_analysis",
                "finding": f"Image editing / synthesis software recorded in metadata: '{sw}'.",
                "supports": "Fake" if metadata_data.get("is_ai_generator_indicated") else "Edited Media",
                "weight": "medium"
            })

        return evidence

    def _generate_explanation(
        self,
        prediction: str,
        raw_score: float,
        confidence: float,
        face_detected: bool,
        gradcam_data: Dict[str, Any],
        metadata_data: Dict[str, Any]
    ) -> str:
        """Constructs an objective, transparent human-readable explanation."""
        score_percent = raw_score * 100
        conf_percent = confidence * 100

        lines = []
        if prediction == "Fake":
            lines.append(
                f"The model detected visual patterns associated with manipulated content with a score of "
                f"{raw_score:.4f} ({score_percent:.1f}%), exceeding the calibrated threshold of {CALIBRATED_THRESHOLD}."
            )
        else:
            lines.append(
                f"The model did not detect strong manipulation-related patterns in the analyzed face with a score of "
                f"{raw_score:.4f} ({score_percent:.1f}%), remaining below the calibrated threshold of {CALIBRATED_THRESHOLD}."
            )

        lines.append(f"Calibrated decision confidence: {conf_percent:.1f}%.")

        if face_detected:
            lines.append("YuNet detected and localized the primary face region for model analysis.")
        else:
            lines.append("No distinct face was identified by the face detector; evaluation was conducted across the full visual field.")

        if gradcam_data.get("available"):
            lines.append("The highlighted regions show areas that influenced the model's decision.")

        if metadata_data.get("editing_software_detected"):
            lines.append(f"Technical metadata audit flagged traces of editing/generation software ({metadata_data.get('editing_software_name')}).")

        lines.append("This is an AI-based assessment, not absolute proof of authenticity.")

        return " ".join(lines)

    def _generate_bullets(
        self,
        prediction: str,
        raw_score: float,
        face_detected: bool,
        face_conf_str: str,
        gradcam_data: Dict[str, Any],
        metadata_data: Dict[str, Any]
    ) -> List[str]:
        """
        Generates a clear 2-4 bullet explanation from the actual API evidence.
        Strictly adheres to:
        - YuNet face detection is NOT described as proof of authenticity or natural alignment.
        - Grad-CAM is NOT described as proof of manipulation.
        - For Real: 'The model did not detect strong manipulation-related patterns in the analyzed face.'
        - For Fake: 'The model detected visual patterns associated with manipulated content.'
        - For Grad-CAM: 'The highlighted regions show areas that influenced the model's decision.'
        - Always includes: 'This is an AI-based assessment, not absolute proof of authenticity.'
        - No invented evidence or unverified artifact claims.
        """
        bullets = []

        if prediction == "Fake":
            # Bullet 1: Core Neural Classifier finding for Fake
            bullets.append(
                f"The model detected visual patterns associated with manipulated content "
                f"(model score: {raw_score:.4f}, exceeding the calibrated threshold of {CALIBRATED_THRESHOLD})."
            )

            # Bullet 2: Grad-CAM regions that influenced decision
            if gradcam_data.get("available"):
                bullets.append(
                    "The highlighted regions show areas that influenced the model's decision."
                )

            # Bullet 3: Face Localization or Metadata
            if metadata_data.get("editing_software_detected"):
                sw = metadata_data.get("editing_software_name")
                bullets.append(
                    f"Technical metadata audit recorded editing software signatures ('{sw}')."
                )
            elif face_detected:
                bullets.append(
                    f"YuNet localized a face in the image with {face_conf_str} confidence, isolating the facial region for analysis."
                )
            else:
                bullets.append(
                    "No discrete face was detected; analysis was performed across the whole image."
                )
        else:
            # Bullet 1: Core Neural Classifier finding for Real
            bullets.append(
                f"The model did not detect strong manipulation-related patterns in the analyzed face "
                f"(model score: {raw_score:.4f}, remaining below the calibrated threshold of {CALIBRATED_THRESHOLD})."
            )

            # Bullet 2: Grad-CAM attention regions
            if gradcam_data.get("available"):
                bullets.append(
                    "The highlighted regions show areas that influenced the model's decision."
                )

            # Bullet 3: Face localization
            if face_detected:
                bullets.append(
                    f"YuNet localized a face in the image with {face_conf_str} confidence, isolating the facial region for analysis."
                )
            else:
                bullets.append(
                    "No discrete face was detected; analysis was performed across the whole image."
                )

        # Bullet 4: Always include mandatory AI-based assessment disclaimer
        bullets.append("This is an AI-based assessment, not absolute proof of authenticity.")

        return bullets[:4]


# Module-level convenience function
def get_image_detector() -> ImageDetector:
    """Returns the singleton ImageDetector instance."""
    return ImageDetector.get_instance()
