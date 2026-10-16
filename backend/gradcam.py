"""
DeepShield V5 - Explainability Engine (Grad-CAM)
================================================
Generates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps
for DeepShield V5 (EfficientNetB2 backbone) to provide visual explanations
of deepfake manipulation artifacts.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
import tensorflow as tf

from backend.config import GENERATED_FOLDER

logger = logging.getLogger("deepshield.gradcam")


class GradCAMExplainer:
    """
    Grad-CAM generator for DeepShield V5 classification models.
    Identifies salient facial regions driving Real vs. Fake classification.
    """

    def __init__(self, model: tf.keras.Model, target_layer_name: Optional[str] = None):
        """
        Initializes GradCAMExplainer with a trained Keras model.
        
        Args:
            model: Trained tf.keras.Model
            target_layer_name: Optional specific name of the target conv layer.
                               If None, automatically resolves the last 4D conv layer.
        """
        self.model = model
        self.target_layer_name = target_layer_name or self._find_target_layer_name(model)
        self.grad_model = self._build_grad_model()
        logger.info(f"GradCAMExplainer initialized with target layer: '{self.target_layer_name}'")

    def _find_target_layer_name(self, model: tf.keras.Model) -> Optional[str]:
        """
        Recursively searches for the last convolutional or 4D activation layer
        in the model or nested backbone submodels.
        """
        # 1. Direct check for known EfficientNetB2 top conv names
        for preferred in ["top_conv", "top_activation", "block7b_project_conv"]:
            for layer in model.layers:
                if layer.name == preferred:
                    return layer.name
                if hasattr(layer, "layers"):  # Nested model
                    for sub_layer in layer.layers:
                        if sub_layer.name == preferred:
                            return f"{layer.name}/{sub_layer.name}"

        # 2. Search backwards through layers for Conv2D or 4D output
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):  # Nested submodel
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization)):
                        return f"{layer.name}/{sub_layer.name}"
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

        # 3. Fallback: inspect layer output shapes
        for layer in reversed(model.layers):
            try:
                out_shape = layer.output_shape if hasattr(layer, "output_shape") else None
                if out_shape and len(out_shape) == 4 and "input" not in layer.name.lower():
                    return layer.name
            except Exception:
                continue

        return None

    def _build_grad_model(self) -> Optional[tf.keras.Model]:
        """
        Constructs a multi-output Keras model that returns both the
        target feature activations and the final prediction output.
        """
        if not self.target_layer_name:
            logger.warning("No suitable target convolutional layer found for Grad-CAM.")
            return None

        try:
            # Check if target layer is in a nested submodel (e.g. 'efficientnetb2/top_conv')
            if "/" in self.target_layer_name:
                parent_name, child_name = self.target_layer_name.split("/", 1)
                parent_layer = self.model.get_layer(parent_name)
                target_layer = parent_layer.get_layer(child_name)
                
                # Intermediate model for nested backbone
                sub_grad_model = tf.keras.Model(
                    inputs=parent_layer.inputs,
                    outputs=[target_layer.output, parent_layer.output]
                )
                
                # Full model wrapper
                inputs = self.model.inputs
                x = inputs
                for l in self.model.layers:
                    if l.name == parent_name:
                        target_act, x = sub_grad_model(x)
                    elif "input" not in l.name.lower():
                        try:
                            x = l(x)
                        except Exception:
                            pass
                return tf.keras.Model(inputs=inputs, outputs=[target_act, self.model.output])
            else:
                target_layer = self.model.get_layer(self.target_layer_name)
                return tf.keras.Model(
                    inputs=self.model.inputs,
                    outputs=[target_layer.output, self.model.output]
                )
        except Exception as e:
            logger.warning(f"Direct grad_model construction failed: {e}. Trying fallback execution.")
            return None

    def generate(
        self,
        image_tensor: tf.Tensor,
        original_image_rgb: np.ndarray,
        alpha: float = 0.55
    ) -> Dict[str, Any]:
        """
        Computes Grad-CAM for the provided image tensor.

        Args:
            image_tensor: Preprocessed tensor of shape (1, 224, 224, 3)
            original_image_rgb: Original face crop in RGB uint8 format
            alpha: Heatmap blend factor [0.0 = only original, 1.0 = only heatmap]

        Returns:
            Dictionary containing:
                - available: bool
                - heatmap_filename: str or None
                - overlay_filename: str or None
                - heatmap_url: str or None
                - overlay_url: str or None
                - peak_intensity: float
                - mean_intensity: float
                - salient_ratio: float
                - error: str or None
        """
        result = {
            "available": False,
            "heatmap_filename": None,
            "overlay_filename": None,
            "heatmap_url": None,
            "overlay_url": None,
            "peak_intensity": 0.0,
            "mean_intensity": 0.0,
            "salient_ratio": 0.0,
            "error": None
        }

        try:
            heatmap = self._compute_heatmap(image_tensor)
            if heatmap is None or np.isnan(heatmap).any():
                result["error"] = "Grad-CAM computation produced invalid or NaN activations."
                return result

            # Resize heatmap to match original crop resolution
            orig_h, orig_w = original_image_rgb.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # Compute objective heatmap statistics
            peak_val = float(np.max(heatmap_resized))
            mean_val = float(np.mean(heatmap_resized))
            salient_ratio = float(np.sum(heatmap_resized >= 0.5) / heatmap_resized.size)

            # Convert to uint8 colormap [0, 255]
            heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
            heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

            # Generate overlay blend
            overlay_rgb = cv2.addWeighted(
                original_image_rgb,
                1.0 - alpha,
                heatmap_colored_rgb,
                alpha,
                0
            )

            # Persist artifacts in generated folder
            file_token = uuid.uuid4().hex[:12]
            heatmap_fname = f"gradcam_heatmap_{file_token}.jpg"
            overlay_fname = f"gradcam_overlay_{file_token}.jpg"

            heatmap_path = GENERATED_FOLDER / heatmap_fname
            overlay_path = GENERATED_FOLDER / overlay_fname

            cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap_colored_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            result.update({
                "available": True,
                "heatmap_filename": heatmap_fname,
                "overlay_filename": overlay_fname,
                "heatmap_url": f"/api/generated/{heatmap_fname}",
                "overlay_url": f"/api/generated/{overlay_fname}",
                "peak_intensity": round(peak_val, 4),
                "mean_intensity": round(mean_val, 4),
                "salient_ratio": round(salient_ratio, 4),
            })
            return result

        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}", exc_info=True)
            result["error"] = str(e)
            return result

    def _compute_heatmap(self, image_tensor: tf.Tensor) -> Optional[np.ndarray]:
        """
        Executes gradient tape forward and backward pass to extract Grad-CAM weights.
        """
        # Primary strategy: Track backbone top feature activations into classification head
        try:
            bb = None
            head_start_idx = 1
            for idx, layer in enumerate(self.model.layers):
                if hasattr(layer, "layers") or "efficientnet" in layer.name.lower():
                    bb = layer
                    head_start_idx = idx + 1
                    break

            if bb is not None:
                with tf.GradientTape() as tape:
                    conv_outputs = bb(image_tensor, training=False)
                    tape.watch(conv_outputs)
                    x = conv_outputs
                    for layer in self.model.layers[head_start_idx:]:
                        x = layer(x, training=False)
                    loss = x[:, 0]

                grads = tape.gradient(loss, conv_outputs)
                if grads is not None:
                    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
                    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
                    cam = tf.maximum(cam, 0)
                    max_cam = tf.reduce_max(cam)
                    if max_cam > 0:
                        cam = cam / max_cam
                    return cam.numpy()
        except Exception as e:
            logger.warning(f"Backbone Grad-CAM computation error: {e}")

        # Case A: We have a compiled grad_model
        if self.grad_model is not None:
            try:
                with tf.GradientTape() as tape:
                    tape.watch(image_tensor)
                    conv_outputs, predictions = self.grad_model(image_tensor, training=False)
                    loss = predictions[:, 0]

                grads = tape.gradient(loss, conv_outputs)
                if grads is not None:
                    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
                    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
                    cam = tf.maximum(cam, 0)
                    max_cam = tf.reduce_max(cam)
                    if max_cam > 0:
                        cam = cam / max_cam
                    return cam.numpy()
            except Exception as e:
                logger.warning(f"Grad model gradient computation failed: {e}")

        return None
