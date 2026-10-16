import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from backend.image_detector import get_image_detector
from backend.face_detector import get_face_detector
from backend.gradcam import GradCAMExplainer

detector = get_image_detector()
print(f"Model loaded: {detector.is_model_loaded()}")
print(f"Explainer initialized: {detector.explainer is not None}")
if detector.explainer:
    print(f"Explainer target layer: {detector.explainer.target_layer_name}")
    print(f"Explainer grad_model: {detector.explainer.grad_model}")

for name in ['real.jpeg', 'fake.jpeg']:
    p = PROJECT_ROOT / 'backend' / 'uploads' / name
    print(f"\n{'='*50}\nTesting Grad-CAM on: {name}\n{'='*50}")
    
    # Run full predict
    res = detector.predict(str(p))
    score = res.get('score')
    prediction = res.get('prediction')
    gradcam = res.get('gradcam', {})
    
    print(f"Score: {score}")
    print(f"Prediction: {prediction}")
    print(f"Grad-CAM Available: {gradcam.get('available')}")
    print(f"Grad-CAM Heatmap URL: {gradcam.get('heatmap_url')}")
    print(f"Grad-CAM Overlay URL: {gradcam.get('overlay_url')}")
    print(f"Grad-CAM Peak Intensity: {gradcam.get('peak_intensity')}")
    print(f"Grad-CAM Mean Intensity: {gradcam.get('mean_intensity')}")
    print(f"Grad-CAM Salient Ratio: {gradcam.get('salient_ratio')}")
    print(f"Grad-CAM Error: {gradcam.get('error')}")
