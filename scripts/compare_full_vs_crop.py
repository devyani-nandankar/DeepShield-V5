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
from backend.config import CALIBRATED_THRESHOLD

detector = get_image_detector()
fd = get_face_detector()

results = {}

for name in ['real.jpeg', 'fake.jpeg']:
    p = PROJECT_ROOT / 'backend' / 'uploads' / name
    img_bgr = cv2.imread(str(p))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Full Image Score (Direct resize to 224x224)
    full_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    full_tensor = np.expand_dims(full_resized.astype(np.float32), axis=0)
    full_pred = detector.model.predict(full_tensor, verbose=0)
    full_score = float(full_pred[0][0])

    # 2. YuNet Face Crop Score
    res = fd.detect_faces(img_bgr)
    if res.face_detected and res.crop is not None and res.crop.size > 0:
        crop_rgb = cv2.cvtColor(res.crop, cv2.COLOR_BGR2RGB)
    else:
        crop_rgb = img_rgb
    crop_resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    crop_tensor = np.expand_dims(crop_resized.astype(np.float32), axis=0)
    crop_pred = detector.model.predict(crop_tensor, verbose=0)
    crop_score = float(crop_pred[0][0])

    results[name] = {
        'full': full_score,
        'crop': crop_score
    }

print("=" * 60)
print(" DeepShield V5 Inference Comparison: Full Image vs Face Crop")
print(f" Calibrated Decision Threshold: {CALIBRATED_THRESHOLD}")
print("=" * 60)

real_full = results['real.jpeg']['full']
real_crop = results['real.jpeg']['crop']
fake_full = results['fake.jpeg']['full']
fake_crop = results['fake.jpeg']['crop']

print(f"Real Full-Image Score:  {real_full:.6f}  (Verdict: {'Fake' if real_full >= CALIBRATED_THRESHOLD else 'Real'})")
print(f"Real Face-Crop Score:   {real_crop:.6f}  (Verdict: {'Fake' if real_crop >= CALIBRATED_THRESHOLD else 'Real'})")
print(f"Fake Full-Image Score:  {fake_full:.6f}  (Verdict: {'Fake' if fake_full >= CALIBRATED_THRESHOLD else 'Real'})")
print(f"Fake Face-Crop Score:   {fake_crop:.6f}  (Verdict: {'Fake' if fake_crop >= CALIBRATED_THRESHOLD else 'Real'})")
print("=" * 60)

# Separation analysis
sep_full = fake_full - real_full
sep_crop = fake_crop - real_crop

print(f"\nSeparation (Fake Score - Real Score):")
print(f"  Full Image Separation: {sep_full:+.6f}")
print(f"  Face Crop Separation:  {sep_crop:+.6f}")
