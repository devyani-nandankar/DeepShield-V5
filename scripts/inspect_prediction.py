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
from backend.config import CALIBRATED_THRESHOLD, CLASS_MAPPING

detector = get_image_detector()
face_det = get_face_detector()

for img_name in ['real.jpeg', 'fake.jpeg']:
    p = Path('backend/uploads') / img_name
    print(f"\n==========================================")
    print(f" INSPECTING: {img_name}")
    print(f"==========================================")
    img_bgr = cv2.imread(str(p))
    if img_bgr is None:
        print(f"Could not read {p}")
        continue

    print(f"Original image shape (H, W, C): {img_bgr.shape}")

    # 1. YuNet Detection
    res = face_det.detect_faces(img_bgr)
    print(f"1. YuNet face_detected: {res.face_detected}")
    print(f"   Face count: {res.face_count}")
    print(f"   Faces: {res.faces}")
    if res.crop is not None:
        print(f"   Crop shape: {res.crop.shape}")

    # 2. Check face crop vs full image
    if res.face_detected and res.crop is not None and res.crop.size > 0:
        target_crop = cv2.cvtColor(res.crop, cv2.COLOR_BGR2RGB)
        print("2. Face crop selected: YES (crop passed to model)")
    else:
        target_crop = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print("2. Face crop selected: NO (full image fallback)")

    # 3. Resizing
    face_resized = cv2.resize(target_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    print(f"3. Resized shape: {face_resized.shape} (target: 224x224)")

    # 4. Preprocessing check
    input_tensor = np.expand_dims(face_resized.astype(np.float32), axis=0)
    print(f"4. Preprocessing: input_tensor dtype={input_tensor.dtype}, min={input_tensor.min()}, max={input_tensor.max()}")

    # 5. Raw model inference
    raw_pred = detector.model.predict(input_tensor, verbose=0)
    raw_score = float(raw_pred[0][0])
    print(f"5. Raw Sigmoid Score: {raw_score:.6f}")

    # Also test predicting on the FULL image (resized directly to 224x224)
    full_resized = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_LINEAR)
    full_tensor = np.expand_dims(full_resized.astype(np.float32), axis=0)
    full_pred = detector.model.predict(full_tensor, verbose=0)
    full_score = float(full_pred[0][0])
    print(f"   (Comparison) Full-image Sigmoid Score: {full_score:.6f}")

    # 6. Prediction verdict
    is_fake = raw_score >= CALIBRATED_THRESHOLD
    verdict = "Fake" if is_fake else "Real"
    print(f"6. Verdict at threshold {CALIBRATED_THRESHOLD}: {verdict}")
    print(f"   Interpretation: 0=Real, 1=Fake, score={raw_score:.6f} >= {CALIBRATED_THRESHOLD} is {is_fake}")
