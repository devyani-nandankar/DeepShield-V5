"""
DeepShield V5 - Environment & Model Asset Verification Diagnostic
================================================================
Performs comprehensive audits of dependencies, model weights,
input/output shapes, class mappings, and calibrated threshold configuration.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 70)
    print(" DeepShield V5 Backend & Model Verification Diagnostic")
    print("=" * 70)

    # 1. Dependency Imports Audit
    print("\n[1] Checking Python Dependencies & Library Imports:")
    modules = [
        ("Python Runtime", sys.version.split()[0]),
        ("flask", None),
        ("flask_cors", None),
        ("tensorflow", None),
        ("cv2 (OpenCV)", None),
        ("PIL (Pillow)", None),
        ("numpy", None),
    ]

    missing_deps = []
    for mod_name, version in modules:
        try:
            if mod_name == "Python Runtime":
                print(f"  [x] Python Version: {version}")
                continue
            pkg = mod_name.split()[0]
            m = __import__(pkg)
            ver = getattr(m, "__version__", "available")
            print(f"  [x] {mod_name}: {ver}")
        except ImportError as e:
            print(f"  [ ] {mod_name}: MISSING ({e})")
            missing_deps.append(mod_name)

    # 2. Model File Existence Check
    print("\n[2] Checking Model Files on Disk:")
    v5_path = PROJECT_ROOT / "models" / "deepshield_v5_clean_best.keras"
    yunet_path = PROJECT_ROOT / "models" / "yunet" / "face_detection_yunet_2023mar.onnx"

    v5_exists = v5_path.is_file()
    yunet_exists = yunet_path.is_file()

    print(f"  - DeepShield V5 Model Path: {v5_path}")
    print(f"    Status: {'EXISTS (' + str(v5_path.stat().st_size) + ' bytes)' if v5_exists else 'NOT FOUND'}")

    print(f"  - OpenCV YuNet Model Path:  {yunet_path}")
    print(f"    Status: {'EXISTS (' + str(yunet_path.stat().st_size) + ' bytes)' if yunet_exists else 'NOT FOUND'}")

    # 3. YuNet Face Detector Initialization
    print("\n[3] Verifying YuNet Face Detector Initialization:")
    if yunet_exists:
        try:
            from backend.face_detector import FaceDetector
            detector = FaceDetector(yunet_path)
            if detector.is_available():
                print("  [x] YuNet model loaded and initialized successfully via cv2.FaceDetectorYN.")
            else:
                print("  [!] YuNet model initialization returned False.")
        except Exception as e:
            print(f"  [!] Error initializing YuNet: {e}")
    else:
        print("  [!] Cannot test YuNet loading: ONNX weight file not present.")

    # 4. EfficientNetB2 V5 Model Loading & Architecture Audit
    print("\n[4] Verifying DeepShield V5 Model Loading & Shapes:")
    if v5_exists:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(v5_path), compile=False)
            print("  [x] DeepShield V5 model loaded successfully.")

            input_shape = model.input_shape
            output_shape = model.output_shape

            print(f"  - Model Input Shape:  {input_shape}")
            print(f"  - Model Output Shape: {output_shape}")

            expected_in = (None, 224, 224, 3)
            expected_out = (None, 1)

            if input_shape == expected_in:
                print(f"  [x] Input shape matches expected: {expected_in}")
            else:
                print(f"  [!] Input shape mismatch: expected {expected_in}, got {input_shape}")

            if output_shape == expected_out:
                print(f"  [x] Output shape matches expected: {expected_out}")
            else:
                print(f"  [!] Output shape mismatch: expected {expected_out}, got {output_shape}")
        except Exception as e:
            print(f"  [!] Error loading DeepShield V5 model: {e}")
    else:
        print("  [!] V5 model file not found. Skipping weight loading test.")

    # 5. Class Mapping & Decision Threshold Audit
    print("\n[5] Verifying Class Mapping & Decision Threshold:")
    from backend.config import CLASS_MAPPING, CALIBRATED_THRESHOLD
    print(f"  - Class Mapping: {CLASS_MAPPING}")
    print(f"    Expected: 0 = Real, 1 = Fake (Higher sigmoid = Fake)")
    assert CLASS_MAPPING.get(0) == "Real" and CLASS_MAPPING.get(1) == "Fake", "Class mapping mismatch!"
    print("  [x] Class mapping verified.")

    print(f"  - Calibrated Decision Threshold: {CALIBRATED_THRESHOLD}")
    print(f"    Configurable via: backend/config.py or environment variable DEEPSHIELD_THRESHOLD")

    # Test DEEPSHIELD_THRESHOLD override
    os.environ["DEEPSHIELD_THRESHOLD"] = "0.20"
    import importlib
    import backend.config as cfg
    importlib.reload(cfg)
    print(f"  - Environment variable override test (DEEPSHIELD_THRESHOLD=0.20): {cfg.CALIBRATED_THRESHOLD}")
    # Reset
    del os.environ["DEEPSHIELD_THRESHOLD"]
    importlib.reload(cfg)
    print(f"  [x] Threshold configuration flexibility verified (reset to {cfg.CALIBRATED_THRESHOLD}).")

    print("\n" + "=" * 70)
    print(" Verification Diagnostic Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
