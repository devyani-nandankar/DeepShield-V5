"""
DeepShield V5 - Google Colab Environment & Dataset Setup Script
===============================================================

File: training/colab_setup_v5.py

Purpose:
Prepares and verifies the Google Colab environment, GPU acceleration,
dataset folder structure, image counts, manifest file, and training script
before initiating Phase 3 DeepShield V5 training.

NOTE:
- This script DOES NOT start model training.
- It only performs diagnostics and verification.
"""

import os
import sys
from pathlib import Path

# Expected constants
DATASET_ROOT = Path("/content/drive/MyDrive/DeepShield_Datasets/Face_Cropped_FAST10")
MANIFEST_PATH = Path("/content/drive/MyDrive/DeepShield_Datasets/Metadata/face_crop_FAST10_manifest_split.csv")

# Local project path (for reference only):
# C:\Users\ADMIN\Desktop\DeepShield_V5_Final\training\train_deepshield_v5.py
# In Google Colab, the script is expected to be uploaded or placed at:
COLAB_TRAINING_SCRIPT = Path("/content/training/train_deepshield_v5.py")

EXPECTED_COUNTS = {
    "train": 17208,
    "validation": 3688,
    "test": 3693
}

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==============================================================================
# 1. MOUNT GOOGLE DRIVE
# ==============================================================================
def mount_drive() -> bool:
    print("\n" + "=" * 60)
    print("STEP 1: MOUNTING GOOGLE DRIVE")
    print("=" * 60)
    mount_point = Path("/content/drive")
    mydrive_path = mount_point / "MyDrive"

    if mydrive_path.exists():
        print(f"[+] Google Drive already mounted at '{mount_point}'.")
        return True

    try:
        from google.colab import drive
        print("[*] Mounting Google Drive to '/content/drive'...")
        drive.mount('/content/drive')
        if mydrive_path.exists():
            print("[+] Google Drive successfully mounted.")
            return True
        else:
            print("[!] Warning: Mount finished but MyDrive path not detected.")
            return False
    except ImportError:
        print("[*] google.colab not available; running in local/standalone environment.")
        return mydrive_path.exists()
    except Exception as exc:
        print(f"[!] Error mounting Google Drive: {exc}")
        return False


# ==============================================================================
# 2, 3 & 4. CHECK TENSORFLOW & GPU AVAILABILITY
# ==============================================================================
def check_gpu() -> bool:
    print("\n" + "=" * 60)
    print("STEP 2, 3 & 4: TENSORFLOW & GPU CHECK")
    print("=" * 60)
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU Availability: True ({len(gpus)} device(s) found)")
            for i, gpu in enumerate(gpus):
                details = tf.config.experimental.get_device_details(gpu)
                device_name = details.get('device_name', gpu.name)
                print(f"GPU Device [{i}]: {device_name}")
            print("\nGPU READY")
            return True
        else:
            print("GPU Availability: False (No GPU detected)")
            print("\nGPU NOT AVAILABLE")
            return False
    except Exception as exc:
        print(f"[!] Error during TensorFlow GPU check: {exc}")
        print("\nGPU NOT AVAILABLE")
        return False


# ==============================================================================
# 5. VERIFY DATASET ROOT
# ==============================================================================
def check_dataset_root() -> bool:
    print("\n" + "=" * 60)
    print("STEP 5: DATASET ROOT CHECK")
    print("=" * 60)
    print(f"Target Path: {DATASET_ROOT}")
    if DATASET_ROOT.exists() and DATASET_ROOT.is_dir():
        print("[+] Dataset directory exists.")
        print("\nDATASET READY")
        return True
    else:
        print(f"[!] Dataset directory not found at: {DATASET_ROOT}")
        return False


# ==============================================================================
# 6 & 8. VERIFY SPLITS AND IMAGE COUNTS
# ==============================================================================
def check_splits_and_counts() -> dict:
    print("\n" + "=" * 60)
    print("STEP 6 & 8: SPLITS & IMAGE COUNTS VERIFICATION")
    print("=" * 60)
    results = {}

    for split_name, expected_count in EXPECTED_COUNTS.items():
        split_dir = DATASET_ROOT / split_name
        if not split_dir.exists():
            print(f"[!] Split directory missing: {split_dir}")
            results[split_name] = False
            continue

        images = [
            f for f in split_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        actual_count = len(images)
        is_pass = (actual_count == expected_count)
        status_tag = "PASS" if is_pass else "FAIL (Count mismatch)"
        print(f"Split: '{split_name}' -> Found {actual_count:,} images (Expected: {expected_count:,}) [{status_tag}]")
        results[split_name] = is_pass

    return results


# ==============================================================================
# 7. VERIFY MANIFEST
# ==============================================================================
def check_manifest() -> bool:
    print("\n" + "=" * 60)
    print("STEP 7: MANIFEST FILE VERIFICATION")
    print("=" * 60)
    print(f"Target Path: {MANIFEST_PATH}")
    if MANIFEST_PATH.exists() and MANIFEST_PATH.is_file():
        try:
            import pandas as pd
            df = pd.read_csv(MANIFEST_PATH)
            print(f"[+] Manifest file verified. Total records: {len(df):,}")
            print(f"[+] Columns present: {list(df.columns)}")
            print("\nMANIFEST READY")
            return True
        except Exception as exc:
            print(f"[!] Failed to parse manifest CSV: {exc}")
            return False
    else:
        print(f"[!] Manifest not found at: {MANIFEST_PATH}")
        return False


# ==============================================================================
# 9. VERIFY V5 TRAINING SCRIPT
# ==============================================================================
def check_training_script() -> bool:
    print("\n" + "=" * 60)
    print("STEP 9: V5 TRAINING SCRIPT CHECK")
    print("=" * 60)
    print("Expected Colab training script:")
    print(f" {COLAB_TRAINING_SCRIPT}")
    if COLAB_TRAINING_SCRIPT.exists() and COLAB_TRAINING_SCRIPT.is_file():
        print(f"[+] Training script verified ({COLAB_TRAINING_SCRIPT.stat().st_size:,} bytes).")
        return True
    else:
        print(f"[!] Training script not found at: {COLAB_TRAINING_SCRIPT}")
        return False


# ==============================================================================
# 10. PRINT TRAINING COMMAND (WITHOUT EXECUTING)
# ==============================================================================
def print_training_command():
    print("\n" + "=" * 60)
    print("STEP 10: TRAINING EXECUTION COMMAND")
    print("=" * 60)
    print("When you are ready to start training, run the following command manually:")
    print(f"\n    python {COLAB_TRAINING_SCRIPT}\n")
    print("(This setup script will NOT execute the training command.)")


# ==============================================================================
# MAIN SUMMARY
# ==============================================================================
def main():
    mount_drive()
    gpu_ok = check_gpu()
    dataset_ok = check_dataset_root()
    splits_status = check_splits_and_counts()
    manifest_ok = check_manifest()
    script_ok = check_training_script()
    print_training_command()

    train_ok = splits_status.get("train", False)
    val_ok = splits_status.get("validation", False)
    test_ok = splits_status.get("test", False)

    all_critical_passed = (
        gpu_ok and
        dataset_ok and
        manifest_ok and
        train_ok and
        val_ok and
        test_ok and
        script_ok
    )

    print("\n" + "=" * 50)
    print("DEEPSHIELD V5 COLAB SETUP")
    print("=" * 50)
    print(f"GPU: {'PASS' if gpu_ok else 'FAIL'}")
    print(f"Dataset: {'PASS' if dataset_ok else 'FAIL'}")
    print(f"Manifest: {'PASS' if manifest_ok else 'FAIL'}")
    print(f"Train: {'PASS' if train_ok else 'FAIL'}")
    print(f"Validation: {'PASS' if val_ok else 'FAIL'}")
    print(f"Test: {'PASS' if test_ok else 'FAIL'}")
    print(f"Training script: {'PASS' if script_ok else 'FAIL'}")
    print("\nSTATUS:")
    if all_critical_passed:
        print("READY FOR V5 TRAINING")
    else:
        print("NOT READY - Please address failing checks above before training.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
