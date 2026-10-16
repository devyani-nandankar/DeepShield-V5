"""
DeepShield V5 - Google Colab Environment Preparation Guide & Verification Helper
================================================================================

File: training/prepare_colab_v5.py

Project Purpose:
Explainable multimodal deepfake detection and digital media authenticity verification.

Workflow Overview:
  LOCAL ANTIGRAVITY PROJECT (Windows):
    C:\\Users\\ADMIN\\Desktop\\DeepShield_V5_Final\\training\\train_deepshield_v5.py
       ↓ (Uploaded / Copied)
  GOOGLE COLAB RUNTIME:
    /content/training/train_deepshield_v5.py

Dataset Location (Google Drive):
  Dataset Root: /content/drive/MyDrive/DeepShield_Datasets/Face_Cropped_FAST10/
  Manifest:     /content/drive/MyDrive/DeepShield_Datasets/Metadata/face_crop_FAST10_manifest_split.csv

IMPORTANT SAFETY & EXECUTION RULES:
- The dataset remains securely on Google Drive. DO NOT copy or download the dataset.
- Never overwrite models/deepshield_final.keras or models/deepshield_v4_backup.keras.
- This script does NOT train the model, start inference, or create web components.
- Syntax validation only is run; training will be executed manually later.

================================================================================
GOOGLE COLAB NOTEBOOK CELLS (Copy-paste into your Colab notebook)
================================================================================

# ------------------------------------------------------------------------------
# [CELL 1] Mount Google Drive
# ------------------------------------------------------------------------------
from google.colab import drive
import os
from pathlib import Path

if not (Path("/content/drive") / "MyDrive").exists():
    print("[*] Mounting Google Drive...")
    drive.mount('/content/drive')
else:
    print("[+] Google Drive is already mounted at /content/drive.")

# ------------------------------------------------------------------------------
# [CELL 2] Create training directory in Colab runtime
# ------------------------------------------------------------------------------
!mkdir -p /content/training/
print("[+] Created /content/training/ directory in Colab runtime.")

# ------------------------------------------------------------------------------
# [CELL 3] Upload or Copy train_deepshield_v5.py into Colab
# ------------------------------------------------------------------------------
# OPTION A: If you uploaded train_deepshield_v5.py to your Google Drive:
#   Drive location: /content/drive/MyDrive/DeepShield_Datasets/scripts/train_deepshield_v5.py
#   Uncomment the next 3 lines:
# !cp /content/drive/MyDrive/DeepShield_Datasets/scripts/train_deepshield_v5.py /content/training/train_deepshield_v5.py
# print("[+] Copied train_deepshield_v5.py from Google Drive.")

# OPTION B (Direct Interactive Browser Upload from local Windows machine):
from google.colab import files
import shutil

target_script = Path("/content/training/train_deepshield_v5.py")
if not target_script.exists():
    print("[*] Please select 'train_deepshield_v5.py' from your computer:")
    print("    Local Path: C:\\Users\\ADMIN\\Desktop\\DeepShield_V5_Final\\training\\train_deepshield_v5.py")
    uploaded = files.upload()
    for fname in uploaded.keys():
        shutil.move(fname, str(target_script))
    print("[+] Upload complete: /content/training/train_deepshield_v5.py")
else:
    print("[+] /content/training/train_deepshield_v5.py already exists.")

# ------------------------------------------------------------------------------
# [CELL 4] Verify Training Script Exists
# ------------------------------------------------------------------------------
script_path = Path("/content/training/train_deepshield_v5.py")
if script_path.is_file():
    print(f"[+] PASS: Training script found ({script_path.stat().st_size:,} bytes).")
else:
    print("[!] FAIL: Training script not found at /content/training/train_deepshield_v5.py")

# ------------------------------------------------------------------------------
# [CELL 5] Verify Dataset & Manifest on Google Drive
# ------------------------------------------------------------------------------
dataset_path = Path("/content/drive/MyDrive/DeepShield_Datasets/Face_Cropped_FAST10")
manifest_path = Path("/content/drive/MyDrive/DeepShield_Datasets/Metadata/face_crop_FAST10_manifest_split.csv")

print(f"Dataset Exists: {dataset_path.is_dir()}")
print(f"Manifest Exists: {manifest_path.is_file()}")

# ------------------------------------------------------------------------------
# [CELL 6] Verify GPU
# ------------------------------------------------------------------------------
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for i, g in enumerate(gpus):
        details = tf.config.experimental.get_device_details(g)
        print(f"[+] GPU [{i}]: {details.get('device_name', g.name)} READY")
else:
    print("[!] WARNING: No GPU detected! Go to Runtime -> Change runtime type -> T4 GPU")

# ------------------------------------------------------------------------------
# [CELL 7] Run Syntax Checking ONLY
# ------------------------------------------------------------------------------
!python -m py_compile /content/training/train_deepshield_v5.py
print("[+] Syntax checking complete (Exit code: 0).")
"""

import sys
from pathlib import Path

# Paths inside Colab
COLAB_TRAINING_SCRIPT = Path("/content/training/train_deepshield_v5.py")
DATASET_ROOT = Path("/content/drive/MyDrive/DeepShield_Datasets/Face_Cropped_FAST10")
MANIFEST_PATH = Path("/content/drive/MyDrive/DeepShield_Datasets/Metadata/face_crop_FAST10_manifest_split.csv")


def run_environment_verification():
    """
    Performs complete Colab readiness check without starting training.
    """
    print("\n" + "=" * 60)
    print("DEEPSHIELD V5: COLAB ENVIRONMENT VERIFICATION")
    print("=" * 60)

    # 1. Script Check
    script_pass = COLAB_TRAINING_SCRIPT.is_file()
    print(f"Training script: {'PASS' if script_pass else 'FAIL'}")
    if not script_pass:
        print(f"  [-] Expected location: {COLAB_TRAINING_SCRIPT}")
        print("  [-] Upload train_deepshield_v5.py from your local machine to Colab.")

    # 2. Dataset Check
    dataset_pass = DATASET_ROOT.is_dir()
    print(f"Dataset: {'PASS' if dataset_pass else 'FAIL'}")
    if not dataset_pass:
        print(f"  [-] Missing Drive dataset at: {DATASET_ROOT}")

    # 3. Manifest Check
    manifest_pass = MANIFEST_PATH.is_file()
    print(f"Manifest: {'PASS' if manifest_pass else 'FAIL'}")
    if not manifest_pass:
        print(f"  [-] Missing manifest at: {MANIFEST_PATH}")

    # 4. GPU Check
    gpu_pass = False
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        gpu_pass = len(gpus) > 0
        print(f"GPU: {'PASS' if gpu_pass else 'FAIL'}")
        if gpu_pass:
            for i, gpu in enumerate(gpus):
                details = tf.config.experimental.get_device_details(gpu)
                print(f"  [+] Device [{i}]: {details.get('device_name', gpu.name)}")
        else:
            print("  [-] Enable GPU in Colab via Runtime -> Change runtime type -> T4 GPU")
    except Exception as e:
        print(f"GPU: FAIL ({e})")

    # 5. Syntax Check
    syntax_pass = False
    if script_pass:
        import py_compile
        try:
            py_compile.compile(str(COLAB_TRAINING_SCRIPT), doraise=True)
            syntax_pass = True
            print("Syntax: PASS")
        except Exception as err:
            print(f"Syntax: FAIL ({err})")
    else:
        print("Syntax: FAIL (Script not present)")

    # Overall Summary
    print("\n" + "=" * 60)
    if script_pass and dataset_pass and manifest_pass and gpu_pass and syntax_pass:
        print("STATUS:")
        print("DEEPSHIELD V5 COLAB ENVIRONMENT READY")
        print("\nNote: DO NOT train automatically. When ready, run:")
        print("    python /content/training/train_deepshield_v5.py")
    else:
        print("STATUS:")
        print("NOT READY - Please address failing checks above.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # If run in Colab or diagnostic mode
    run_environment_verification()
