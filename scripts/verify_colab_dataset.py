"""
DeepShield Phase 2 - Google Colab Dataset Verification Script

Purpose:
Verifies the dataset structure, image counts, manifest integrity, column definitions,
label mappings, distribution, missing files, duplicate entries, video_group data leakage,
and face_detected information for DeepShield V5 on Google Colab.

Dataset path:
/content/drive/MyDrive/DeepShield_Datasets/Face_Cropped_FAST10/

Manifest path:
/content/drive/MyDrive/DeepShield_Datasets/Metadata/face_crop_FAST10_manifest_split.csv

Expected counts:
- train: 17,208
- validation: 3,688
- test: 3,693

Class mapping:
- 0 = Real
- 1 = Fake
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 1. Configuration & Expected Constants
# ==========================================
DRIVE_MOUNT_POINT = "/content/drive"
DATASET_ROOT = Path("/content/drive/MyDrive/DeepShield_Datasets/Face_Cropped_FAST10")
MANIFEST_PATH = Path("/content/drive/MyDrive/DeepShield_Datasets/Metadata/face_crop_FAST10_manifest_split.csv")

EXPECTED_COUNTS = {
    "train": 17208,
    "validation": 3688,
    "test": 3693
}

CLASS_MAPPING = {
    0: "Real",
    1: "Fake"
}

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def print_section(title: str):
    print(f"\n{'='*70}\n[Phase 2 Verification] {title}\n{'='*70}")


# ==========================================
# Step 1: Google Drive Mounting Verification
# ==========================================
def verify_google_drive():
    print_section("1. Google Drive Mounting Check")
    if not os.path.exists(DRIVE_MOUNT_POINT):
        print(f"[-] Google Drive mount point '{DRIVE_MOUNT_POINT}' not found.")
        print("[*] Attempting Colab drive mount...")
        try:
            from google.colab import drive
            drive.mount(DRIVE_MOUNT_POINT)
            print("[+] Google Drive successfully mounted.")
        except Exception as e:
            print(f"[!] Critical Error: Unable to mount Google Drive: {e}")
            sys.exit(1)
    else:
        print(f"[+] Google Drive is mounted at '{DRIVE_MOUNT_POINT}'.")


# ==========================================
# Step 2: Dataset Path Verification
# ==========================================
def verify_dataset_path():
    print_section("2. Dataset Path Check")
    if not DATASET_ROOT.exists():
        print(f"[!] Critical Error: Dataset directory not found at: {DATASET_ROOT}")
        sys.exit(1)
    print(f"[+] Dataset root path verified: {DATASET_ROOT}")


# ==========================================
# Steps 3, 4, 5 & 6: Directory Structure & File Counts
# ==========================================
def verify_directories_and_counts():
    print_section("3, 4, 5 & 6. Directory Structure & Image Counts")
    split_file_counts = {}
    all_passed = True

    for split_name, expected_count in EXPECTED_COUNTS.items():
        split_dir = DATASET_ROOT / split_name
        
        # Check directory existence
        if not split_dir.exists():
            print(f"[!] Directory missing for split '{split_name}': {split_dir}")
            all_passed = False
            continue

        # Count image files inside split directory recursively
        files = [
            p for p in split_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        actual_count = len(files)
        split_file_counts[split_name] = actual_count

        status = "[+] PASS" if actual_count == expected_count else "[!] MISMATCH"
        print(f"{status} -> Split: '{split_name}': Found {actual_count:,} images (Expected: {expected_count:,})")
        if actual_count != expected_count:
            all_passed = False

    return split_file_counts, all_passed


# ==========================================
# Steps 7, 8, 9, 10, 11, 12, 13, 14: Manifest & Data Auditing
# ==========================================
def verify_manifest():
    print_section("7. Manifest File Existence & Load")
    if not MANIFEST_PATH.exists():
        print(f"[!] Critical Error: Manifest CSV not found at: {MANIFEST_PATH}")
        sys.exit(1)

    print(f"[+] Manifest file verified: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH)
    print(f"[+] Total manifest entries loaded: {len(df):,}")

    # -------------------------------------------------------------
    # Step 8: Required Columns
    # -------------------------------------------------------------
    print_section("8. Required Columns Check")
    print(f"Columns present in manifest: {list(df.columns)}")
    
    # Identify split, label, file path, video group, and face detected columns
    split_col = next((c for c in ['split', 'dataset_split', 'subset'] if c in df.columns), None)
    label_col = next((c for c in ['label', 'target', 'class'] if c in df.columns), None)
    path_col = next((c for c in ['filepath', 'file_path', 'relative_path', 'path', 'image_path', 'filename'] if c in df.columns), None)
    video_group_col = next((c for c in ['video_group', 'video_id', 'group', 'video', 'original_video'] if c in df.columns), None)
    face_detected_col = next((c for c in ['face_detected', 'face_found', 'is_face', 'num_faces'] if c in df.columns), None)

    col_status = {
        "split column": split_col,
        "label column": label_col,
        "filepath column": path_col,
        "video_group column": video_group_col,
        "face_detected column": face_detected_col,
    }

    for desc, col_name in col_status.items():
        if col_name:
            print(f"[+] Found {desc}: '{col_name}'")
        else:
            print(f"[!] Warning/Missing {desc}")

    # -------------------------------------------------------------
    # Step 9 & 10: Label Mapping & Label Distribution
    # -------------------------------------------------------------
    print_section("9 & 10. Label Mapping & Distribution Analysis")
    if label_col:
        unique_labels = sorted(df[label_col].unique())
        print(f"Unique label values in manifest: {unique_labels}")
        print("Expected class mapping: 0 = Real, 1 = Fake")

        if split_col:
            for s in EXPECTED_COUNTS.keys():
                subset = df[df[split_col].astype(str).str.lower() == s]
                print(f"\n--- Split: {s} (Total: {len(subset):,}) ---")
                dist = subset[label_col].value_counts().to_dict()
                for lbl, count in dist.items():
                    class_name = CLASS_MAPPING.get(lbl, "Unknown")
                    pct = (count / len(subset) * 100) if len(subset) > 0 else 0
                    print(f"    Label {lbl} ({class_name}): {count:,} ({pct:.2f}%)")
        else:
            print("\nOverall Label Distribution:")
            dist = df[label_col].value_counts().to_dict()
            for lbl, count in dist.items():
                class_name = CLASS_MAPPING.get(lbl, "Unknown")
                print(f"    Label {lbl} ({class_name}): {count:,}")
    else:
        print("[!] Label column not detected; skipping distribution analysis.")

    # Check manifest row counts per split against expected counts
    if split_col:
        print("\nManifest split count validation:")
        for s, exp in EXPECTED_COUNTS.items():
            actual = (df[split_col].astype(str).str.lower() == s).sum()
            status = "[+] PASS" if actual == exp else "[!] MISMATCH"
            print(f"  {status} -> Manifest split '{s}': {actual:,} (Expected: {exp:,})")

    # -------------------------------------------------------------
    # Step 11: Missing Files Verification (Manifest vs Disk)
    # -------------------------------------------------------------
    print_section("11. Missing Files Check (Manifest vs Disk)")
    if path_col:
        missing_files = []
        sample_checked = 0
        
        # Check existence of files
        for _, row in df.iterrows():
            fpath_str = str(row[path_col]).strip().replace("\\", "/")
            # Handle absolute or relative paths
            candidate_path = Path(fpath_str)
            if not candidate_path.is_absolute():
                # Try relative to DATASET_ROOT or split
                if split_col and str(row[split_col]).lower() in fpath_str:
                    full_path = DATASET_ROOT / fpath_str
                elif split_col:
                    full_path = DATASET_ROOT / str(row[split_col]).lower() / fpath_str
                else:
                    full_path = DATASET_ROOT / fpath_str
            else:
                full_path = candidate_path

            sample_checked += 1
            if not full_path.exists():
                missing_files.append((fpath_str, str(full_path)))
                if len(missing_files) <= 5:
                    print(f"  [!] Missing file: {fpath_str} (Checked: {full_path})")

        if len(missing_files) == 0:
            print(f"[+] PASS: All {sample_checked:,} files referenced in the manifest exist on disk.")
        else:
            print(f"[!] Total missing files found: {len(missing_files):,} out of {sample_checked:,}")
    else:
        print("[!] Filepath column not found; skipping disk path verification.")

    # -------------------------------------------------------------
    # Step 12: Duplicate Entries
    # -------------------------------------------------------------
    print_section("12. Duplicate Check")
    if path_col:
        duplicates = df.duplicated(subset=[path_col], keep=False)
        num_duplicates = duplicates.sum()
        if num_duplicates == 0:
            print(f"[+] PASS: No duplicate entries found for file paths in manifest.")
        else:
            print(f"[!] Found {num_duplicates:,} duplicate entries based on column '{path_col}'.")
    else:
        total_dups = df.duplicated().sum()
        print(f"Overall duplicate rows: {total_dups:,}")

    # -------------------------------------------------------------
    # Step 13: Data Leakage Verification (video_group)
    # -------------------------------------------------------------
    print_section("13. Video Group Leakage Check")
    if video_group_col and split_col:
        groups_by_split = {}
        for s in EXPECTED_COUNTS.keys():
            subset = df[df[split_col].astype(str).str.lower() == s]
            groups_by_split[s] = set(subset[video_group_col].dropna().unique())
            print(f"[+] Split '{s}': {len(groups_by_split[s]):,} unique '{video_group_col}' values.")

        train_val_overlap = groups_by_split.get("train", set()).intersection(groups_by_split.get("validation", set()))
        train_test_overlap = groups_by_split.get("train", set()).intersection(groups_by_split.get("test", set()))
        val_test_overlap = groups_by_split.get("validation", set()).intersection(groups_by_split.get("test", set()))

        leakage = False
        if train_val_overlap:
            leakage = True
            print(f"[!] LEAKAGE DETECTED: {len(train_val_overlap)} groups shared between Train and Validation!")
        if train_test_overlap:
            leakage = True
            print(f"[!] LEAKAGE DETECTED: {len(train_test_overlap)} groups shared between Train and Test!")
        if val_test_overlap:
            leakage = True
            print(f"[!] LEAKAGE DETECTED: {len(val_test_overlap)} groups shared between Validation and Test!")

        if not leakage:
            print(f"[+] PASS: Zero group leakage detected across train, validation, and test splits.")
    else:
        print("[*] video_group or split column not available to assess split group leakage.")

    # -------------------------------------------------------------
    # Step 14: Face Detected Information
    # -------------------------------------------------------------
    print_section("14. Face Detected Information Check")
    if face_detected_col:
        print(f"Summary of '{face_detected_col}':")
        print(df[face_detected_col].value_counts(dropna=False))
        
        # Check if any rows failed face detection
        if df[face_detected_col].dtype == bool:
            no_faces = (df[face_detected_col] == False).sum()
        elif np.issubdtype(df[face_detected_col].dtype, np.number):
            no_faces = (df[face_detected_col] == 0).sum()
        else:
            no_faces = df[face_detected_col].astype(str).str.lower().isin(['0', 'false', 'no']).sum()

        if no_faces > 0:
            print(f"[!] Attention: {no_faces:,} entries have face_detected = False / 0.")
        else:
            print(f"[+] PASS: All entries have valid detected faces.")
    else:
        print("[*] Note: Explicit 'face_detected' column not found in manifest; all samples are assumed pre-cropped.")


# ==========================================
# Main Execution Entrypoint
# ==========================================
def main():
    print("="*70)
    print("      DEEPSHIELD PHASE 2: DATASET AUDIT & VERIFICATION SCRIPT")
    print("="*70)

    # 1. Drive mount
    verify_google_drive()

    # 2. Dataset path
    verify_dataset_path()

    # 3-6. Directory structure and image counts
    verify_directories_and_counts()

    # 7-14. Manifest, columns, distribution, leakage, duplicates, missing files
    verify_manifest()

    print_section("AUDIT COMPLETE")
    print("Dataset verification routine finished. No modifications made to the dataset.")


if __name__ == "__main__":
    main()
