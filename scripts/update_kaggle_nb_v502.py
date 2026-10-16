import json
from pathlib import Path

nb_path = Path("training/DeepShield_V5_Kaggle.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 4 (index 3) default paths
cell4 = nb["cells"][3]
new_src_4 = []
for line in cell4["source"]:
    if line.startswith("KAGGLE_DATASET_ROOT ="):
        new_src_4.append('KAGGLE_DATASET_ROOT = "/kaggle/input/datasets/devyaninandankar/deepshield-face-dataset/Kaggle_Transfer/Face_Cropped_FAST10"\n')
    elif line.startswith("KAGGLE_MANIFEST_PATH ="):
        new_src_4.append('KAGGLE_MANIFEST_PATH = "/kaggle/input/datasets/devyaninandankar/deepshield-face-dataset/Kaggle_Transfer/face_crop_FAST10_manifest_split.csv"\n')
    else:
        new_src_4.append(line)
cell4["source"] = new_src_4

# 2. Update Setup Cell (index 7) to ensure stale script prevention
setup_cell = nb["cells"][7]
setup_cell["source"] = [
    "# SETUP CELL -- UPLOAD & CONFIGURE TRAINING SCRIPT\n",
    "# Ensures the LATEST path-resolution-fixed script is used, overwriting any stale /kaggle/working/ copy.\n",
    "import os\n",
    "import shutil\n",
    "from pathlib import Path\n",
    "\n",
    "target_script = Path(\"/kaggle/working/train_deepshield_v5.py\")\n",
    "EXPECTED_VERSION = \"5.0.2-path-fix\"\n",
    "\n",
    "# 1. Check attached Kaggle datasets in /kaggle/input for the latest script\n",
    "found_input_script = None\n",
    "if Path(\"/kaggle/input\").exists():\n",
    "    for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
    "        if \"train_deepshield_v5.py\" in files:\n",
    "            p = Path(root) / \"train_deepshield_v5.py\"\n",
    "            found_input_script = p\n",
    "            break\n",
    "\n",
    "# 2. If found in /kaggle/input, ALWAYS copy & overwrite to prevent stale /kaggle/working/ copies\n",
    "if found_input_script:\n",
    "    shutil.copy2(found_input_script, target_script)\n",
    "    print(f\"[+] Synchronized latest training script from: {found_input_script} -> {target_script}\")\n",
    "\n",
    "# 3. Verify target script version and integrity\n",
    "print(\"=\" * 60)\n",
    "if target_script.is_file():\n",
    "    content = target_script.read_text(encoding=\"utf-8\", errors=\"ignore\")\n",
    "    is_latest = EXPECTED_VERSION in content\n",
    "    print(f\"File location:     {target_script}\")\n",
    "    print(f\"File size:         {target_script.stat().st_size:,} bytes\")\n",
    "    print(f\"Script version:    {'LATEST (' + EXPECTED_VERSION + ')' if is_latest else 'STALE / OUTDATED'}\")\n",
    "    print(\"=\" * 60)\n",
    "    if is_latest:\n",
    "        print(\"TRAINING SCRIPT READY (Latest Version Confirmed)\")\n",
    "    else:\n",
    "        print(\"[!] WARNING: /kaggle/working/train_deepshield_v5.py is STALE!\")\n",
    "        print(\"    Please upload the updated train_deepshield_v5.py to /kaggle/working/ to overwrite the old copy.\")\n",
    "else:\n",
    "    print(\"INSTRUCTIONS TO DEPLOY LATEST SCRIPT TO KAGGLE:\")\n",
    "    print(\"1. In Kaggle Notebook, attach the 'deepshield-v5-code' dataset via '+ Add Input'.\")\n",
    "    print(\"   OR upload the local file: training/train_deepshield_v5.py directly into /kaggle/working/.\")\n",
    "    print(\"2. After uploading, re-run this setup cell.\")\n",
    "    print(\"=\" * 60)\n",
    "    print(\"TRAINING SCRIPT NOT FOUND — upload train_deepshield_v5.py to Kaggle first.\")\n",
    "print(\"=\" * 60)\n"
]

# 3. Update Cell 8 (index 8) to verify SCRIPT_VERSION
cell8 = nb["cells"][8]
cell8["source"] = [
    "# CELL 8 -- VERIFY TRAIN_DEEPSHIELD_V5.PY\n",
    "import os\n",
    "import py_compile\n",
    "from pathlib import Path\n",
    "\n",
    "print(\"=\" * 60)\n",
    "print(\"DEEPSHIELD V5: TRAINING SCRIPT INTEGRITY AUDIT\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "script_path = Path(\"/kaggle/working/train_deepshield_v5.py\")\n",
    "script_pass = False\n",
    "\n",
    "if script_path.is_file():\n",
    "    print(f\"Located training script at: {script_path}\")\n",
    "    try:\n",
    "        py_compile.compile(str(script_path), doraise=True)\n",
    "        print(\"[+] Python Syntax Check: PASS\")\n",
    "\n",
    "        with open(script_path, \"r\", encoding=\"utf-8\") as f:\n",
    "            content = f.read()\n",
    "\n",
    "        has_version = \"5.0.2-path-fix\" in content\n",
    "        has_effnet = \"EfficientNetB2\" in content\n",
    "        has_input_size = \"IMAGE_SIZE = (224, 224)\" in content\n",
    "        has_freeze_bn = \"isinstance(layer, layers.BatchNormalization)\" in content\n",
    "        has_output_dir = \"DeepShield_V5\" in content\n",
    "        has_multitier_resolver = \"file_lookup_basename\" in content and \"file_lookup_stem\" in content\n",
    "\n",
    "        print(f\"[+] Version 5.0.2 (Path-Fix):    {'PASS' if has_version else 'FAIL (STALE SCRIPT)'}\")\n",
    "        print(f\"[+] EfficientNetB2 Architecture: {'PASS' if has_effnet else 'FAIL'}\")\n",
    "        print(f\"[+] Input Size 224x224x3:        {'PASS' if has_input_size else 'FAIL'}\")\n",
    "        print(f\"[+] BatchNorm Layers Frozen:     {'PASS' if has_freeze_bn else 'FAIL'}\")\n",
    "        print(f\"[+] Multi-Tier Path Resolver:    {'PASS' if has_multitier_resolver else 'FAIL'}\")\n",
    "        print(f\"[+] Target Output DeepShield_V5: {'PASS' if has_output_dir else 'FAIL'}\")\n",
    "\n",
    "        script_pass = has_version and has_effnet and has_input_size and has_freeze_bn and has_output_dir and has_multitier_resolver\n",
    "    except Exception as exc:\n",
    "        print(f\"[!] Syntax or validation error: {exc}\")\n",
    "else:\n",
    "    print(\"[!] Training script 'train_deepshield_v5.py' not found at /kaggle/working/.\")\n",
    "\n",
    "print(\"=\" * 60)\n",
    "if script_pass:\n",
    "    print(\"TRAINING SCRIPT AUDIT: PASS (Latest Version Verified)\")\n",
    "else:\n",
    "    print(\"TRAINING SCRIPT AUDIT: CHECK REQUIRED\")\n",
    "print(\"=\" * 60)\n"
]

# 4. Update Cell 9 (index 9) to configure environment and print EXACT reporting format
cell9 = nb["cells"][9]
cell9["source"] = [
    "# CELL 9 -- KAGGLE PATH & ENVIRONMENT CONFIGURATION\n",
    "import os\n",
    "import sys\n",
    "import shutil\n",
    "from pathlib import Path\n",
    "\n",
    "print(\"=\" * 60)\n",
    "print(\"DEEPSHIELD V5: KAGGLE ENVIRONMENT CONFIGURATION\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "# 1. Set runtime environment variables for training script\n",
    "os.environ[\"ENVIRONMENT\"] = \"kaggle\"\n",
    "if 'ACTIVE_DATASET_ROOT' in globals() and ACTIVE_DATASET_ROOT:\n",
    "    os.environ[\"KAGGLE_DATASET_ROOT\"] = str(ACTIVE_DATASET_ROOT)\n",
    "if 'ACTIVE_MANIFEST_PATH' in globals() and ACTIVE_MANIFEST_PATH:\n",
    "    os.environ[\"KAGGLE_MANIFEST_PATH\"] = str(ACTIVE_MANIFEST_PATH)\n",
    "\n",
    "# 2. Establish required output directory hierarchy in /kaggle/working/\n",
    "KAGGLE_OUTPUT_ROOT = Path(\"/kaggle/working/DeepShield_V5\")\n",
    "KAGGLE_CHECKPOINTS = KAGGLE_OUTPUT_ROOT / \"checkpoints\"\n",
    "KAGGLE_TRAINING_LOGS = KAGGLE_OUTPUT_ROOT / \"training\"\n",
    "KAGGLE_EVALUATION = KAGGLE_OUTPUT_ROOT / \"evaluation\"\n",
    "\n",
    "for folder in [KAGGLE_OUTPUT_ROOT, KAGGLE_CHECKPOINTS, KAGGLE_TRAINING_LOGS, KAGGLE_EVALUATION]:\n",
    "    folder.mkdir(parents=True, exist_ok=True)\n",
    "    print(f\"[+] Initialized directory: {folder}\")\n",
    "\n",
    "# 3. Verify working script presence\n",
    "working_script = Path(\"/kaggle/working/train_deepshield_v5.py\")\n",
    "if working_script.is_file():\n",
    "    print(f\"[+] Verified training script at: {working_script}\")\n",
    "\n",
    "print(\"=\" * 60)\n",
    "print(\"Configured Artifact Destinations:\")\n",
    "print(f\"  - Final V5 Model:      {KAGGLE_OUTPUT_ROOT / 'deepshield_v5_best.keras'}\")\n",
    "print(f\"  - Stage 1 Checkpoint:  {KAGGLE_CHECKPOINTS / 'deepshield_v5_stage1_best.keras'}\")\n",
    "print(f\"  - Stage 2 Checkpoint:  {KAGGLE_CHECKPOINTS / 'deepshield_v5_stage2_best.keras'}\")\n",
    "print(f\"  - Training Histories:  {KAGGLE_TRAINING_LOGS / 'history_stage*.csv'}\")\n",
    "print(f\"  - Evaluation Results:  {KAGGLE_EVALUATION / 'test_evaluation_results.json'}\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "# 4. Pre-Training Path Resolution Audit (Explicit 3-Split Verification)\n",
    "resolution_pass = False\n",
    "if working_script.is_file() and 'ACTIVE_DATASET_ROOT' in globals() and 'ACTIVE_MANIFEST_PATH' in globals():\n",
    "    sys.path.insert(0, \"/kaggle/working\")\n",
    "    try:\n",
    "        import importlib\n",
    "        import train_deepshield_v5 as v5_mod\n",
    "        importlib.reload(v5_mod)\n",
    "        m_df = v5_mod.load_and_verify_manifest(ACTIVE_MANIFEST_PATH)\n",
    "        tr_p, _ = v5_mod.resolve_split_filepaths(m_df, 'train', ACTIVE_DATASET_ROOT)\n",
    "        va_p, _ = v5_mod.resolve_split_filepaths(m_df, 'validation', ACTIVE_DATASET_ROOT)\n",
    "        te_p, _ = v5_mod.resolve_split_filepaths(m_df, 'test', ACTIVE_DATASET_ROOT)\n",
    "\n",
    "        tr_missing = 17208 - len(tr_p)\n",
    "        va_missing = 3688 - len(va_p)\n",
    "        te_missing = 3693 - len(te_p)\n",
    "        total_missing = tr_missing + va_missing + te_missing\n",
    "\n",
    "        print(\"\\n\" + \"=\" * 50)\n",
    "        print(\"PATH RESOLUTION AUDIT (PRE-TRAINING)\")\n",
    "        print(\"=\" * 50)\n",
    "        print(f\"Train:      {len(tr_p):,} / 17,208\")\n",
    "        print(f\"Validation: {len(va_p):,} / 3,688\")\n",
    "        print(f\"Test:       {len(te_p):,} / 3,693\")\n",
    "        print(f\"Missing:    {total_missing} (Train: {tr_missing}, Val: {va_missing}, Test: {te_missing})\")\n",
    "        print(\"=\" * 50)\n",
    "\n",
    "        if len(tr_p) == 17208 and len(va_p) == 3688 and len(te_p) == 3693 and total_missing == 0:\n",
    "            resolution_pass = True\n",
    "            print(\"[+] ALL THREE SPLITS VERIFIED: Missing = 0\")\n",
    "        else:\n",
    "            print(\"[!] RESOLUTION MISMATCH DETECTED\")\n",
    "    except Exception as e:\n",
    "        print(f\"[!] Path resolution dry-run check encountered error: {e}\")\n",
    "print(\"=\" * 60)\n",
    "print(\"KAGGLE ENVIRONMENT SETUP: READY\")\n",
    "print(\"=\" * 60)\n"
]

# 5. Cell 10 (index 10) Final Readiness Report
cell10 = nb["cells"][10]
cell10["source"] = [
    "# CELL 10 -- FINAL READINESS REPORT\n",
    "print(\"=\" * 60)\n",
    "print(\"DEEPSHIELD V5: KAGGLE FINAL READINESS REPORT\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "checks = [\n",
    "    (\"TensorFlow & Packages\", globals().get(\"tf_ok\", False)),\n",
    "    (\"GPU Accelerator\", globals().get(\"gpu_available\", False)),\n",
    "    (\"Dataset (24,589 images)\", globals().get(\"dataset_pass\", False)),\n",
    "    (\"Manifest (24,589 records)\", globals().get(\"manifest_pass\", False)),\n",
    "    (\"Split & Leakage Integrity\", globals().get(\"split_pass\", False)),\n",
    "    (\"Training Script Valid\", globals().get(\"script_pass\", False)),\n",
    "    (\"Path Resolution (3 splits)\", globals().get(\"resolution_pass\", False)),\n",
    "    (\"Output Hierarchy Ready\", Path(\"/kaggle/working/DeepShield_V5\").is_dir())\n",
    "]\n",
    "\n",
    "all_passed = True\n",
    "for name, passed in checks:\n",
    "    status_str = \"READY / PASS\" if passed else \"FAILED / INCOMPLETE\"\n",
    "    print(f\"  {name:<28}: {status_str}\")\n",
    "    if not passed:\n",
    "        all_passed = False\n",
    "\n",
    "print(\"=\" * 60)\n",
    "if all_passed:\n",
    "    print(\"V5 TRAINING ON KAGGLE: READY TO START\")\n",
    "else:\n",
    "    print(\"V5 TRAINING ON KAGGLE: CHECK REQUIRED (Resolve failed items above)\")\n",
    "print(\"=\" * 60)\n",
    "print(\"\\nTRAINING STATUS: NOT STARTED\\n\")\n"
]

# 6. Ensure Cell 11 (index 11) is strictly unexecuted
cell11 = nb["cells"][11]
cell11["execution_count"] = None
cell11["outputs"] = []

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[+] DeepShield_V5_Kaggle.ipynb updated successfully.")
