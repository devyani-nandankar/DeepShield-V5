import json
from pathlib import Path

nb_path = Path("training/DeepShield_V5_Kaggle.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 4 default paths
cell4 = nb["cells"][3]
new_src = []
for line in cell4["source"]:
    if line.startswith("KAGGLE_DATASET_ROOT ="):
        new_src.append('KAGGLE_DATASET_ROOT = "/kaggle/input/datasets/devyaninandankar/deepshield-face-dataset/Kaggle_Transfer/Face_Cropped_FAST10"\n')
    elif line.startswith("KAGGLE_MANIFEST_PATH ="):
        new_src.append('KAGGLE_MANIFEST_PATH = "/kaggle/input/datasets/devyaninandankar/deepshield-face-dataset/Kaggle_Transfer/face_crop_FAST10_manifest_split.csv"\n')
    else:
        new_src.append(line)
cell4["source"] = new_src

# 2. Update Cell 9 (Index 9) to run the dry-run path resolution test
cell9 = nb["cells"][9]
cell9_src = [
    "# CELL 9 -- KAGGLE PATH & ENVIRONMENT CONFIGURATION\n",
    "import os\n",
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
    "# 3. Copy train_deepshield_v5.py into /kaggle/working/ if not already present\n",
    "working_script = Path(\"/kaggle/working/train_deepshield_v5.py\")\n",
    "if 'script_path' in globals() and script_path and script_path != working_script:\n",
    "    shutil.copy2(script_path, working_script)\n",
    "    print(f\"[+] Replicated script to: {working_script}\")\n",
    "elif working_script.is_file():\n",
    "    print(f\"[+] Script verified at: {working_script}\")\n",
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
    "# 4. Path Resolution Dry-Run Verification\n",
    "resolution_pass = False\n",
    "if working_script.is_file() and 'ACTIVE_DATASET_ROOT' in globals() and 'ACTIVE_MANIFEST_PATH' in globals():\n",
    "    print(\"\\n[*] Running Pre-Training Path Resolution Dry-Run...\")\n",
    "    import sys\n",
    "    sys.path.insert(0, \"/kaggle/working\")\n",
    "    try:\n",
    "        import importlib\n",
    "        import train_deepshield_v5 as v5_mod\n",
    "        importlib.reload(v5_mod)\n",
    "        m_df = v5_mod.load_and_verify_manifest(ACTIVE_MANIFEST_PATH)\n",
    "        tr_p, _ = v5_mod.resolve_split_filepaths(m_df, 'train', ACTIVE_DATASET_ROOT)\n",
    "        va_p, _ = v5_mod.resolve_split_filepaths(m_df, 'validation', ACTIVE_DATASET_ROOT)\n",
    "        te_p, _ = v5_mod.resolve_split_filepaths(m_df, 'test', ACTIVE_DATASET_ROOT)\n",
    "        if len(tr_p) == 17208 and len(va_p) == 3688 and len(te_p) == 3693:\n",
    "            resolution_pass = True\n",
    "            print(\"\\n[+] PRE-TRAINING RESOLUTION AUDIT: 100% PASS (24,589 / 24,589 images verified)\")\n",
    "        else:\n",
    "            print(\"\\n[!] PRE-TRAINING RESOLUTION AUDIT: COUNT MISMATCH\")\n",
    "    except Exception as e:\n",
    "        print(f\"[!] Path resolution dry-run check encountered error: {e}\")\n",
    "print(\"=\" * 60)\n",
    "print(\"KAGGLE ENVIRONMENT SETUP: READY\")\n",
    "print(\"=\" * 60)\n"
]
cell9["source"] = cell9_src

# 3. Update Cell 10 (Index 10) Final Readiness Report to include resolution_pass
cell10 = nb["cells"][10]
cell10_src = [
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
cell10["source"] = cell10_src

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[+] Successfully updated DeepShield_V5_Kaggle.ipynb")
