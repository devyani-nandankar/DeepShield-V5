import json
from pathlib import Path

nb_path = Path("training/DeepShield_V5_Colab.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_11_source = [
    "# CELL 11 -- SPEED OPTIMIZATION: LOCAL DATASET COPY & VERIFICATION\n",
    "import os\n",
    "import shutil\n",
    "from pathlib import Path\n",
    "import pandas as pd\n",
    "\n",
    "# Auto-mount Google Drive if not already mounted in active runtime\n",
    "if not Path(\"/content/drive/MyDrive\").exists():\n",
    "    print(\"[*] Google Drive is not mounted. Mounting now...\")\n",
    "    try:\n",
    "        from google.colab import drive\n",
    "        drive.mount(\"/content/drive\")\n",
    "        print(\"[+] Google Drive successfully mounted.\")\n",
    "    except Exception as exc:\n",
    "        print(f\"[!] Drive auto-mount note: {exc}\")\n",
    "\n",
    "# Paths\n",
    "DRIVE_DATASET_ROOT = Path(\"/content/drive/MyDrive/DeepShield_Datasets/Face_Cropped_FAST10\")\n",
    "DRIVE_MANIFEST_PATH = Path(\"/content/drive/MyDrive/DeepShield_Datasets/Metadata/face_crop_FAST10_manifest_split.csv\")\n",
    "LOCAL_DATASET_ROOT = Path(\"/content/DeepShield_FAST10\")\n",
    "LOCAL_MANIFEST_PATH = Path(\"/content/DeepShield_FAST10_manifest_split.csv\")\n",
    "\n",
    "EXPECTED = {\n",
    "    \"train\": 17208,\n",
    "    \"validation\": 3688,\n",
    "    \"test\": 3693,\n",
    "    \"total\": 24589\n",
    "}\n",
    "\n",
    "print(\"=\" * 60)\n",
    "print(\"DEEPSHIELD V5: LOCAL DATASET SPEED OPTIMIZATION\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "# Helper to count image files in directory\n",
    "def count_images(dir_path: Path) -> int:\n",
    "    if not dir_path.is_dir():\n",
    "        return 0\n",
    "    valid_exts = {\".jpg\", \".jpeg\", \".png\"}\n",
    "    return sum(1 for f in os.scandir(dir_path) if f.is_file() and Path(f.name).suffix.lower() in valid_exts)\n",
    "\n",
    "# Step 1: Check if already copied with exact counts\n",
    "train_cnt = count_images(LOCAL_DATASET_ROOT / \"train\")\n",
    "val_cnt = count_images(LOCAL_DATASET_ROOT / \"validation\")\n",
    "test_cnt = count_images(LOCAL_DATASET_ROOT / \"test\")\n",
    "total_cnt = train_cnt + val_cnt + test_cnt\n",
    "manifest_ready = LOCAL_MANIFEST_PATH.is_file()\n",
    "\n",
    "if (total_cnt == EXPECTED[\"total\"] and train_cnt == EXPECTED[\"train\"] and \n",
    "    val_cnt == EXPECTED[\"validation\"] and test_cnt == EXPECTED[\"test\"] and manifest_ready):\n",
    "    print(\"[+] Local dataset already exists with verified counts. Reusing local copy:\")\n",
    "    print(f\"    Train: {train_cnt:,} | Validation: {val_cnt:,} | Test: {test_cnt:,} | Total: {total_cnt:,}\")\n",
    "else:\n",
    "    # Verify Drive source exists first\n",
    "    if not DRIVE_DATASET_ROOT.is_dir():\n",
    "        raise FileNotFoundError(f\"Source dataset not found on Google Drive: {DRIVE_DATASET_ROOT}\")\n",
    "    if not DRIVE_MANIFEST_PATH.is_file():\n",
    "        raise FileNotFoundError(f\"Source manifest not found on Google Drive: {DRIVE_MANIFEST_PATH}\")\n",
    "\n",
    "    # Copy manifest\n",
    "    print(f\"[*] Copying manifest to local storage: {LOCAL_MANIFEST_PATH}...\")\n",
    "    shutil.copy2(DRIVE_MANIFEST_PATH, LOCAL_MANIFEST_PATH)\n",
    "    print(\"[+] Manifest copied successfully.\")\n",
    "\n",
    "    # Copy dataset using rsync (fastest, preserves timestamps, resumable) with fallback to cp -r\n",
    "    LOCAL_DATASET_ROOT.mkdir(parents=True, exist_ok=True)\n",
    "    print(f\"[*] Copying FAST10 dataset to high-speed NVMe storage: {LOCAL_DATASET_ROOT}...\")\n",
    "    rsync_cmd = f\"rsync -a '{DRIVE_DATASET_ROOT}/' '{LOCAL_DATASET_ROOT}/'\"\n",
    "    ret = os.system(rsync_cmd)\n",
    "    if ret != 0:\n",
    "        print(\"[*] rsync returned non-zero, using cp -r fallback...\")\n",
    "        os.system(f\"cp -r '{DRIVE_DATASET_ROOT}/.' '{LOCAL_DATASET_ROOT}/'\")\n",
    "    print(\"[+] Dataset copy complete.\")\n",
    "\n",
    "# Step 2: Verification of local files and manifest\n",
    "print(\"\\n\" + \"=\" * 60)\n",
    "print(\"VERIFYING LOCAL REPLICATED DATASET & MANIFEST\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "local_train = count_images(LOCAL_DATASET_ROOT / \"train\")\n",
    "local_val = count_images(LOCAL_DATASET_ROOT / \"validation\")\n",
    "local_test = count_images(LOCAL_DATASET_ROOT / \"test\")\n",
    "local_total = local_train + local_val + local_test\n",
    "\n",
    "df_manifest = pd.read_csv(LOCAL_MANIFEST_PATH)\n",
    "manifest_total = len(df_manifest)\n",
    "\n",
    "print(f\"Local Dataset Path:  {LOCAL_DATASET_ROOT}\")\n",
    "print(f\"Local Manifest Path: {LOCAL_MANIFEST_PATH}\")\n",
    "print(f\"Train Images:        {local_train:,}  (Expected: {EXPECTED['train']:,})\")\n",
    "print(f\"Validation Images:   {local_val:,}  (Expected: {EXPECTED['validation']:,})\")\n",
    "print(f\"Test Images:         {local_test:,}  (Expected: {EXPECTED['test']:,})\")\n",
    "print(f\"Total Images:        {local_total:,}  (Expected: {EXPECTED['total']:,})\")\n",
    "print(f\"Manifest Records:    {manifest_total:,}  (Expected: {EXPECTED['total']:,})\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "assert local_train == EXPECTED[\"train\"], f\"Mismatch train count: {local_train}\"\n",
    "assert local_val == EXPECTED[\"validation\"], f\"Mismatch val count: {local_val}\"\n",
    "assert local_test == EXPECTED[\"test\"], f\"Mismatch test count: {local_test}\"\n",
    "assert local_total == EXPECTED[\"total\"], f\"Mismatch total count: {local_total}\"\n",
    "assert manifest_total == EXPECTED[\"total\"], f\"Mismatch manifest count: {manifest_total}\"\n",
    "print(\"\\n\" + \"=\" * 50)\n",
    "print(\"LOCAL DATASET: READY\")\n",
    "print(\"LOCAL MANIFEST: READY\")\n",
    "print(f\"TOTAL: {local_total:,}\")\n",
    "print(\"TRAINING: NOT STARTED\")\n",
    "print(\"=\" * 50)\n"
]

cell_12_source = [
    "# CELL 12 -- START DEEPSHIELD V5 TRAINING\n",
    "# Execute the optimized two-stage training script in Google Colab\n",
    "!python /content/training/train_deepshield_v5.py\n"
]

cell_11 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell_11_source
}

cell_12 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell_12_source
}

current_cells = nb["cells"][:10]
current_cells.append(cell_11)
current_cells.append(cell_12)
nb["cells"] = current_cells

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Updated {nb_path} successfully. Total cells now: {len(nb['cells'])}")
