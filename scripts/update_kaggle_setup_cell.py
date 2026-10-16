import json
from pathlib import Path

nb_path = Path("training/DeepShield_V5_Kaggle.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The setup cell source code
setup_cell_source = [
    "# SETUP CELL -- UPLOAD & CONFIGURE TRAINING SCRIPT\n",
    "import os\n",
    "import shutil\n",
    "from pathlib import Path\n",
    "\n",
    "target_script = Path(\"/kaggle/working/train_deepshield_v5.py\")\n",
    "\n",
    "print(\"=\" * 60)\n",
    "print(\"DEEPSHIELD V5: TRAINING SCRIPT SETUP\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "# 1. If not directly at /kaggle/working/train_deepshield_v5.py, check /kaggle/working subdirectories\n",
    "if not target_script.is_file():\n",
    "    for root, dirs, files in os.walk(\"/kaggle/working\"):\n",
    "        if \"train_deepshield_v5.py\" in files:\n",
    "            found_path = Path(root) / \"train_deepshield_v5.py\"\n",
    "            if found_path.resolve() != target_script.resolve():\n",
    "                shutil.copy2(found_path, target_script)\n",
    "                print(f\"[+] Found and copied script from: {found_path}\")\n",
    "            break\n",
    "\n",
    "# 2. Search /kaggle/input if attached as a dataset or utility script\n",
    "if not target_script.is_file() and Path(\"/kaggle/input\").exists():\n",
    "    for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
    "        if \"train_deepshield_v5.py\" in files:\n",
    "            found_path = Path(root) / \"train_deepshield_v5.py\"\n",
    "            shutil.copy2(found_path, target_script)\n",
    "            print(f\"[+] Found and copied script from Kaggle input: {found_path}\")\n",
    "            break\n",
    "\n",
    "# 3. Verification check as specified\n",
    "print(\"=\" * 60)\n",
    "if Path(\"/kaggle/working/train_deepshield_v5.py\").is_file():\n",
    "    print(f\"File location: {target_script}\")\n",
    "    print(f\"File size:     {target_script.stat().st_size:,} bytes\")\n",
    "    print(\"=\" * 60)\n",
    "    print(\"TRAINING SCRIPT READY\")\n",
    "else:\n",
    "    print(\"INSTRUCTIONS TO UPLOAD SCRIPT TO KAGGLE:\")\n",
    "    print(\"1. In Kaggle Notebook, click on the right-hand panel (or 'Files' tab).\")\n",
    "    print(\"2. Upload the local file:\")\n",
    "    print(\"   C:\\\\Users\\\\ADMIN\\\\Desktop\\\\DeepShield_V5_Final\\\\training\\\\train_deepshield_v5.py\")\n",
    "    print(\"   into /kaggle/working/ (or drag and drop it into the files panel).\")\n",
    "    print(\"3. Alternatively, click '+ Add Input' -> 'Upload Dataset', upload train_deepshield_v5.py, and attach it.\")\n",
    "    print(\"4. After uploading, re-run this setup cell.\")\n",
    "    print(\"=\" * 60)\n",
    "    print(\"TRAINING SCRIPT NOT FOUND — upload train_deepshield_v5.py to Kaggle first.\")\n",
    "print(\"=\" * 60)\n"
]

# Find the setup cell (currently at index 7)
setup_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "SETUP CELL" in src:
        setup_idx = i
        break

assert setup_idx is not None, "Error: Setup cell not found!"
nb["cells"][setup_idx]["source"] = setup_cell_source

# Verify Cell 8 is immediately after setup_idx
cell_8_src = "".join(nb["cells"][setup_idx + 1].get("source", []))
assert "CELL 8 -- VERIFY TRAIN_DEEPSHIELD_V5.PY" in cell_8_src, "Error: Cell 8 must follow setup cell!"

# Verify Cell 11 is last and unchanged
cell_11_src = "".join(nb["cells"][-1].get("source", []))
assert "!python /kaggle/working/train_deepshield_v5.py" in cell_11_src, "Error: Cell 11 must be training command!"

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully updated setup cell in DeepShield_V5_Kaggle.ipynb")
