import json
import base64
from pathlib import Path

nb_path = Path("training/DeepShield_V5_Kaggle.ipynb")
script_file = Path("training/train_deepshield_v5.py")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Read train_deepshield_v5.py and encode to base64
with open(script_file, "rb") as f:
    script_bytes = f.read()

script_b64_str = base64.b64encode(script_bytes).decode("ascii")

setup_cell_source = [
    "# SETUP CELL -- DEPLOY TRAINING SCRIPT TO /kaggle/working/\n",
    "# Copies train_deepshield_v5.py from input or deploys the verified project script directly\n",
    "import os\n",
    "import shutil\n",
    "import base64\n",
    "from pathlib import Path\n",
    "\n",
    "print(\"=\" * 60)\n",
    "print(\"DEEPSHIELD V5: DEPLOYING TRAINING SCRIPT TO /kaggle/working/\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "target_dir = Path(\"/kaggle/working\")\n",
    "target_dir.mkdir(parents=True, exist_ok=True)\n",
    "target_script = target_dir / \"train_deepshield_v5.py\"\n",
    "\n",
    "# Method 1: Check if attached as an input dataset under /kaggle/input\n",
    "found_in_input = False\n",
    "if Path(\"/kaggle/input\").exists():\n",
    "    for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
    "        if \"train_deepshield_v5.py\" in files:\n",
    "            src_path = Path(root) / \"train_deepshield_v5.py\"\n",
    "            shutil.copy2(src_path, target_script)\n",
    "            print(f\"[+] Found and copied training script from Kaggle input: {src_path}\")\n",
    "            found_in_input = True\n",
    "            break\n",
    "\n",
    "# Method 2: Deploy verified project training script directly\n",
    "if not found_in_input:\n",
    "    print(\"[*] Deploying verified training script into /kaggle/working/train_deepshield_v5.py...\")\n",
    f"    EMBEDDED_B64 = b\"{script_b64_str}\"\n",
    "    with open(target_script, \"wb\") as f:\n",
    "        f.write(base64.b64decode(EMBEDDED_B64))\n",
    "    print(\"[+] Training script successfully written.\")\n",
    "\n",
    "if target_script.is_file():\n",
    "    file_size = target_script.stat().st_size\n",
    "    print(f\"[+] Script Location: {target_script}\")\n",
    "    print(f\"[+] File Size:       {file_size:,} bytes\")\n",
    "    print(\"=\" * 60)\n",
    "    print(\"TRAINING SCRIPT SETUP: READY\")\n",
    "else:\n",
    "    print(\"=\" * 60)\n",
    "    print(\"[!] Failed to deploy train_deepshield_v5.py\")\n",
    "print(\"=\" * 60)\n"
]

setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": setup_cell_source
}

# Find the cell labeled CELL 8
cell_8_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "CELL 8 -- VERIFY TRAIN_DEEPSHIELD_V5.PY" in src:
        cell_8_idx = i
        break

assert cell_8_idx is not None, "Error: Cell 8 not found in notebook!"

# Insert setup cell immediately before Cell 8
nb["cells"].insert(cell_8_idx, setup_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully inserted setup cell before Cell 8. Total cells now: {len(nb['cells'])}")
