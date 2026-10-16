import json

notebook_path = "training/DeepShield_V5_Colab.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["cells"][4]["source"] = [
    "# CELL 5 -- UPLOAD TRAINING SCRIPT\n",
    "from pathlib import Path\n",
    "import shutil\n",
    "from google.colab import files\n",
    "\n",
    "# Create /content/training/ if necessary\n",
    "target_dir = Path(\"/content/training\")\n",
    "target_dir.mkdir(parents=True, exist_ok=True)\n",
    "target_script = target_dir / \"train_deepshield_v5.py\"\n",
    "\n",
    "if not target_script.exists():\n",
    "    print(\"[*] Please select and upload 'train_deepshield_v5.py' from your computer:\")\n",
    "    print(\"    Local source: C:\\\\Users\\\\ADMIN\\\\Desktop\\\\DeepShield_V5_Final\\\\training\\\\train_deepshield_v5.py\\n\")\n",
    "    uploaded = files.upload()\n",
    "    for fname in uploaded.keys():\n",
    "        if fname.endswith(\"train_deepshield_v5.py\") or fname.endswith(\".py\"):\n",
    "            shutil.move(fname, str(target_script))\n",
    "            break\n",
    "\n",
    "script_exists = target_script.is_file()\n",
    "file_size = target_script.stat().st_size if script_exists else 0\n",
    "\n",
    "print(\"=\" * 50)\n",
    "print(\"TRAINING SCRIPT CHECK\")\n",
    "print(\"=\" * 50)\n",
    "print(\"Training script:\")\n",
    "print(f\" {target_script}\")\n",
    "if script_exists:\n",
    "    print(f\"Size: {file_size:,} bytes\")\n",
    "print(\"\\nStatus:\")\n",
    "print(f\" {'PASS' if script_exists else 'FAIL'}\")\n",
    "\n",
    "if script_exists:\n",
    "    print(\"\\nTRAINING SCRIPT: READY\")\n",
    "else:\n",
    "    print(\"\\nTRAINING SCRIPT: FAILED\")\n",
    "print(\"=\" * 50)\n"
]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print("SUCCESS: Cell 5 updated.")
