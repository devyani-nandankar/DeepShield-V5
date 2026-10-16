import json

notebook_path = "training/DeepShield_V5_Colab.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["cells"][3]["source"] = [
    "# CELL 4 -- CREATE TRAINING DIRECTORY\n",
    "from pathlib import Path\n",
    "\n",
    "target_dir = Path(\"/content/training\")\n",
    "target_dir.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "dir_exists = target_dir.is_dir()\n",
    "\n",
    "print(\"=\" * 50)\n",
    "print(\"COLAB TRAINING DIRECTORY CHECK\")\n",
    "print(\"=\" * 50)\n",
    "print(\"Directory:\")\n",
    "print(f\" {target_dir}/\")\n",
    "print(\"\\nStatus:\")\n",
    "print(f\" {'PASS' if dir_exists else 'FAIL'}\")\n",
    "\n",
    "if dir_exists:\n",
    "    print(\"\\nCOLAB TRAINING DIRECTORY: READY\")\n",
    "else:\n",
    "    print(\"\\nCOLAB TRAINING DIRECTORY: FAILED\")\n",
    "print(\"=\" * 50)\n"
]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print("SUCCESS: Cell 4 updated.")
