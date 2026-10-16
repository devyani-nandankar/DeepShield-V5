import json

notebook_path = "training/DeepShield_V5_Colab.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["cells"][5]["source"] = [
    "# CELL 6 -- VERIFY DATASET\n",
    "from pathlib import Path\n",
    "\n",
    "dataset_root = Path(\"/content/drive/MyDrive/DeepShield_Datasets/Face_Cropped_FAST10\")\n",
    "valid_exts = {\".jpg\", \".jpeg\", \".png\", \".bmp\", \".webp\"}\n",
    "\n",
    "expected_counts = {\n",
    "    \"train\": 17208,\n",
    "    \"validation\": 3688,\n",
    "    \"test\": 3693\n",
    "}\n",
    "expected_total = sum(expected_counts.values())\n",
    "\n",
    "actual_counts = {}\n",
    "dataset_split_results = {}\n",
    "mismatches = []\n",
    "\n",
    "for split, exp_count in expected_counts.items():\n",
    "    split_dir = dataset_root / split\n",
    "    if split_dir.is_dir():\n",
    "        count = len([f for f in split_dir.rglob(\"*\") if f.is_file() and f.suffix.lower() in valid_exts])\n",
    "    else:\n",
    "        count = 0\n",
    "    actual_counts[split] = count\n",
    "    is_match = (count == exp_count)\n",
    "    dataset_split_results[split] = is_match\n",
    "    if not is_match:\n",
    "        mismatches.append(split)\n",
    "\n",
    "actual_total = sum(actual_counts.values())\n",
    "\n",
    "print(\"=\" * 50)\n",
    "print(\"DEEPSHIELD V5 DATASET CHECK\")\n",
    "print(\"=\" * 50)\n",
    "print(f\"Dataset:\\n {dataset_root}\\n\")\n",
    "print(f\"Train:\\n {actual_counts['train']:,} / {expected_counts['train']:,}\\n\")\n",
    "print(f\"Validation:\\n {actual_counts['validation']:,} / {expected_counts['validation']:,}\\n\")\n",
    "print(f\"Test:\\n {actual_counts['test']:,} / {expected_counts['test']:,}\\n\")\n",
    "print(f\"Total:\\n {actual_total:,} / {expected_total:,}\")\n",
    "print(\"=\" * 50)\n",
    "\n",
    "if len(mismatches) == 0 and dataset_root.is_dir():\n",
    "    print(\"\\nDATASET: READY\")\n",
    "else:\n",
    "    print(\"\\nDATASET: CHECK REQUIRED\")\n",
    "    if not dataset_root.is_dir():\n",
    "        print(f\"[!] Dataset root folder not found at: {dataset_root}\")\n",
    "    for m in mismatches:\n",
    "        print(f\"[!] Count mismatch in '{m}' split: Actual {actual_counts[m]:,} vs Expected {expected_counts[m]:,}\")\n",
    "print(\"=\" * 50)\n"
]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print("SUCCESS: Cell 6 updated.")
