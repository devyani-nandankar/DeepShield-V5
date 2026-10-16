import shutil
from pathlib import Path

target_dir = Path("kaggle_training_script_dataset")
target_dir.mkdir(parents=True, exist_ok=True)

src_script = Path("training/train_deepshield_v5.py")
dst_script = target_dir / "train_deepshield_v5.py"
shutil.copy2(src_script, dst_script)

meta_path = target_dir / "dataset-metadata.json"
meta_content = """{
  "title": "deepshield-v5-code",
  "id": "devyaninandankar/deepshield-v5-code",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}"""

with open(meta_path, "w", encoding="utf-8") as f:
    f.write(meta_content)

readme_path = target_dir / "README.md"
readme_content = """# DeepShield V5 Training Script Dataset

This dataset contains the latest verified `train_deepshield_v5.py` training script for DeepShield V5 on Kaggle GPU.

### Files:
- `train_deepshield_v5.py`: Two-stage transfer learning pipeline (EfficientNetB2) with multi-tier path resolver.
- `dataset-metadata.json`: Kaggle API dataset metadata descriptor.

### Uploading to Kaggle:
- **Option A (Web UI)**: Go to Kaggle -> Datasets -> 'New Dataset' -> Upload `train_deepshield_v5.py` -> Name it `deepshield-v5-code`.
- **Option B (Kaggle CLI)**: Run `kaggle datasets create -p kaggle_training_script_dataset` (or `version -m "Update path resolver"`).
"""

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"[+] Kaggle training-script dataset prepared at: {target_dir.resolve()}")
print(f"    - Script: {dst_script} ({dst_script.stat().st_size:,} bytes)")
print(f"    - Metadata: {meta_path}")
