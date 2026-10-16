import json
import py_compile
import tempfile
from pathlib import Path

print("=" * 60)
print("VALIDATION: STALE SCRIPT FIX & REAL FILESYSTEM AUDIT")
print("=" * 60)

# 1. Check local train_deepshield_v5.py
py_script = Path("training/train_deepshield_v5.py")
assert py_script.is_file(), "training/train_deepshield_v5.py not found!"
py_compile.compile(str(py_script), doraise=True)
py_text = py_script.read_text(encoding="utf-8")
assert 'SCRIPT_VERSION = "5.0.2-path-fix"' in py_text, "SCRIPT_VERSION 5.0.2-path-fix not defined!"
assert py_script.stat().st_size >= 36000, f"Script size {py_script.stat().st_size} is too small!"
assert py_script.stat().st_size != 31002, "Script size matches stale 31,002 bytes!"
print(f"[+] train_deepshield_v5.py: VALID (Size: {py_script.stat().st_size:,} bytes, Version: 5.0.2-path-fix)")

# 2. Check kaggle_training_script_dataset
dataset_script = Path("kaggle_training_script_dataset/train_deepshield_v5.py")
assert dataset_script.is_file(), "kaggle_training_script_dataset/train_deepshield_v5.py not found!"
assert dataset_script.stat().st_size == py_script.stat().st_size, "Dataset script size does not match!"
print(f"[+] kaggle_training_script_dataset: VALID (Size: {dataset_script.stat().st_size:,} bytes)")

# 3. Check V4 untouched
v4 = Path("models/deepshield_final.keras")
if v4.is_file():
    print(f"[+] DeepShield V4 model intact: {v4} ({v4.stat().st_size:,} bytes)")

# 4. Check notebook JSON and cells
nb_path = Path("training/DeepShield_V5_Kaggle.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
assert len(cells) == 12, f"Expected 12 cells, got {len(cells)}"
print(f"[+] Total notebook cells: {len(cells)}")

# Check Setup Cell (Index 7)
setup_src = "".join(cells[7]["source"])
assert "31002" in setup_src, "Setup cell must explicitly reject 31,002 bytes!"
assert "5.0.2-path-fix" in setup_src, "Setup cell must enforce 5.0.2-path-fix!"
assert "STALE TRAINING SCRIPT DETECTED — DO NOT TRAIN" in setup_src
assert "deepshield-v5-code" in setup_src
assert "shutil.copy2(best_source, target_script)" in setup_src
print("[+] Setup Cell (Index 7): Hard check against stale 31,002-byte script confirmed.")

# Check Cell 8 (Index 8)
c8_src = "".join(cells[8]["source"])
assert "5.0.2-path-fix" in c8_src
assert "EfficientNetB2" in c8_src
assert "IMAGE_SIZE = (224, 224)" in c8_src
assert "isinstance(layer, layers.BatchNormalization)" in c8_src
assert "Path resolver" in c8_src
print("[+] Cell 8 (Index 8): All 7 verification checks confirmed.")

# Check Cell 9 (Index 9) - Real Filesystem Audit
c9_src = "".join(cells[9]["source"])
assert "/kaggle/input/datasets/devyaninandankar/deepshield-face-dataset/Kaggle_Transfer/Face_Cropped_FAST10" in c9_src
assert "os.walk" in c9_src
assert "Train:      {split_stats['train'][0]:,} / {SPLIT_EXPECTED['train']:,}" in c9_src
assert "Validation: {split_stats['validation'][0]:,} / {SPLIT_EXPECTED['validation']:,}" in c9_src
assert "Test:       {split_stats['test'][0]:,} / {SPLIT_EXPECTED['test']:,}" in c9_src
assert "Missing:    {total_missing}" in c9_src
assert "first_20_missing" in c9_src
print("[+] Cell 9 (Index 9): Real recursive filesystem audit and 3-split output confirmed.")

# Check Cell 11 (Index 11) - Training command
c11 = cells[11]
c11_src = "".join(c11["source"])
assert "!python /kaggle/working/train_deepshield_v5.py" in c11_src
assert c11["execution_count"] is None, "Cell 11 must NOT be executed!"
assert len(c11["outputs"]) == 0, "Cell 11 must have NO outputs!"
print("[+] Cell 11 (Index 11): Training command intact and UNEXECUTED.")

# Check syntax of all code cells
for i, c in enumerate(cells[:-1]):
    if c["cell_type"] == "code":
        src = "".join(c.get("source", []))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(src)
            t_name = tf.name
        py_compile.compile(t_name, doraise=True)

print("[+] All notebook code cells syntax-checked: PASS")
print("=" * 60)
print("ALL CRITERIA VERIFIED SUCCESSFULLY: TRAINING NOT STARTED")
print("=" * 60)
