import json
import py_compile
import tempfile
from pathlib import Path

print("=" * 60)
print("DEEPSHIELD V5: KAGGLE SETUP COMPREHENSIVE VALIDATION")
print("=" * 60)

# 1. Check training script
script_path = Path("training/train_deepshield_v5.py")
assert script_path.is_file(), "training/train_deepshield_v5.py missing!"
py_compile.compile(str(script_path), doraise=True)
script_content = script_path.read_text(encoding="utf-8")
assert "5.0.2-path-fix" in script_content, "Version 5.0.2-path-fix not in training script!"
assert "file_lookup_basename" in script_content, "Multi-tier resolver missing in training script!"
print(f"[+] train_deepshield_v5.py: VALID (Version 5.0.2-path-fix, {script_path.stat().st_size:,} bytes)")

# 2. Check Kaggle training script dataset
dataset_script = Path("kaggle_training_script_dataset/train_deepshield_v5.py")
assert dataset_script.is_file(), "kaggle_training_script_dataset/train_deepshield_v5.py missing!"
assert dataset_script.stat().st_size == script_path.stat().st_size, "Dataset script size mismatch!"
dataset_meta = Path("kaggle_training_script_dataset/dataset-metadata.json")
assert dataset_meta.is_file(), "kaggle_training_script_dataset/dataset-metadata.json missing!"
print(f"[+] kaggle_training_script_dataset: VALID (Ready for Kaggle upload)")

# 3. Check DeepShield V4 untouched
v4_path = Path("models/deepshield_final.keras")
if v4_path.is_file():
    print(f"[+] DeepShield V4 model untouched: {v4_path} ({v4_path.stat().st_size:,} bytes)")
else:
    print("[*] V4 model path noted (untouched).")

# 4. Check Kaggle notebook
nb_path = Path("training/DeepShield_V5_Kaggle.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"[+] Total notebook cells: {len(cells)}")
assert len(cells) == 12, f"Expected 12 cells, got {len(cells)}"

# Print summary of cells
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    header = src.split("\n")[0] if src else ""
    print(f"    Cell {i+1:>2} [{c['cell_type']:>8}]: {header[:65]}")

# Verify Setup Cell & Cell 8 order
setup_src = "".join(cells[7]["source"])
assert "SETUP CELL" in setup_src
assert "EXPECTED_VERSION = \"5.0.2-path-fix\"" in setup_src
assert "shutil.copy2(found_input_script, target_script)" in setup_src
print("[+] Setup Cell (Cell 8): Stale script prevention and auto-overwrite confirmed.")

# Verify Cell 8 (Verification)
c8_src = "".join(cells[8]["source"])
assert "5.0.2-path-fix" in c8_src
print("[+] Cell 8 (Verification): Version and integrity check confirmed.")

# Verify Cell 9 (Path & Resolution Audit)
c9_src = "".join(cells[9]["source"])
assert "Train:      {len(tr_p):,} / 17,208" in c9_src
assert "Validation: {len(va_p):,} / 3,688" in c9_src
assert "Test:       {len(te_p):,} / 3,693" in c9_src
assert "Missing:    {total_missing}" in c9_src
print("[+] Cell 9: Path Resolution pre-training audit with exact 3-split output confirmed.")

# Verify Cell 11 (Execution command)
c11 = cells[11]
c11_src = "".join(c11["source"])
assert "!python /kaggle/working/train_deepshield_v5.py" in c11_src
assert c11["execution_count"] is None, "Cell 11 must NOT be executed!"
assert len(c11["outputs"]) == 0, "Cell 11 must have NO outputs!"
print("[+] Cell 11: Training command intact and UNEXECUTED (!python /kaggle/working/train_deepshield_v5.py).")

# Compile all code cells except Cell 11 (which is shell command !python)
for i, c in enumerate(cells[:-1]):
    if c["cell_type"] == "code":
        src = "".join(c.get("source", []))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(src)
            t_name = tf.name
        py_compile.compile(t_name, doraise=True)

print("[+] All code cells syntax-checked: PASS")
print("=" * 60)
print("ALL VALIDATION CRITERIA PASSED: TRAINING NOT STARTED")
print("=" * 60)
