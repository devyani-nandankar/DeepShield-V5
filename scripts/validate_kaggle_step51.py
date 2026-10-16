import json
import py_compile
import tempfile
from pathlib import Path

nb_path = Path("training/DeepShield_V5_Kaggle.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total notebook cells: {len(cells)}")
assert len(cells) == 12, f"Expected 12 cells, got {len(cells)}"

# Print cell headers and inspect
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    header = src.split("\n")[0] if src else ""
    print(f"Index {i:>2} [{c['cell_type']:>8}]: {header}")

# Check setup cell is before Cell 8
setup_idx = None
cell_8_idx = None
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "SETUP CELL" in src:
        setup_idx = i
        assert 'Path("/kaggle/working/train_deepshield_v5.py").is_file()' in src
        assert "TRAINING SCRIPT NOT FOUND — upload train_deepshield_v5.py to Kaggle first." in src
        assert "TRAINING SCRIPT READY" in src
    if "CELL 8 -- VERIFY TRAIN_DEEPSHIELD_V5.PY" in src:
        cell_8_idx = i

assert setup_idx is not None, "Setup cell not found!"
assert cell_8_idx is not None, "Cell 8 not found!"
assert setup_idx == cell_8_idx - 1, f"Setup cell ({setup_idx}) must be immediately before Cell 8 ({cell_8_idx})!"
print("\n[+] Setup cell confirmed immediately BEFORE Cell 8.")

# Check Cell 11 training command unchanged
c11_src = "".join(cells[-1]["source"])
assert "!python /kaggle/working/train_deepshield_v5.py" in c11_src
assert cells[-1]["execution_count"] is None
assert len(cells[-1]["outputs"]) == 0
print("[+] Cell 11 training command confirmed unchanged (!python /kaggle/working/train_deepshield_v5.py).")

# Compile check for all code cells except last
for i, c in enumerate(cells[:-1]):
    if c["cell_type"] == "code":
        src = "".join(c.get("source", []))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(src)
            t_name = tf.name
        py_compile.compile(t_name, doraise=True)

print("[+] All code cells syntax-checked: PASS")
print("[+] Training status: NOT STARTED")
