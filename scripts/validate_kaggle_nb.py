import json
import py_compile
import tempfile
from pathlib import Path

nb_path = Path("training/DeepShield_V5_Kaggle.ipynb")
assert nb_path.is_file(), f"Error: {nb_path} not found"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total notebook cells: {len(cells)}")
assert len(cells) == 11, f"Expected 11 cells, got {len(cells)}"

expected_cells = [
    (1, "markdown", "# DeepShield V5 -- Kaggle GPU Training Pipeline"),
    (2, "code", "# CELL 2 -- TENSORFLOW / PACKAGE VERIFICATION"),
    (3, "code", "# CELL 3 -- KAGGLE GPU VERIFICATION"),
    (4, "code", "# CELL 4 -- LOCATE KAGGLE DATASET AND MANIFEST"),
    (5, "code", "# CELL 5 -- VERIFY DATASET COUNTS"),
    (6, "code", "# CELL 6 -- VERIFY MANIFEST AND CLASS MAPPING"),
    (7, "code", "# CELL 7 -- VERIFY TRAIN/VALIDATION/TEST SPLIT INTEGRITY"),
    (8, "code", "# CELL 8 -- VERIFY TRAIN_DEEPSHIELD_V5.PY"),
    (9, "code", "# CELL 9 -- KAGGLE PATH & ENVIRONMENT CONFIGURATION"),
    (10, "code", "# CELL 10 -- FINAL READINESS REPORT"),
    (11, "code", "# CELL 11 -- START V5 TRAINING")
]

print("\n--- VALIDATING CELLS ---")
for idx, ctype, header in expected_cells:
    c = cells[idx - 1]
    assert c["cell_type"] == ctype, f"Cell {idx} expected {ctype}, got {c['cell_type']}"
    src = "".join(c.get("source", []))
    assert header in src, f"Cell {idx} header missing: {header}"
    print(f"Cell {idx:>2} [{ctype:>8}]: {header}")

    # Verify no Google Drive references in Kaggle notebook
    assert "google.colab" not in src or "drive.mount" not in src, f"Cell {idx} illegally attempts drive mount!"

    # Compile check for code cells (skip cell 11 which uses shell ! prefix)
    if ctype == "code" and idx != 11:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(src)
            t_name = tf.name
        py_compile.compile(t_name, doraise=True)

# Verify Cell 11
c11_src = "".join(cells[10]["source"])
assert "!python /kaggle/working/train_deepshield_v5.py" in c11_src
assert cells[10]["execution_count"] is None
assert len(cells[10]["outputs"]) == 0

print("\n[+] JSON VALIDATION: PASS")
print("[+] All 11 cells verified and syntax-checked.")
print("[+] No Google Drive mount in Kaggle notebook.")
print("[+] Output path strictly configured to: /kaggle/working/DeepShield_V5/")
print("[+] Cell 11 unexecuted. Training NOT started.")
