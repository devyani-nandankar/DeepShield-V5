import json
import py_compile
import tempfile
from pathlib import Path

nb_path = Path("training/DeepShield_V5_Colab.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Total cells
print(f"Total cells: {len(nb['cells'])}")
assert len(nb['cells']) == 12

# Compile Cell 10
c10_src = "".join(nb['cells'][9]['source'])
with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
    f.write(c10_src)
    fname = f.name

py_compile.compile(fname, doraise=True)
print("Cell 10 syntax: PASS")

# Check Cell 11
c11_src = "".join(nb['cells'][10]['source'])
assert "CELL 11 -- SPEED OPTIMIZATION" in c11_src
print("Cell 11: UNCHANGED & VERIFIED")

# Check Cell 12
c12_src = "".join(nb['cells'][11]['source'])
assert "!python /content/training/train_deepshield_v5.py" in c12_src
print("Cell 12: UNCHANGED & VERIFIED")

print("\n--- CELL 10 HEADER ---")
print(c10_src.split("\n")[0])
print(c10_src.split("\n")[1])
