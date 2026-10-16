import json
from pathlib import Path

nb_path = Path("training/DeepShield_V5_Colab.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total notebook cells: {len(cells)}")
assert len(cells) == 12, f"Expected 12 cells, got {len(cells)}"

c10 = cells[9]
c11 = cells[10]
c12 = cells[11]

c10_src = "".join(c10.get("source", []))
c11_src = "".join(c11.get("source", []))
c12_src = "".join(c12.get("source", []))

assert "CELL 10 -- FINAL READINESS" in c10_src, "Cell 10 header mismatch"
assert "CELL 11 -- SPEED OPTIMIZATION" in c11_src, "Cell 11 header mismatch"
assert "CELL 12 -- START DEEPSHIELD V5 TRAINING" in c12_src, "Cell 12 header mismatch"

assert "/content/DeepShield_FAST10" in c11_src
assert "/content/DeepShield_FAST10_manifest_split.csv" in c11_src
assert "17208" in c11_src
assert "3688" in c11_src
assert "3693" in c11_src
assert "24589" in c11_src
assert "LOCAL DATASET: READY" in c11_src
assert "LOCAL MANIFEST: READY" in c11_src
assert "TRAINING: NOT STARTED" in c11_src

assert "!python /content/training/train_deepshield_v5.py" in c12_src

print("\n--- CELL ORDER VERIFICATION ---")
print("Cell 10:", c10_src.split("\n")[0])
print("Cell 11:", c11_src.split("\n")[0])
print("Cell 12:", c12_src.split("\n")[0])

print("\n[+] JSON VALIDATION RESULT: PASS")
print("[+] Cell 11 is positioned immediately BEFORE Cell 12.")
print("[+] Training has NOT started.")
