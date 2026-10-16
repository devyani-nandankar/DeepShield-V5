import json

with open("training/DeepShield_V5_Kaggle.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    lines = src.splitlines()
    first = lines[0] if lines else "EMPTY"
    second = lines[1] if len(lines) > 1 else ""
    print(f"Cell {i+1} ({c['cell_type']}):\n  Line 1: {first}\n  Line 2: {second}\n")
