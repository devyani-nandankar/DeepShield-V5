import json
from pathlib import Path

notebook_path = Path("training/DeepShield_V5_Colab.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

cells = data.get("cells", [])
print(f"File exists: {notebook_path.exists()}")
print(f"Total cells: {len(cells)}")

for i, cell in enumerate(cells):
    ctype = cell.get("cell_type")
    first_line = cell.get("source", [""])[0].strip() if cell.get("source") else ""
    print(f"Cell {i+1} [{ctype}]: {first_line}")
