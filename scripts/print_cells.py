import json

with open('training/DeepShield_V5_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    ctype = cell.get('cell_type', '')
    src = ''.join(cell.get('source', []))
    header = src.split('\n')[0] if src else ''
    print(f"Cell {i+1} [{ctype}]: {header}")
    if i >= 9:
        print("--- Full source ---")
        print(src)
        print("-------------------")
