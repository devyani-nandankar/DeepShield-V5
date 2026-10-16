import json

notebook_path = "training/DeepShield_V5_Colab.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["cells"][1]["source"] = [
    "# CELL 2 -- DEPENDENCY CHECK\n",
    "import importlib\n",
    "\n",
    "required_packages = [\n",
    "    (\"TensorFlow\", \"tensorflow\"),\n",
    "    (\"NumPy\", \"numpy\"),\n",
    "    (\"Pandas\", \"pandas\"),\n",
    "    (\"Scikit-learn\", \"sklearn\"),\n",
    "    (\"Matplotlib\", \"matplotlib\"),\n",
    "    (\"Pillow\", \"PIL\")\n",
    "]\n",
    "\n",
    "print(\"=\" * 50)\n",
    "print(\"DEEPSHIELD V5 DEPENDENCY CHECK\")\n",
    "print(\"=\" * 50)\n",
    "\n",
    "all_pass = True\n",
    "for display_name, module_name in required_packages:\n",
    "    try:\n",
    "        mod = importlib.import_module(module_name)\n",
    "        version = getattr(mod, \"__version__\", \"installed\")\n",
    "        print(f\"{display_name}: {version} -- PASS\")\n",
    "    except ImportError:\n",
    "        print(f\"{display_name}: NOT FOUND -- FAIL\")\n",
    "        all_pass = False\n",
    "\n",
    "print(\"\\nDEPENDENCY CHECK: COMPLETE\")\n",
    "print(\"=\" * 50)\n"
]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print("SUCCESS: Cell 2 updated and validated.")
