import json

notebook_path = "training/DeepShield_V5_Colab.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["cells"][8]["source"] = [
    "# CELL 9 -- SYNTAX CHECK\n",
    "from pathlib import Path\n",
    "import py_compile\n",
    "\n",
    "target_script = Path(\"/content/training/train_deepshield_v5.py\")\n",
    "\n",
    "syntax_pass = False\n",
    "print(\"=\" * 50)\n",
    "print(\"DEEPSHIELD V5 SYNTAX CHECK\")\n",
    "print(\"=\" * 50)\n",
    "print(\"File:\")\n",
    "print(f\" {target_script}\\n\")\n",
    "\n",
    "if target_script.is_file():\n",
    "    try:\n",
    "        py_compile.compile(str(target_script), doraise=True)\n",
    "        syntax_pass = True\n",
    "        print(\"Syntax:\")\n",
    "        print(\" PASS\\n\")\n",
    "        print(\"V5 TRAINING SCRIPT SYNTAX: PASS\")\n",
    "    except py_compile.PyCompileError as err:\n",
    "        print(\"Syntax:\")\n",
    "        print(\" FAIL\\n\")\n",
    "        print(f\"[!] Syntax Error details:\\n{err}\")\n",
    "    except Exception as ex:\n",
    "        print(\"Syntax:\")\n",
    "        print(\" FAIL\\n\")\n",
    "        print(f\"[!] Error during compilation check:\\n{ex}\")\n",
    "else:\n",
    "    print(\"Syntax:\")\n",
    "    print(\" FAIL\\n\")\n",
    "    print(f\"[!] File not found at: {target_script}\")\n",
    "\n",
    "print(\"=\" * 50)\n"
]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print("SUCCESS: Cell 9 updated.")
