import json

notebook_path = "training/DeepShield_V5_Colab.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["cells"][2]["source"] = [
    "# CELL 3 -- MOUNT GOOGLE DRIVE\n",
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "mount_success = False\n",
    "mydrive_accessible = False\n",
    "\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    mount_success = True\n",
    "except Exception as e:\n",
    "    print(f\"[!] Error during drive.mount: {e}\")\n",
    "\n",
    "mydrive_path = Path(\"/content/drive/MyDrive\")\n",
    "if mydrive_path.exists():\n",
    "    mydrive_accessible = True\n",
    "\n",
    "print(\"=\" * 50)\n",
    "print(\"GOOGLE DRIVE CHECK\")\n",
    "print(\"=\" * 50)\n",
    "print(f\"Drive mount: {'PASS' if mount_success else 'FAIL'}\")\n",
    "print(f\"MyDrive accessible: {'PASS' if mydrive_accessible else 'FAIL'}\")\n",
    "\n",
    "if mount_success and mydrive_accessible:\n",
    "    print(\"\\nGOOGLE DRIVE: READY\")\n",
    "else:\n",
    "    print(\"\\nGOOGLE DRIVE: FAILED\")\n",
    "print(\"=\" * 50)\n"
]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print("SUCCESS: Cell 3 updated.")
