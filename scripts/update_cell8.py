import json

notebook_path = "training/DeepShield_V5_Colab.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["cells"][7]["source"] = [
    "# CELL 8 -- VERIFY GPU\n",
    "import tensorflow as tf\n",
    "\n",
    "tf_version = tf.__version__\n",
    "gpu_devices = tf.config.list_physical_devices('GPU')\n",
    "gpu_count = len(gpu_devices)\n",
    "gpu_ok = (gpu_count > 0)\n",
    "\n",
    "device_names = []\n",
    "for dev in gpu_devices:\n",
    "    try:\n",
    "        details = tf.config.experimental.get_device_details(dev)\n",
    "        device_names.append(details.get('device_name', dev.name))\n",
    "    except Exception:\n",
    "        device_names.append(dev.name)\n",
    "\n",
    "gpu_device_str = \", \".join(device_names) if device_names else \"None\"\n",
    "\n",
    "print(\"=\" * 50)\n",
    "print(\"DEEPSHIELD V5 GPU CHECK\")\n",
    "print(\"=\" * 50)\n",
    "print(f\"TensorFlow:\\n {tf_version}\\n\")\n",
    "print(f\"GPU devices:\\n {gpu_devices}\\n\")\n",
    "print(f\"GPU count:\\n {gpu_count}\\n\")\n",
    "print(f\"GPU device:\\n {gpu_device_str}\")\n",
    "print(\"=\" * 50)\n",
    "\n",
    "if gpu_ok:\n",
    "    print(\"\\nGPU STATUS: READY\")\n",
    "else:\n",
    "    print(\"\\nGPU STATUS: NOT AVAILABLE\")\n",
    "print(\"=\" * 50)\n"
]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print("SUCCESS: Cell 8 updated.")
