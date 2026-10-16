import io
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app

client = app.test_client()

print("=" * 60)
print(" DeepShield V5 API Prediction Test (cv2.INTER_AREA)")
print("=" * 60)

for img_name in ['real.jpeg', 'fake.jpeg']:
    p = PROJECT_ROOT / 'backend' / 'uploads' / img_name
    if not p.is_file():
        print(f"Error: {p} not found!")
        continue

    with open(p, 'rb') as f:
        img_bytes = f.read()

    response = client.post(
        '/api/predict/image',
        data={'image': (io.BytesIO(img_bytes), img_name)},
        content_type='multipart/form-data'
    )

    data = response.get_json() or {}
    print(f"\n--- {img_name} ---")
    print(f"HTTP Status:     {response.status_code}")
    print(f"Prediction:      {data.get('prediction')}")
    print(f"Raw Score:       {data.get('score')}")
    print(f"Confidence:      {data.get('confidence')}")
    print(f"Threshold:       {data.get('threshold')}")
    print(f"Face Detected:   {data.get('face_detected')}")
    print(f"Face Count:      {data.get('face_count')}")
    print(f"GradCAM Avail:   {data.get('gradcam_available')}")
    print(f"Explanation:     {data.get('explanation')}")

print("\n" + "=" * 60)
