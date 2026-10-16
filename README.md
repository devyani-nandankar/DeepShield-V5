# DeepShield

**An Explainable Multimodal Deepfake Detection and Digital Media Authenticity Verification Platform**

DeepShield is an academic and forensic-grade multimodal verification platform engineered to detect digital face manipulations and synthetic media artifacts across image and video assets.

---

## 1. DeepShield V5 Architecture & Benchmark Performance

- **Backbone Network**: EfficientNetB2 (ImageNet transfer-pretrained)
- **Input Dimensions**: 224 × 224 × 3 (RGB Float32 `[0, 255]`)
- **Classification Head**: `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(128, ReLU)` → `Dropout(0.2)` → `Dense(1, Sigmoid)`
- **Class Encoding**: `0 = Real`, `1 = Fake` (higher sigmoid score indicates higher probability of manipulation)
- **Model Checkpoint**: `models/deepshield_v5_clean_best.keras`

### Empirical Test Benchmark Metrics (DeepShield FAST10 Independent Test Split: 3,693 images)
- **Accuracy**: `77.55%`
- **ROC-AUC**: `83.30%`
- **Precision**: `87.83%`
- **Recall**: `80.44%`
- **Calibrated Optimal F1 Threshold**: `0.16`
- **Optimal F1 Score**: `0.8771`

> **Scientific Integrity Notice**: Benchmark metrics are derived strictly from the evaluated test split. The platform reports genuine performance without unsupported 90%+ accuracy claims.

---

## 2. Directory Structure

```text
DeepShield_V5_Final/
├── backend/
│   ├── app.py                   # Flask REST API application
│   ├── config.py                # Operational configuration & calibrated thresholds
│   ├── image_detector.py        # V5 inference engine with singleton model management
│   ├── face_detector.py         # OpenCV YuNet CNN face detector with dynamic resizing
│   ├── gradcam.py               # Grad-CAM explainability engine & colormap blending
│   ├── metadata_forensics.py    # Pillow-based EXIF and editing software forensic extractor
│   ├── video_detector.py        # Decoupled video temporal inspection interface
│   ├── fusion.py                # Adaptive multimodal fusion & uncertainty quantification
│   ├── report_generator.py      # Structured 10-section forensic audit report compiler
│   ├── utils.py                 # File sanitization, safety cleanup & response formatters
│   ├── requirements.txt         # Backend Python dependencies
│   ├── uploads/                 # Ephemeral upload buffer (auto-cleaned)
│   ├── generated/               # Artifact storage for Grad-CAM heatmaps & overlays
│   └── tests/                   # Backend unit tests
│       └── test_backend.py
│
├── models/
│   ├── deepshield_v5_clean_best.keras  # DeepShield V5 trained model weights
│   ├── deepshield_final.keras         # DeepShield V4 legacy model (preserved untouched)
│   └── yunet/
│       └── face_detection_yunet_2023mar.onnx  # YuNet face detector ONNX weights
│
├── training/
│   ├── train_deepshield_v5.py         # Two-stage training pipeline (v5.0.3-face-path-fix)
│   └── DeepShield_V5_Kaggle.ipynb     # Kaggle verification & training notebook
│
├── requirements.txt
└── README.md
```

---

## 3. REST API Specifications

### `GET /api/health`
System diagnostic endpoint reporting runtime status, model availability, and calibrated threshold.
```json
{
  "success": true,
  "status": "online",
  "service": "DeepShield Multimodal Verification Platform",
  "version": "5.0.0",
  "model_info": {
    "loaded": true,
    "backbone": "EfficientNetB2",
    "calibrated_threshold": 0.16,
    "benchmark_test_metrics": {
      "accuracy": "77.55%",
      "roc_auc": "83.30%",
      "precision": "87.83%",
      "recall": "80.44%",
      "optimal_f1": 0.8771
    }
  }
}
```

### `POST /api/predict/image`
Accepts `multipart/form-data` with key `file` or `image`. Performs YuNet face localization, EfficientNetB2 classification, Grad-CAM saliency generation, and metadata forensics.
```json
{
  "success": true,
  "media_type": "image",
  "prediction": "Fake",
  "score": 0.8421,
  "confidence": 0.906,
  "threshold": 0.16,
  "face_detected": true,
  "face_count": 1,
  "explanation": "DeepShield V5 identified digital manipulation indicators with a deepfake probability score of 0.8421 (84.2%), exceeding the benchmark calibrated decision threshold of 0.16.",
  "evidence": [
    {
      "source": "EfficientNetB2 Deep Learning Classifier",
      "type": "neural_network_classification",
      "finding": "Sigmoid output score is 0.8421 vs calibrated threshold 0.16.",
      "supports": "Fake"
    }
  ],
  "gradcam_available": true,
  "gradcam": {
    "available": true,
    "heatmap_url": "/api/generated/gradcam_heatmap_abcd1234ef56.jpg",
    "overlay_url": "/api/generated/gradcam_overlay_abcd1234ef56.jpg",
    "peak_intensity": 0.94,
    "salient_ratio": 0.26
  }
}
```

### `GET /api/generated/<filename>`
Serves generated Grad-CAM heatmap and overlay visualization images.

### `POST /api/predict/video`
Accepts video uploads in `mp4`, `avi`, `mov`, `mkv`, `webm` formats. Returns decoupled pipeline status and keyframe extraction guidance.

### `POST /api/report`
Compiles a 10-section forensic audit report. Accepts either an image upload (`multipart/form-data`) or a prior prediction JSON payload (`{"prediction_result": {...}}`).
- Use query parameter `?format=markdown` for GitHub-formatted markdown reports.

---

## 4. Setup & Running the Backend

### Prerequisites
- Python 3.10+
- TensorFlow 2.15+
- OpenCV (`opencv-python` or `opencv-python-headless`)

### Step 1: Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 2: Model Weights Deployment
Ensure model weights are located at:
- `models/deepshield_v5_clean_best.keras` (DeepShield V5 model)
- `models/yunet/face_detection_yunet_2023mar.onnx` (YuNet face detector)

*(Note: If weights are pending, the backend will still run and report missing weight status via `/api/health` without crashing).*

### Step 3: Run Backend Server
```bash
python -m backend.app
```
By default, the server binds to `http://0.0.0.0:5000`.

### Step 4: Run Test Suite
```bash
python -m unittest backend/tests/test_backend.py
```
