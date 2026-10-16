"""
DeepShield V5 - Backend Configuration Module
============================================
Defines operational parameters, calibrated thresholds, security rules,
and model asset locations for the DeepShield multimodal deepfake detection backend.
"""

import os
from pathlib import Path

# Base Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Models & Weights Locations
# Primary DeepShield V5 Keras model
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "deepshield_v5_clean_best.keras"
MODEL_PATH = Path(os.getenv("DEEPSHIELD_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

# OpenCV YuNet ONNX Face Detection model
DEFAULT_YUNET_PATH = PROJECT_ROOT / "models" / "yunet" / "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = Path(os.getenv("DEEPSHIELD_YUNET_PATH", str(DEFAULT_YUNET_PATH)))

# Calibrated Decision Threshold
# Benchmark test analysis demonstrated optimal F1 at 0.21 on the website pipeline (Precision: 81.15%, Recall: 94.26%, F1: 0.8722)
# Note: Configurable via DEEPSHIELD_THRESHOLD environment variable.
CALIBRATED_THRESHOLD = float(os.getenv("DEEPSHIELD_THRESHOLD", "0.21"))

# DeepShield V5.2 Dedicated Temporal Video Model (Model A: No Augmentation)
DEFAULT_V5_2_TEMPORAL_MODEL_PATH = PROJECT_ROOT / "models" / "v5_2" / "deepshield_v5_2_temporal_no_aug.keras"
V5_2_TEMPORAL_MODEL_PATH = Path(os.getenv("DEEPSHIELD_V5_2_MODEL_PATH", str(DEFAULT_V5_2_TEMPORAL_MODEL_PATH)))

# Calibrated Temporal Decision Threshold (derived from validation set F1 optimization)
VIDEO_TEMPORAL_THRESHOLD = float(os.getenv("DEEPSHIELD_VIDEO_TEMPORAL_THRESHOLD", "0.30"))
VIDEO_TEMPORAL_MARGIN = float(os.getenv("DEEPSHIELD_VIDEO_TEMPORAL_MARGIN", "0.03"))

# Model Input Specifications
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Class Mapping (Sigmoid Output: 0 = Real, 1 = Fake)
CLASS_MAPPING = {
    0: "Real",
    1: "Fake"
}

# Storage & Artifact Directories
UPLOAD_FOLDER = BACKEND_DIR / "uploads"
GENERATED_FOLDER = BACKEND_DIR / "generated"
FRONTEND_FOLDER = PROJECT_ROOT / "frontend"

# Ensure runtime directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
GENERATED_FOLDER.mkdir(parents=True, exist_ok=True)
FRONTEND_FOLDER.mkdir(parents=True, exist_ok=True)

# File Validation & Security Constraints
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB maximum payload

# YuNet Detection Parameters
YUNET_SCORE_THRESHOLD = float(os.getenv("YUNET_SCORE_THRESHOLD", "0.6"))
YUNET_NMS_THRESHOLD = float(os.getenv("YUNET_NMS_THRESHOLD", "0.3"))
YUNET_TOP_K = int(os.getenv("YUNET_TOP_K", "5000"))
FACE_CROP_MARGIN = float(os.getenv("FACE_CROP_MARGIN", "0.15"))  # 15% bounding box margin

# CORS Settings (Allow React development servers and production domains)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,*").split(",")

# Flask Runtime Configuration
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
DEBUG = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")
