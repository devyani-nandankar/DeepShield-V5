"""
DeepShield V5 - Multimodal Deepfake Detection Web Backend
=========================================================
Flask REST API powering the DeepShield web platform for explainable
multimodal deepfake detection and digital media authenticity verification.
"""

import os
import time
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

from backend.config import (
    UPLOAD_FOLDER,
    GENERATED_FOLDER,
    FRONTEND_FOLDER,
    MAX_CONTENT_LENGTH,
    ALLOWED_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    CALIBRATED_THRESHOLD,
    FLASK_HOST,
    FLASK_PORT,
    DEBUG,
    CORS_ORIGINS
)
from backend.utils import (
    allowed_file,
    sanitize_upload_filename,
    safe_delete_file,
    api_success,
    api_error
)
from backend.image_detector import get_image_detector
from backend.video_detector import get_video_detector
from backend.face_detector import get_face_detector
from backend.report_generator import ForensicReportGenerator
from backend.fusion import MultimodalFusionEngine

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("deepshield.api")

# Server start timestamp for uptime reporting
START_TIME = time.time()

# Create Flask Application
app = Flask(
    __name__,
    static_folder=str(FRONTEND_FOLDER),
    static_url_path="",
    template_folder=str(FRONTEND_FOLDER)
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["GENERATED_FOLDER"] = str(GENERATED_FOLDER)

# Configure CORS
if CORS_AVAILABLE:
    CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})
else:
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response


# ==============================================================================
# ERROR HANDLERS
# ==============================================================================
@app.errorhandler(400)
def handle_bad_request(e):
    return api_error(str(e.description if hasattr(e, "description") else "Bad request"), status=400)

@app.errorhandler(404)
def handle_not_found(e):
    return api_error("The requested API endpoint was not found.", status=404)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    max_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
    return api_error(f"Payload exceeds maximum allowed file size of {max_mb:.0f} MB.", status=413)

@app.errorhandler(500)
def handle_server_error(e):
    logger.error(f"Internal server error: {e}", exc_info=True)
    return api_error("An unexpected server error occurred during processing.", status=500)


# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    System health and capability diagnostic endpoint.
    Reports operational status of neural models, face detectors, and calibrated thresholds.
    """
    detector = get_image_detector()
    face_detector = get_face_detector()
    uptime_sec = round(time.time() - START_TIME, 1)

    return api_success(
        message="DeepShield Multimodal API is healthy and operational.",
        status="online",
        service="DeepShield Multimodal Verification Platform",
        version="5.0.0",
        uptime_seconds=uptime_sec,
        model_info={
            "loaded": detector.is_model_loaded(),
            "model_path": str(detector.model_path),
            "backbone": "EfficientNetB2",
            "input_resolution": "224x224x3",
            "class_mapping": {"0": "Real", "1": "Fake"},
            "calibrated_threshold": CALIBRATED_THRESHOLD,
            "benchmark_test_metrics": {
                "accuracy": "77.55%",
                "roc_auc": "83.30%",
                "precision": "87.83%",
                "recall": "80.44%",
                "optimal_f1": 0.8771
            }
        },
        face_detector_info={
            "yunet_loaded": face_detector.is_available(),
            "model_path": str(face_detector.model_path)
        }
    )


@app.route("/api/predict/image", methods=["POST"])
def predict_image():
    """
    Primary Image Authenticity Verification Endpoint.
    Accepts multipart/form-data image, executes YuNet face detection,
    EfficientNetB2 inference, Grad-CAM saliency generation, and metadata forensics.
    """
    # 1. Validate File Upload Request
    file = None
    for field_name in ["file", "image", "media"]:
        if field_name in request.files:
            file = request.files[field_name]
            break

    if file is None or file.filename == "":
        return api_error("No image file provided in request. Submit with key 'file' or 'image'.", status=400)

    # 2. Validate File Extension
    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        exts = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return api_error(f"Unsupported file type. Supported extensions: {exts}", status=400)

    # 3. Save Upload Securely
    safe_filename = sanitize_upload_filename(file.filename)
    upload_path = UPLOAD_FOLDER / safe_filename

    try:
        file.save(str(upload_path))
    except Exception as e:
        logger.error(f"Failed to save upload to disk: {e}")
        return api_error("Server failed to save uploaded image.", status=500)

    try:
        # 4. Execute Detection Pipeline
        detector = get_image_detector()
        result = detector.predict(str(upload_path))

        if not result.get("success", False):
            # If model is not loaded, return clear 503
            if not result.get("model_available", True):
                return api_error(result.get("error", "Model weights missing."), status=503)
            return api_error(result.get("error", "Inference failed."), status=500)

        # 5. Multimodal Fusion & Risk Analysis
        fusion = MultimodalFusionEngine.evaluate_fusion(result)
        result["fusion"] = fusion

        # 6. Conform to specific response contract required by specifications
        display_verdict = result.get("display_verdict", "Likely Fake" if result["prediction"] == "Fake" else "Real")
        response_payload = {
            "success": True,
            "media_type": "image",
            "prediction": result["prediction"],
            "display_verdict": display_verdict,
            "score": result["score"],
            "confidence": result["confidence"],
            "threshold": result["threshold"],
            "face_detected": result["face_detected"],
            "face_count": result["face_count"],
            "explanation": result["explanation"],
            "bullets": result.get("bullets", []),
            "disclaimer": result.get("disclaimer", "DeepShield provides an AI-based assessment and should not be treated as absolute proof of authenticity."),
            "visual_evidence": result.get("visual_evidence", {}),
            "technical_details": result.get("technical_details", {}),
            "evidence": result["evidence"],
            "gradcam_available": result.get("gradcam", {}).get("available", False),
            "gradcam": result.get("gradcam", {}),
            "metadata": result.get("metadata", {}),
            "fusion": fusion,
            "model_info": result.get("model_info", {})
        }

        return jsonify(response_payload), 200

    finally:
        # Safe cleanup of uploaded temp file
        safe_delete_file(upload_path)


@app.route("/api/generated/<path:filename>", methods=["GET"])
def serve_generated(filename: str):
    """
    Safely serves generated artifacts (Grad-CAM heatmaps, visual overlays)
    from the backend/generated folder.
    """
    safe_name = Path(filename).name
    target_file = GENERATED_FOLDER / safe_name
    if not target_file.is_file():
        return api_error(f"Artifact '{safe_name}' not found.", status=404)

    return send_from_directory(str(GENERATED_FOLDER), safe_name)


@app.route("/api/sample/<filename>", methods=["GET"])
def serve_sample(filename: str):
    """
    Serves sample test images (e.g. real.jpeg, fake.jpeg) from backend/uploads/
    for quick demonstration in the web interface.
    """
    safe_name = Path(filename).name
    sample_path = Path(__file__).resolve().parent / "uploads" / safe_name
    if not sample_path.is_file():
        return api_error(f"Sample '{safe_name}' not found.", status=404)
    return send_from_directory(str(sample_path.parent), safe_name)


@app.route("/api/predict/video", methods=["POST"])
def predict_video():
    """
    Video Authenticity Inspection Endpoint (Decoupled Interface).
    Accepts video uploads, inspects stream properties, and provides
    guidance on frame extraction while temporal models are staged.
    """
    file = None
    for field_name in ["file", "video", "media"]:
        if field_name in request.files:
            file = request.files[field_name]
            break

    if file is None or file.filename == "":
        return api_error("No video file provided in request. Submit with key 'file' or 'video'.", status=400)

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        exts = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        return api_error(f"Unsupported video format. Supported formats: {exts}", status=400)

    safe_filename = sanitize_upload_filename(file.filename)
    upload_path = UPLOAD_FOLDER / safe_filename

    try:
        file.save(str(upload_path))
    except Exception as e:
        logger.error(f"Failed to save video upload to disk: {e}")
        return api_error("Server failed to save uploaded video.", status=500)

    try:
        video_detector = get_video_detector()
        result = video_detector.predict(str(upload_path))
        if "fusion" not in result:
            result["fusion"] = MultimodalFusionEngine.evaluate_fusion(video_prediction=result)
        return jsonify(result), 200
    finally:
        safe_delete_file(upload_path)


@app.route("/api/report", methods=["POST"])
def generate_forensic_report():
    """
    Generates a full forensic audit report.
    Can accept either:
    1. A raw image or video file in multipart/form-data (runs prediction then compiles report)
    2. A previously computed prediction result object in JSON body: {"prediction_result": {...}}
    """
    report_format = request.args.get("format", "json").lower()

    # Case A: JSON body containing prior prediction
    if request.is_json:
        data = request.get_json() or {}
        pred = data.get("prediction_result")
        if not pred:
            return api_error("Missing 'prediction_result' in JSON request body.", status=400)

        report = ForensicReportGenerator.generate_report(pred)
        if report_format == "markdown":
            md_text = ForensicReportGenerator.format_markdown(report)
            return md_text, 200, {"Content-Type": "text/markdown; charset=utf-8"}
        return jsonify(report), 200

    # Case B: File upload (image or video)
    file = None
    for field_name in ["file", "image", "video"]:
        if field_name in request.files:
            file = request.files[field_name]
            break

    if file is None or file.filename == "":
        return api_error("Please submit an image or video file, or a JSON payload containing 'prediction_result'.", status=400)

    is_image = allowed_file(file.filename, ALLOWED_EXTENSIONS)
    is_video = allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS)

    if not is_image and not is_video:
        return api_error("Unsupported file format. Supported: images (JPG, PNG, WEBP) and videos (MP4, AVI, MOV, MKV).", status=400)

    safe_filename = sanitize_upload_filename(file.filename)
    upload_path = UPLOAD_FOLDER / safe_filename

    try:
        file.save(str(upload_path))
        if is_image:
            detector = get_image_detector()
            result = detector.predict(str(upload_path))
            if not result.get("success"):
                return api_error(result.get("error", "Image processing failed."), status=500)
        else:
            detector = get_video_detector()
            result = detector.predict(str(upload_path))
            if not result.get("success"):
                return api_error(result.get("error", "Video processing failed."), status=500)

        report = ForensicReportGenerator.generate_report(result)
        if report_format == "markdown":
            md_text = ForensicReportGenerator.format_markdown(report)
            return md_text, 200, {"Content-Type": "text/markdown; charset=utf-8"}
        return jsonify(report), 200
    finally:
        safe_delete_file(upload_path)


# ==============================================================================
# FRONTEND APPLICATION ROUTES
# ==============================================================================
@app.route("/", methods=["GET"])
def serve_index():
    """Serves the DeepShield Explainable AI Single Page Application."""
    index_file = FRONTEND_FOLDER / "index.html"
    if index_file.is_file():
        return send_from_directory(str(FRONTEND_FOLDER), "index.html", mimetype="text/html")
    return api_error("DeepShield frontend homepage (index.html) not found.", status=404)


@app.route("/<path:path>", methods=["GET"])
def serve_frontend_assets(path):
    """Serves static assets or falls back to index.html."""
    target_file = FRONTEND_FOLDER / path
    if target_file.is_file():
        return send_from_directory(str(FRONTEND_FOLDER), path)
    index_file = FRONTEND_FOLDER / "index.html"
    if index_file.is_file():
        return send_from_directory(str(FRONTEND_FOLDER), "index.html")
    return api_error(f"Asset '{path}' not found.", status=404)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    # Ensure directories exist
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    GENERATED_FOLDER.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" DeepShield V5 - Multimodal Deepfake Verification Backend")
    print(f" Listening on: http://{FLASK_HOST}:{FLASK_PORT}")
    print(f" Calibrated Threshold: {CALIBRATED_THRESHOLD}")
    print("=" * 70 + "\n")

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=DEBUG
    )
