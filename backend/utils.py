"""
DeepShield V5 - Backend Utilities Module
========================================
Provides validation, security sanitation, temporary file management,
and unified JSON response formatters.
"""

import os
import uuid
import time
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import jsonify

from backend.config import ALLOWED_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS


def allowed_file(filename: str, allowed_extensions: set = None) -> bool:
    """Verifies whether an uploaded image filename matches allowed extensions."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    valid_exts = allowed_extensions if allowed_extensions is not None else ALLOWED_EXTENSIONS
    return ext in valid_exts


def allowed_video_file(filename: str, allowed_extensions: set = None) -> bool:
    """Verifies whether an uploaded video filename matches allowed video extensions."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    valid_exts = allowed_extensions if allowed_extensions is not None else ALLOWED_VIDEO_EXTENSIONS
    return ext in valid_exts


def secure_unique_filename(original_filename: str, prefix: str = "") -> str:
    """
    Generates a sanitized, collision-free filename preserving the original extension.
    Example output: 'img_20260917_a8f3b20c_uploaded_face.jpg'
    """
    clean_name = secure_filename(original_filename)
    stem = Path(clean_name).stem[:30]  # Limit length
    ext = Path(clean_name).suffix.lower()
    if not ext and "." in original_filename:
        ext = "." + original_filename.rsplit(".", 1)[1].lower()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    prefix_str = f"{prefix}_" if prefix else ""
    return f"{prefix_str}{timestamp}_{unique_id}_{stem}{ext}"


# Alias for compatibility
sanitize_upload_filename = secure_unique_filename


def safe_remove_file(filepath: Path | str | None):
    """Safely removes a temporary file from disk without raising exceptions."""
    if not filepath:
        return
    try:
        p = Path(filepath)
        if p.is_file():
            p.unlink(missing_ok=True)
    except Exception:
        pass


# Alias for compatibility
safe_delete_file = safe_remove_file


def api_success(data: dict = None, status_code: int = 200, **kwargs):
    """Formats a standardized successful JSON API response supporting dicts or kwargs."""
    payload = {"success": True}
    if data and isinstance(data, dict):
        payload.update(data)
    if kwargs:
        payload.update(kwargs)
    return jsonify(payload), status_code


def api_error(message: str, status_code: int = 400, status: int = None, error_code: str = "BAD_REQUEST", details: dict = None, **kwargs):
    """Formats a standardized error JSON API response."""
    code = status if status is not None else status_code
    payload = {
        "success": False,
        "error": {
            "code": error_code,
            "message": message
        }
    }
    if details:
        payload["error"]["details"] = details
    if kwargs:
        payload.update(kwargs)
    return jsonify(payload), code
