"""
DeepShield V5 - Metadata Forensics Module
=========================================
Extracts objective image metadata and EXIF properties using Pillow.
Provides evidence of editing tools, camera hardware, and timestamps
without fabricating conclusions or assumptions.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image, ExifTags

logger = logging.getLogger("deepshield.metadata")

# Known image manipulation & generative AI software signatures
SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "canva", "picsart", "snapseed",
    "stable diffusion", "midjourney", "dall-e", "novelai", "comfyui",
    "automatic1111", "faceapp", "reface", "deepfacelab"
]


class MetadataForensics:
    """
    Forensic extractor for image metadata, EXIF headers, and compression traces.
    """

    @staticmethod
    def extract_metadata(image_path: str) -> Dict[str, Any]:
        """
        Extracts objective technical metadata and EXIF properties from an image file.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            Dictionary containing structured metadata forensics.
        """
        p = Path(image_path)
        if not p.is_file():
            return {
                "success": False,
                "error": f"Image file not found: {image_path}"
            }

        file_size_bytes = p.stat().st_size

        result: Dict[str, Any] = {
            "success": True,
            "filename": p.name,
            "file_size_bytes": file_size_bytes,
            "file_size_human": MetadataForensics._format_bytes(file_size_bytes),
            "format": None,
            "mode": None,
            "dimensions": {"width": 0, "height": 0},
            "has_exif": False,
            "camera_make": None,
            "camera_model": None,
            "software": None,
            "datetime_original": None,
            "datetime_digitized": None,
            "has_gps": False,
            "editing_software_detected": False,
            "editing_software_name": None,
            "is_ai_generator_indicated": False,
            "raw_exif": {},
            "forensic_notes": []
        }

        try:
            with Image.open(p) as img:
                result["format"] = img.format
                result["mode"] = img.mode
                result["dimensions"] = {
                    "width": img.width,
                    "height": img.height
                }

                # Attempt EXIF extraction
                exif_data = img.getexif()
                if exif_data and len(exif_data) > 0:
                    result["has_exif"] = True
                    parsed_exif = {}

                    for tag_id, value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        # Filter out large binary tags
                        if isinstance(value, bytes):
                            if len(value) > 64:
                                continue
                            try:
                                value = value.decode("utf-8", errors="ignore").strip()
                            except Exception:
                                continue
                        parsed_exif[tag_name] = str(value)

                    result["raw_exif"] = parsed_exif

                    # Extract specific fields
                    if "Make" in parsed_exif:
                        result["camera_make"] = parsed_exif["Make"].strip()
                    if "Model" in parsed_exif:
                        result["camera_model"] = parsed_exif["Model"].strip()
                    if "Software" in parsed_exif:
                        software_str = parsed_exif["Software"].strip()
                        result["software"] = software_str
                        result["editing_software_name"] = software_str

                        # Check for known editing or AI signatures
                        sw_lower = software_str.lower()
                        for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
                            if kw in sw_lower:
                                result["editing_software_detected"] = True
                                if kw in ["stable diffusion", "midjourney", "dall-e", "novelai", "comfyui", "faceapp", "reface", "deepfacelab"]:
                                    result["is_ai_generator_indicated"] = True
                                break

                    if "DateTimeOriginal" in parsed_exif:
                        result["datetime_original"] = parsed_exif["DateTimeOriginal"]
                    elif "DateTime" in parsed_exif:
                        result["datetime_original"] = parsed_exif["DateTime"]

                    if "DateTimeDigitized" in parsed_exif:
                        result["datetime_digitized"] = parsed_exif["DateTimeDigitized"]

                    if "GPSInfo" in parsed_exif:
                        result["has_gps"] = True

            # Synthesize objective forensic observations
            notes: List[str] = []
            if result["has_exif"]:
                notes.append(f"EXIF header present with {len(result['raw_exif'])} recorded tags.")
                if result["camera_make"] or result["camera_model"]:
                    notes.append(f"Device recorded: {result['camera_make'] or ''} {result['camera_model'] or ''}".strip())
                if result["editing_software_detected"]:
                    notes.append(f"Editing software signature detected: '{result['editing_software_name']}'.")
                if result["is_ai_generator_indicated"]:
                    notes.append(f"AI generation/manipulation signature matched: '{result['editing_software_name']}'.")
            else:
                notes.append("No EXIF metadata found. Metadata may have been stripped during web upload, social media re-encoding, or synthetic generation.")

            result["forensic_notes"] = notes
            return result

        except Exception as e:
            logger.error(f"Error extracting metadata from {image_path}: {e}", exc_info=True)
            result["forensic_notes"].append(f"Metadata parsing encountered an issue: {str(e)}")
            return result

    @staticmethod
    def _format_bytes(size: int) -> str:
        """Helper to format byte sizes into readable strings."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"
