"""
DeepShield V5 - Adaptive Multimodal Evidence Fusion Engine
===========================================================
Synthesizes predictions across disparate forensic evidence sources:
1. Neural Image Analysis (EfficientNetB2 V5 spatial features & calibrated score)
2. Temporal Video Analysis (Multi-frame temporal consistency, volatility, & persistence)
3. Video Quality & Compression Forensics (Resolution, bitrate, blur variance, codec degradation)
4. Metadata Forensics (EXIF integrity, editing software, and AI generator signatures)

Key Architectural Principles:
- Adaptive Reliability Weighting: Never blind averaging; degraded or volatile streams are down-weighted.
- Inter-Modal Conflict Detection: Identifies directional discordance between visual, temporal, and metadata findings.
- 3-State Uncertainty Resolution: Formally reports 'Uncertain' on severe conflicts, degraded media, or insufficient data.
- Forensic Integrity: Strictly factual explanations without unwarranted claims (e.g. metadata never 'proves' fake;
  Grad-CAM highlights influential regions, not proof of manipulation).
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

from backend.config import CALIBRATED_THRESHOLD, VIDEO_TEMPORAL_THRESHOLD

logger = logging.getLogger("deepshield.fusion")

# Uncertainty buffer width around calibrated decision threshold (0.21)
# Scores within [CALIBRATED_THRESHOLD - UNCERTAINTY_MARGIN, CALIBRATED_THRESHOLD + UNCERTAINTY_MARGIN]
# (i.e. [0.16, 0.26]) represent model boundary decision ambiguity.
UNCERTAINTY_MARGIN = 0.05
DISCLAIMER_TEXT = "This is an AI-based assessment, not absolute proof of authenticity."


def calibrate_temporal_score(raw_temporal_score: Optional[float], temporal_threshold: float = VIDEO_TEMPORAL_THRESHOLD) -> Optional[float]:
    """
    Maps V5.2 temporal score (threshold 0.30, higher=Fake) into the standard DeepShield
    calibrated score space (threshold 0.21, higher=Fake) to allow seamless multimodal blending.
    Score directions:
      V5.2 temporal score >= 0.30 = Fake (calibrated score >= 0.21)
      V5.2 temporal score < 0.30  = Real (calibrated score < 0.21)
    """
    if raw_temporal_score is None:
        return None
    raw_s = float(np.clip(raw_temporal_score, 0.0, 1.0))
    t_thresh = float(temporal_threshold)
    target_thresh = CALIBRATED_THRESHOLD  # 0.21

    if raw_s < t_thresh:
        # Maps [0.0, t_thresh) -> [0.0, target_thresh)
        calibrated = target_thresh * (raw_s / (t_thresh + 1e-7))
    else:
        # Maps [t_thresh, 1.0] -> [target_thresh, 1.0]
        calibrated = target_thresh + (1.0 - target_thresh) * ((raw_s - t_thresh) / (1.0 - t_thresh + 1e-7))
    return float(np.clip(calibrated, 0.0, 1.0))


def _compute_calibrated_confidence(score: float, threshold: float = CALIBRATED_THRESHOLD) -> float:
    """Computes calibrated confidence in range [0.5, 0.9999] relative to threshold."""
    score_clipped = float(np.clip(score, 0.0, 1.0))
    if score_clipped >= threshold:
        span = 1.0 - threshold
        conf = 0.5 + 0.5 * ((score_clipped - threshold) / (span if span > 1e-6 else 1e-6))
    else:
        span = threshold
        conf = 0.5 + 0.5 * ((threshold - score_clipped) / (span if span > 1e-6 else 1e-6))
    return float(np.clip(conf, 0.5, 0.9999))


def parse_image_evidence(image_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts and normalizes forensic evidence from image analysis."""
    if not image_data or not isinstance(image_data, dict):
        return {
            "available": False,
            "raw_score": None,
            "prediction": None,
            "confidence": None,
            "face_detected": False,
            "face_count": 0,
            "in_boundary_uncertainty": False,
            "reliability_weight": 0.0,
            "findings": []
        }

    # Extract score
    raw_score = None
    for key in ["score", "model_score", "calibrated_score"]:
        if key in image_data and image_data[key] is not None:
            try:
                raw_score = float(image_data[key])
                break
            except (ValueError, TypeError):
                pass

    if raw_score is None:
        return {
            "available": False,
            "raw_score": None,
            "prediction": None,
            "confidence": None,
            "face_detected": False,
            "face_count": 0,
            "in_boundary_uncertainty": False,
            "reliability_weight": 0.0,
            "findings": ["Image payload contained no valid model score."]
        }

    raw_score = float(np.clip(raw_score, 0.0, 1.0))
    face_detected = bool(image_data.get("face_detected", True))
    face_count = int(image_data.get("face_count", 1 if face_detected else 0))
    confidence = float(image_data.get("confidence", _compute_calibrated_confidence(raw_score)))

    # Directional prediction
    pred = image_data.get("prediction")
    if not pred or pred not in ["Real", "Fake", "Likely Fake", "Uncertain"]:
        pred = "Likely Fake" if raw_score >= CALIBRATED_THRESHOLD else "Real"

    # Boundary uncertainty check
    lower_b = max(0.0, CALIBRATED_THRESHOLD - UNCERTAINTY_MARGIN)
    upper_b = min(1.0, CALIBRATED_THRESHOLD + UNCERTAINTY_MARGIN)
    in_boundary = lower_b <= raw_score <= upper_b

    # Dynamic reliability assessment
    reliability = 1.0
    findings = []

    if not face_detected:
        reliability *= 0.40
        findings.append("No localized facial region detected; inference executed on general context.")
    else:
        findings.append(f"Localized face analyzed with model score: {raw_score:.4f}.")

    if in_boundary:
        reliability *= 0.70
        findings.append(f"Score ({raw_score:.4f}) is within threshold boundary window [{lower_b:.2f}, {upper_b:.2f}].")

    gradcam = image_data.get("gradcam", {})
    if gradcam and gradcam.get("available"):
        salient_ratio = gradcam.get("salient_ratio")
        if salient_ratio is not None:
            findings.append(f"Grad-CAM indicates {salient_ratio * 100:.1f}% salient region activation.")

    return {
        "available": True,
        "raw_score": round(raw_score, 4),
        "prediction": pred,
        "confidence": round(confidence, 4),
        "face_detected": face_detected,
        "face_count": face_count,
        "in_boundary_uncertainty": in_boundary,
        "reliability_weight": round(reliability, 3),
        "findings": findings
    }


def parse_video_evidence(video_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts and normalizes forensic evidence from temporal video analysis."""
    if not video_data or not isinstance(video_data, dict):
        return {
            "available": False,
            "verdict": None,
            "prediction": None,
            "confidence": None,
            "median_score": None,
            "p75_score": None,
            "temporal_volatility": None,
            "score_std": None,
            "persistence_ratio": None,
            "compression_degradation": False,
            "uncertainty_reason": None,
            "raw_model_score": None,
            "video_calibrated_score": None,
            "video_score_direction": None,
            "calibration_threshold": CALIBRATED_THRESHOLD,
            "calibration_note": None,
            "valid_prediction_count": 0,
            "total_frames_sampled": 0,
            "face_frames": 0,
            "skipped_no_face_frames": 0,
            "reliability_weight": 0.0,
            "findings": []
        }

    # Extract V5.2 temporal evidence if present
    v5_2_ev = None
    if "v5_2_temporal_evidence" in video_data and isinstance(video_data["v5_2_temporal_evidence"], dict):
        v5_2_ev = dict(video_data["v5_2_temporal_evidence"])
    elif "temporal_evidence" in video_data and isinstance(video_data["temporal_evidence"], dict):
        v5_2_ev = dict(video_data["temporal_evidence"])
    elif video_data.get("temporal_model") in ["DeepShield V5.2", "DeepShield V5.1"]:
        v5_2_ev = {k: v for k, v in video_data.items() if k not in ["fusion", "phase1_video_evidence"]}

    # Extract Phase 1 evidence sub-dict if present
    p1_ev = None
    if "phase1_video_evidence" in video_data and isinstance(video_data["phase1_video_evidence"], dict):
        p1_ev = dict(video_data["phase1_video_evidence"])

    # Verify if video data actually contains analysis
    has_score = (
        ("median_score" in video_data) or
        ("mean_score" in video_data) or
        ("raw_model_score" in video_data) or
        ("video_calibrated_score" in video_data) or
        (v5_2_ev is not None and v5_2_ev.get("temporal_score") is not None)
    )
    verdict = video_data.get("verdict", video_data.get("prediction"))

    if not has_score and not verdict:
        return {
            "available": False,
            "verdict": None,
            "prediction": None,
            "confidence": None,
            "median_score": None,
            "p75_score": None,
            "temporal_volatility": None,
            "score_std": None,
            "persistence_ratio": None,
            "compression_degradation": False,
            "uncertainty_reason": None,
            "raw_model_score": None,
            "video_calibrated_score": None,
            "video_score_direction": None,
            "calibration_threshold": CALIBRATED_THRESHOLD,
            "calibration_note": None,
            "valid_prediction_count": 0,
            "total_frames_sampled": 0,
            "face_frames": 0,
            "skipped_no_face_frames": 0,
            "phase1_video_evidence": None,
            "v5_2_temporal_evidence": None,
            "temporal_evidence": None,
            "reliability_weight": 0.0,
            "findings": ["Video payload contained no valid prediction or scores."]
        }

    raw_model_score = video_data.get("raw_model_score")
    if raw_model_score is None:
        raw_model_score = video_data.get("median_score", video_data.get("mean_score", 0.0))
    raw_model_score = float(raw_model_score) if raw_model_score is not None else 0.0

    video_calibrated_score = video_data.get("video_calibrated_score")
    if video_calibrated_score is not None:
        video_calibrated_score = float(video_calibrated_score)

    video_score_direction = video_data.get("video_score_direction")
    calibration_threshold = float(video_data.get("calibration_threshold", CALIBRATED_THRESHOLD))
    calibration_note = video_data.get("calibration_note")

    median_score = float(video_data.get("median_score", video_data.get("mean_score", raw_model_score)))
    p75_score = float(video_data.get("p75_score", median_score))
    temporal_volatility = float(video_data.get("temporal_volatility", 0.0))
    score_std = float(video_data.get("score_std", 0.0))
    persistence_ratio = float(video_data.get("persistence_ratio", 0.0))
    compression_degradation = bool(video_data.get("compression_degradation", False) or video_data.get("compression_degraded", False))
    uncertainty_reason = video_data.get("uncertainty_reason")
    confidence = float(video_data.get("confidence", 0.50))
    frames_analyzed = int(video_data.get("frames_analyzed", video_data.get("total_frames_sampled", 0)))
    face_frames = int(video_data.get("face_frames", video_data.get("faces_detected", 0)))
    skipped_no_face_frames = int(video_data.get("skipped_no_face_frames", max(0, frames_analyzed - face_frames)))
    valid_prediction_count = int(video_data.get("valid_prediction_count", face_frames))

    # Support standalone V5.2 temporal model evidence
    v5_2_score = v5_2_ev.get("temporal_score") if v5_2_ev else None
    v5_2_thresh = float(v5_2_ev.get("temporal_threshold", VIDEO_TEMPORAL_THRESHOLD)) if v5_2_ev else VIDEO_TEMPORAL_THRESHOLD
    calibrated_temporal_score = calibrate_temporal_score(v5_2_score, v5_2_thresh) if v5_2_score is not None else None
    v5_2_face_frames = int(v5_2_ev.get("valid_face_frames", face_frames)) if v5_2_ev else 0
    v5_2_pred = v5_2_ev.get("prediction") if v5_2_ev else None

    # High-confidence V5.2 criteria: >= 16 valid face frames and temporal_score <= 0.22 or >= 0.40
    is_v5_2_high_conf = (
        v5_2_ev is not None and
        v5_2_face_frames >= 16 and
        v5_2_score is not None and
        (v5_2_score <= 0.22 or v5_2_score >= 0.40)
    )
    is_spatial_volatile = bool(
        video_data.get("is_spatial_volatile") or
        (p1_ev and p1_ev.get("is_spatial_volatile")) or
        (temporal_volatility >= 0.40 and score_std >= 0.35 and (0.15 <= median_score <= 0.35))
    )
    spatial_volatility_decoupled = bool(
        video_data.get("spatial_volatility_decoupled") or
        (p1_ev and p1_ev.get("spatial_volatility_decoupled")) or
        (is_spatial_volatile and v5_2_ev and v5_2_face_frames >= 15 and v5_2_pred in ["Real", "Fake"])
    )

    is_temporal_standalone = (v5_2_ev is not None) and ("video_calibrated_score" not in video_data) and ("median_score" not in video_data)
    if is_temporal_standalone:
        representative_score = calibrated_temporal_score if calibrated_temporal_score is not None else 0.50
        verdict = v5_2_ev.get("verdict", "Uncertain" if v5_2_pred == "Uncertain" else ("Likely Fake" if (v5_2_score is not None and v5_2_score >= v5_2_thresh) else "Real"))
        prediction = v5_2_ev.get("prediction", "Uncertain" if verdict == "Uncertain" else ("Fake" if verdict == "Likely Fake" else "Real"))
        confidence = float(v5_2_ev.get("confidence", 0.50))
        calibration_threshold = v5_2_thresh
    else:
        p1_pred = p1_ev.get("prediction") if p1_ev else None

        if spatial_volatility_decoupled:
            verdict = v5_2_ev.get("verdict", "Likely Fake" if v5_2_pred == "Fake" else "Real")
            prediction = v5_2_pred
            confidence = float(v5_2_ev.get("confidence", 0.65))
            representative_score = calibrated_temporal_score if calibrated_temporal_score is not None else (video_calibrated_score or median_score)
            uncertainty_reason = None
        # Confidence/reliability-aware arbitration if both predictions are present
        elif p1_pred in ["Real", "Fake"] and v5_2_pred in ["Real", "Fake"]:
            if p1_pred == v5_2_pred:
                verdict = "Real" if p1_pred == "Real" else "Likely Fake"
                prediction = p1_pred
                confidence = float(np.clip(max(confidence, v5_2_ev.get("confidence", 0.50)), 0.50, 0.9999))
                if calibrated_temporal_score is not None and video_calibrated_score is not None:
                    representative_score = 0.60 * calibrated_temporal_score + 0.40 * video_calibrated_score
                elif calibrated_temporal_score is not None:
                    representative_score = calibrated_temporal_score
                else:
                    representative_score = video_calibrated_score if video_calibrated_score is not None else median_score
            elif is_v5_2_high_conf:
                # High-confidence V5.2 arbitrates over Phase 1 heuristic
                verdict = v5_2_ev.get("verdict", "Likely Fake" if v5_2_pred == "Fake" else "Real")
                prediction = v5_2_pred
                confidence = float(v5_2_ev.get("confidence", 0.65))
                representative_score = calibrated_temporal_score if calibrated_temporal_score is not None else (video_calibrated_score or median_score)
                uncertainty_reason = None
            else:
                # Disagreement AND V5.2 near boundary (0.22 < score < 0.40) or < 16 face frames
                verdict = "Uncertain"
                prediction = "Uncertain"
                confidence = 0.50
                uncertainty_reason = (
                    "Phase 1 and V5.2 provided sufficiently strong contradictory evidence; "
                    "DeepShield therefore returned Uncertain rather than forcing a binary verdict."
                )
                if calibrated_temporal_score is not None and video_calibrated_score is not None:
                    representative_score = 0.50 * calibrated_temporal_score + 0.50 * video_calibrated_score
                else:
                    representative_score = calibrated_temporal_score or video_calibrated_score or median_score
        else:
            if not verdict:
                if video_calibrated_score is not None:
                    if video_calibrated_score >= calibration_threshold:
                        verdict = "Likely Fake"
                    else:
                        verdict = "Real"
                elif median_score >= 0.25 and persistence_ratio >= 0.30 and temporal_volatility < 0.20:
                    verdict = "Likely Fake"
                elif median_score < 0.08 and p75_score < 0.15 and not compression_degradation:
                    verdict = "Real"
                else:
                    verdict = "Uncertain"

            prediction = "Fake" if verdict == "Likely Fake" else ("Real" if verdict == "Real" else "Uncertain")

            if calibrated_temporal_score is not None and video_calibrated_score is not None:
                representative_score = 0.50 * calibrated_temporal_score + 0.50 * video_calibrated_score
            elif calibrated_temporal_score is not None:
                representative_score = calibrated_temporal_score
            elif video_calibrated_score is not None:
                representative_score = video_calibrated_score
            elif verdict == "Likely Fake":
                representative_score = p75_score
            else:
                representative_score = median_score

    representative_score = float(np.clip(representative_score, 0.0, 1.0))

    # Dynamic reliability assessment
    reliability = 0.90
    findings = []

    if compression_degradation:
        # Soft penalty: low bitrate or texture blur reduces reliability weight, but does not hard-veto verdict
        reliability *= 0.40
        findings.append("Video exhibits compression degradation (sub-HD resolution, low bitrate, or texture blur).")

    if temporal_volatility >= 0.20 or score_std >= 0.25:
        # Volatile score jumping indicates macroblocking or encoding instability
        reliability *= 0.50
        findings.append(f"High temporal score volatility detected (std: {score_std:.2f}, delta: {temporal_volatility:.2f}).")

    if verdict == "Uncertain":
        reliability *= 0.30
        if uncertainty_reason:
            findings.append(f"Video detector flagged uncertainty: {uncertainty_reason}")
        else:
            findings.append("Video temporal analysis yielded ambiguous forensic indicators.")

    # Video Reliability Floor
    # When valid_face_frames >= 15, apply a minimum video evidence reliability of 0.15
    # ONLY as a lower bound after quality/reliability calculation.
    actual_face_frames = max(valid_prediction_count, face_frames, v5_2_face_frames)
    if actual_face_frames >= 15:
        reliability = max(reliability, 0.15)

    if spatial_volatility_decoupled:
        findings.append(
            "Phase 1 spatial score volatility was treated as a reliability factor rather than an automatic final-decision veto because dedicated V5.2 temporal evidence was available."
        )

    if v5_2_ev and v5_2_ev.get("temporal_score") is not None:
        t_thresh = v5_2_ev.get("temporal_threshold", 0.30)
        findings.append(
            f"DeepShield V5.2 temporal model evaluated sequence (score: {v5_2_ev.get('temporal_score'):.4f}, "
            f"threshold: {t_thresh}, prediction: {v5_2_ev.get('prediction')})."
        )

    if frames_analyzed > 0:
        findings.append(
            f"Temporal analysis sampled {frames_analyzed} keyframes ({face_frames} valid face frames, {skipped_no_face_frames} non-face skipped). "
            f"Raw face median: {median_score:.4f}, Calibrated fake score: {representative_score:.4f}."
        )

    return {
        "available": True,
        "verdict": verdict,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "representative_score": round(representative_score, 4),
        "raw_model_score": round(raw_model_score, 4),
        "video_calibrated_score": round(representative_score, 4),
        "video_score_direction": video_score_direction,
        "calibration_threshold": calibration_threshold,
        "calibration_note": calibration_note,
        "median_score": round(median_score, 4),
        "p75_score": round(p75_score, 4),
        "temporal_volatility": round(temporal_volatility, 4),
        "score_std": round(score_std, 4),
        "persistence_ratio": round(persistence_ratio, 4),
        "compression_degradation": compression_degradation,
        "uncertainty_reason": uncertainty_reason,
        "is_spatial_volatile": is_spatial_volatile,
        "spatial_volatility_decoupled": spatial_volatility_decoupled,
        "frames_analyzed": frames_analyzed,
        "total_frames_sampled": frames_analyzed,
        "face_frames": face_frames,
        "skipped_no_face_frames": skipped_no_face_frames,
        "valid_prediction_count": valid_prediction_count,
        "phase1_video_evidence": p1_ev,
        "v5_2_temporal_evidence": v5_2_ev,
        "temporal_evidence": v5_2_ev,
        "temporal_model": v5_2_ev.get("temporal_model") if v5_2_ev else None,
        "temporal_score": v5_2_ev.get("temporal_score") if v5_2_ev else None,
        "temporal_threshold": v5_2_ev.get("temporal_threshold") if v5_2_ev else None,
        "calibrated_temporal_score": round(calibrated_temporal_score, 4) if calibrated_temporal_score is not None else None,
        "temporal_calibrated_score": round(calibrated_temporal_score, 4) if calibrated_temporal_score is not None else None,
        "reliability_weight": round(reliability, 3),
        "findings": findings
    }


def parse_metadata_evidence(meta_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts and normalizes forensic evidence from technical metadata."""
    if not meta_data or not isinstance(meta_data, dict):
        return {
            "available": False,
            "editing_software_detected": False,
            "editing_software_name": None,
            "is_ai_generator_indicated": False,
            "has_camera_exif": False,
            "camera_make": None,
            "camera_model": None,
            "has_exif": False,
            "reliability_weight": 0.0,
            "findings": []
        }

    editing_detected = bool(meta_data.get("editing_software_detected", False))
    software_name = meta_data.get("editing_software_name", meta_data.get("software"))
    ai_generator = bool(meta_data.get("is_ai_generator_indicated", False))
    has_exif = bool(meta_data.get("has_exif", False))

    camera_make = meta_data.get("camera_make")
    camera_model = meta_data.get("camera_model")
    has_camera = bool(camera_make or camera_model)

    # Metadata is strictly supporting evidence only
    reliability = 0.15
    findings = []

    if ai_generator:
        findings.append(f"Metadata matches known AI generative software signature: '{software_name}'.")
        rep_score = 0.70
    elif editing_detected:
        findings.append(f"Metadata indicates image manipulation or post-processing software: '{software_name}'.")
        rep_score = 0.40
    elif has_camera:
        findings.append(f"Verified hardware camera recording tags: {camera_make or ''} {camera_model or ''}".strip())
        rep_score = 0.05
    elif has_exif:
        findings.append("EXIF header present without suspicious software tags.")
        rep_score = 0.15
    else:
        findings.append("No camera or editing EXIF metadata tags located.")
        rep_score = CALIBRATED_THRESHOLD
        reliability = 0.05

    return {
        "available": True,
        "editing_software_detected": editing_detected,
        "editing_software_name": software_name,
        "is_ai_generator_indicated": ai_generator,
        "has_camera_exif": has_camera,
        "camera_make": camera_make,
        "camera_model": camera_model,
        "has_exif": has_exif,
        "representative_score": rep_score,
        "reliability_weight": round(reliability, 3),
        "findings": findings
    }


def fuse_multimodal_evidence(
    image_evidence: Optional[Dict[str, Any]] = None,
    video_evidence: Optional[Dict[str, Any]] = None,
    metadata_evidence: Optional[Dict[str, Any]] = None,
    temporal_evidence: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes Adaptive Multimodal Evidence Fusion across image, video, and metadata.

    Args:
        image_evidence: Output dictionary from ImageDetector.predict or image evidence.
        video_evidence: Output dictionary from VideoDetector.predict or video evidence.
        metadata_evidence: Output dictionary from MetadataForensics.extract_metadata.
        temporal_evidence: Optional dedicated temporal evidence from VideoTemporalDetector.

    Returns:
        Structured dictionary conforming to DeepShield V5 Multimodal Fusion specifications.
    """
    if temporal_evidence:
        if not video_evidence:
            video_evidence = temporal_evidence
        elif isinstance(video_evidence, dict) and "v5_2_temporal_evidence" not in video_evidence:
            video_evidence = dict(video_evidence)
            video_evidence["v5_2_temporal_evidence"] = temporal_evidence

    img = parse_image_evidence(image_evidence)
    vid = parse_video_evidence(video_evidence)
    meta = parse_metadata_evidence(metadata_evidence)

    corroborating_factors: List[str] = []
    conflict_factors: List[str] = []
    conflict_detected = False
    uncertainty_reason: Optional[str] = None

    # -------------------------------------------------------------
    # 1. INSUFFICIENT EVIDENCE HANDLING
    # -------------------------------------------------------------
    if not img["available"] and not vid["available"]:
        if meta["available"]:
            reason = "Insufficient forensic evidence: technical metadata alone cannot verify authenticity without visual or temporal data."
        else:
            reason = "Insufficient forensic evidence: no primary image or video evidence stream provided for analysis."

        explanation = (
            f"{reason} {DISCLAIMER_TEXT}"
        )
        return {
            "overall_prediction": "Uncertain",
            "prediction": "Uncertain",
            "confidence": 0.50,
            "calibrated_confidence": 0.50,
            "image_evidence": img,
            "video_evidence": vid,
            "metadata_evidence": meta,
            "fusion_score": round(CALIBRATED_THRESHOLD, 4),
            "calibrated_score": round(CALIBRATED_THRESHOLD, 4),
            "conflict_detected": False,
            "conflict_factors": [],
            "conflicting_factors": [],
            "corroborating_factors": [],
            "uncertainty_reason": reason,
            "fusion_version": "v5-multimodal",
            "risk_level": "Moderate",
            "boundary_uncertainty": {
                "is_boundary_uncertain": True,
                "uncertainty_window": [round(CALIBRATED_THRESHOLD - UNCERTAINTY_MARGIN, 3), round(CALIBRATED_THRESHOLD + UNCERTAINTY_MARGIN, 3)],
                "threshold": CALIBRATED_THRESHOLD,
                "advisory": reason
            },
            "explanation": explanation
        }

    # -------------------------------------------------------------
    # 2. INTER-MODAL CONFLICT DETECTION
    # -------------------------------------------------------------
    # Case A1: Phase 1 Video vs V5.2 Temporal Video Coherence Check
    p1_ev = vid.get("phase1_video_evidence")
    v5_2_ev = vid.get("v5_2_temporal_evidence") or vid.get("temporal_evidence")
    if p1_ev and v5_2_ev:
        p1_pred = p1_ev.get("prediction")
        v5_2_pred = v5_2_ev.get("prediction")
        v5_2_score = v5_2_ev.get("temporal_score")
        v5_2_face_frames = int(v5_2_ev.get("valid_face_frames", vid.get("face_frames", 0)))

        is_v5_2_high_conf = (
            v5_2_face_frames >= 16 and
            v5_2_score is not None and
            (v5_2_score <= 0.22 or v5_2_score >= 0.40)
        )

        spatial_volatility_decoupled = vid.get("spatial_volatility_decoupled", False)

        if spatial_volatility_decoupled:
            corroborating_factors.append(
                "Phase 1 spatial score volatility was treated as a reliability factor rather than an automatic final-decision veto because dedicated V5.2 temporal evidence was available."
            )
        elif p1_pred in ["Real", "Fake"] and v5_2_pred in ["Real", "Fake"]:
            if p1_pred == v5_2_pred:
                corroborating_factors.append(
                    f"Temporal model (V5.2) and spatial frame analysis (Phase 1) concordantly verify media as {p1_pred} "
                    f"(temporal score: {v5_2_score:.4f}, threshold: {v5_2_ev.get('temporal_threshold', VIDEO_TEMPORAL_THRESHOLD)})."
                )
            elif is_v5_2_high_conf:
                # High-confidence V5.2 arbitrates over Phase 1 heuristic
                corroborating_factors.append(
                    "V5.2 temporal evidence was given higher reliability than the legacy Phase 1 spatial heuristic "
                    "because sufficient face frames were available and the temporal score was outside the uncertainty boundary."
                )
                conflict_factors.append(
                    f"Secondary spatial heuristic ({p1_pred}, score: {p1_ev.get('video_calibrated_score', 0.0):.4f}) "
                    f"disagreed with primary V5.2 temporal model ({v5_2_pred}, score: {v5_2_score:.4f})."
                )
            else:
                # Disagreement AND V5.2 is near boundary or < 16 face frames
                conflict_detected = True
                conflict_factors.append(
                    "Temporal vs Spatial Video Conflict: Phase 1 and V5.2 provided sufficiently strong contradictory evidence; "
                    "DeepShield therefore returned Uncertain rather than forcing a binary verdict."
                )

    # Case A2: Both Image and Video available
    if img["available"] and vid["available"]:
        img_is_fake = img["raw_score"] >= CALIBRATED_THRESHOLD
        vid_verdict = vid["verdict"]

        img_decisive = not img.get("in_boundary_uncertainty", False)
        vid_decisive = (vid_verdict in ["Real", "Likely Fake", "Fake"]) and (not vid.get("compression_degradation", False))

        # True conflict requires genuine strong contradictions where neither is near boundary
        if img_is_fake and vid_verdict == "Real":
            if img_decisive and vid_decisive:
                conflict_detected = True
                conflict_factors.append(
                    f"Strong inter-modal contradiction: Decisive image analysis indicates manipulated content "
                    f"(model score: {img['raw_score']:.4f}, threshold: {CALIBRATED_THRESHOLD}), "
                    f"while decisive temporal video analysis indicates authentic content (verdict: Real, representative score: {vid['representative_score']:.4f})."
                )
            else:
                conflict_factors.append(
                    f"Potential contradiction between image (score: {img['raw_score']:.4f}) and video (verdict: Real), "
                    "but one or more streams are near decision boundary or degraded."
                )
        elif (not img_is_fake) and (vid_verdict in ["Likely Fake", "Fake"]):
            if img_decisive and vid_decisive:
                conflict_detected = True
                conflict_factors.append(
                    f"Strong inter-modal contradiction: Decisive image analysis indicates authentic content "
                    f"(model score: {img['raw_score']:.4f}, threshold: {CALIBRATED_THRESHOLD}), "
                    f"while decisive temporal video analysis indicates manipulated content (verdict: {vid_verdict}, representative score: {vid['representative_score']:.4f})."
                )
            else:
                conflict_factors.append(
                    f"Potential contradiction between image (score: {img['raw_score']:.4f}) and video (verdict: {vid_verdict}), "
                    "but one or more streams are near decision boundary or degraded."
                )
        elif img_is_fake and vid["compression_degradation"]:
            conflict_factors.append(
                "Image analysis indicates manipulation, while video evidence is unreliable because of severe compression."
            )
        elif (not img_is_fake) and vid["compression_degradation"]:
            conflict_factors.append(
                "Image analysis indicates authentic content, while video evidence is degraded by compression and cannot provide corroboration."
            )
        else:
            # Consistent directional findings
            if img_is_fake and (vid_verdict in ["Likely Fake", "Fake"]):
                corroborating_factors.append(
                    f"Image analysis and temporal video analysis both detect manipulation-related patterns (image score: {img['raw_score']:.4f}, video representative score: {vid['representative_score']:.4f})."
                )
            elif (not img_is_fake) and vid_verdict == "Real":
                corroborating_factors.append(
                    f"Image analysis and temporal video analysis consistently show no strong manipulation patterns (image score: {img['raw_score']:.4f}, video representative score: {vid['representative_score']:.4f})."
                )

    # Case B: Metadata Coherence Check (Supporting)
    if meta["available"]:
        primary_is_fake = False
        if img["available"]:
            primary_is_fake = img["raw_score"] >= CALIBRATED_THRESHOLD
        elif vid["available"]:
            primary_is_fake = vid["verdict"] == "Likely Fake"

        if meta["editing_software_detected"] or meta["is_ai_generator_indicated"]:
            sw = meta["editing_software_name"] or "Unknown"
            if primary_is_fake:
                corroborating_factors.append(
                    f"Metadata identifies editing or generation software signature ('{sw}'), corroborating visual manipulation flags."
                )
            else:
                conflict_factors.append(
                    f"Primary visual model classified media as Real, but metadata indicates software post-processing ('{sw}')."
                )

        if meta["has_camera_exif"]:
            cam = f"{meta['camera_make'] or ''} {meta['camera_model'] or ''}".strip()
            if not primary_is_fake:
                corroborating_factors.append(
                    f"Presence of verified camera hardware metadata ({cam}) aligns with authentic media."
                )
            else:
                conflict_factors.append(
                    f"Visual model detected manipulation patterns, but camera hardware metadata ({cam}) is present. Media may be partially composite or metadata preserved."
                )

    # -------------------------------------------------------------
    # 3. RELIABILITY-WEIGHTED FUSION SCORING
    # -------------------------------------------------------------
    weights = []
    scores = []

    if img["available"]:
        weights.append(img["reliability_weight"])
        scores.append(img["raw_score"])

    if vid["available"]:
        weights.append(vid["reliability_weight"])
        scores.append(vid["representative_score"])

    if meta["available"] and meta["reliability_weight"] > 0:
        weights.append(meta["reliability_weight"])
        scores.append(meta["representative_score"])

    total_weight = sum(weights)
    if total_weight > 0:
        fused_score = float(sum(w * s for w, s in zip(weights, scores)) / total_weight)
    else:
        fused_score = CALIBRATED_THRESHOLD

    fused_score = float(np.clip(fused_score, 0.0, 1.0))

    # -------------------------------------------------------------
    # 4. 3-STATE VERDICT RESOLUTION ENGINE
    # -------------------------------------------------------------
    lower_bound = max(0.0, CALIBRATED_THRESHOLD - UNCERTAINTY_MARGIN)
    upper_bound = min(1.0, CALIBRATED_THRESHOLD + UNCERTAINTY_MARGIN)
    is_boundary_uncertain = lower_bound <= fused_score <= upper_bound

    # Condition 1: Significant Inter-Modal Conflict
    if conflict_detected:
        overall_pred = "Uncertain"
        confidence = 0.50
        uncertainty_reason = "Significant inter-modal conflict detected between image and video evidence streams."
        explanation = (
            f"Analysis revealed conflicting evidence across modalities: {'; '.join(conflict_factors)}. "
            f"Because primary evidence streams disagree, the system reports Uncertain. {DISCLAIMER_TEXT}"
        )

    # Condition 2: Video-only and Video is Uncertain
    elif (not img["available"]) and vid["available"] and (vid["verdict"] == "Uncertain"):
        overall_pred = "Uncertain"
        confidence = 0.50
        uncertainty_reason = vid["uncertainty_reason"] or "Video evidence yielded ambiguous forensic indicators."
        explanation = (
            f"Video forensic inspection yielded uncertain results: {uncertainty_reason}. {DISCLAIMER_TEXT}"
        )

    # Condition 3: Both Image & Video provided, but Video is degraded AND Image is near boundary
    elif img["available"] and vid["available"] and vid["compression_degradation"] and img["in_boundary_uncertainty"]:
        overall_pred = "Uncertain"
        confidence = 0.50
        uncertainty_reason = "Video is degraded by compression and image score is near decision boundary."
        explanation = (
            f"Image analysis score ({img['raw_score']:.4f}) is near the decision boundary, while video evidence is unreliable due to severe compression. "
            f"The system therefore reports Uncertain. {DISCLAIMER_TEXT}"
        )

    # Condition 4: Clean decisive classification
    elif fused_score >= CALIBRATED_THRESHOLD:
        overall_pred = "Likely Fake"
        confidence = vid["confidence"] if (not img["available"] and vid["available"]) else _compute_calibrated_confidence(fused_score)
        uncertainty_reason = None
        reasons = []
        if img["available"] and img["raw_score"] >= CALIBRATED_THRESHOLD:
            reasons.append(f"image model detected manipulation patterns (score: {img['raw_score']:.4f})")
        if vid["available"] and vid["verdict"] == "Likely Fake":
            reasons.append(f"temporal video analysis detected manipulation artifacts (calibrated fake score: {vid['video_calibrated_score']:.4f}, raw face median: {vid['median_score']:.4f})")
        if not reasons:
            reasons.append("fused multimodal score exceeded the calibrated threshold")
        explanation = (
            f"The model detected visual patterns associated with manipulated content ({', '.join(reasons)}). {DISCLAIMER_TEXT}"
        )

    else:
        # Fused score < CALIBRATED_THRESHOLD
        overall_pred = "Real"
        confidence = vid["confidence"] if (not img["available"] and vid["available"]) else _compute_calibrated_confidence(fused_score)
        uncertainty_reason = None
        if vid["available"] and not img["available"]:
            explanation = (
                f"Temporal video analysis did not detect strong manipulation patterns on valid face frames "
                f"(raw face median: {vid['median_score']:.4f}, calibrated score: {vid['video_calibrated_score']:.4f}, threshold: {CALIBRATED_THRESHOLD}). {DISCLAIMER_TEXT}"
            )
        else:
            explanation = (
                f"The model did not detect strong manipulation-related patterns in the analyzed content "
                f"(fused score: {fused_score:.4f}, threshold: {CALIBRATED_THRESHOLD}). {DISCLAIMER_TEXT}"
            )

    # -------------------------------------------------------------
    # 5. RISK LEVEL STRATIFICATION
    # -------------------------------------------------------------
    if overall_pred in ["Likely Fake", "Fake"]:
        if fused_score >= 0.50:
            risk_level = "Critical"
        else:
            risk_level = "Elevated"
    elif overall_pred == "Uncertain":
        if is_boundary_uncertain or fused_score >= CALIBRATED_THRESHOLD:
            risk_level = "Elevated"
        else:
            risk_level = "Moderate"
    else:  # Real
        if fused_score <= 0.08:
            risk_level = "Low"
        else:
            risk_level = "Moderate"

    # Boundary uncertainty advisory
    if is_boundary_uncertain:
        boundary_advisory = (
            f"Notice: Prediction score ({fused_score:.4f}) falls within the boundary uncertainty window "
            f"[{lower_bound:.2f} - {upper_bound:.2f}] around calibrated threshold {CALIBRATED_THRESHOLD}. "
            "Manual human review or secondary artifact verification is recommended."
        )
    else:
        boundary_advisory = (
            f"Confidence level is robust. Score ({fused_score:.4f}) is distinctly separated from the decision boundary ({CALIBRATED_THRESHOLD})."
        )

    return {
        "overall_prediction": overall_pred,
        "prediction": overall_pred,
        "confidence": round(confidence, 4),
        "calibrated_confidence": round(confidence, 4),
        "image_evidence": img,
        "video_evidence": vid,
        "metadata_evidence": meta,
        "fusion_score": round(fused_score, 4),
        "calibrated_score": round(fused_score, 4),
        "conflict_detected": conflict_detected,
        "conflict_factors": conflict_factors,
        "conflicting_factors": conflict_factors,  # Backward compatibility alias
        "corroborating_factors": corroborating_factors,
        "uncertainty_reason": uncertainty_reason,
        "fusion_version": "v5-multimodal",
        "risk_level": risk_level,
        "boundary_uncertainty": {
            "is_boundary_uncertain": is_boundary_uncertain,
            "uncertainty_window": [round(lower_bound, 3), round(upper_bound, 3)],
            "threshold": CALIBRATED_THRESHOLD,
            "advisory": boundary_advisory
        },
        "explanation": explanation
    }


class MultimodalFusionEngine:
    """
    Consolidated Multimodal Fusion Engine interface for DeepShield V5.
    Provides backward compatibility with legacy API endpoints and callers.
    """

    @staticmethod
    def evaluate_fusion(
        image_prediction: Optional[Dict[str, Any]] = None,
        metadata_analysis: Optional[Dict[str, Any]] = None,
        video_prediction: Optional[Dict[str, Any]] = None,
        temporal_prediction: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluates multimodal evidence sources and produces an adaptive fused assessment.
        Compatible with both legacy image-only calls and new multimodal calls.
        """
        img_input = image_prediction or kwargs.get("image_evidence")
        vid_input = video_prediction or kwargs.get("video_evidence")
        meta_input = metadata_analysis or kwargs.get("metadata_evidence")
        temp_input = temporal_prediction or kwargs.get("temporal_evidence")

        # Handle case where a single dictionary contains both image and metadata or video
        if isinstance(img_input, dict):
            # Check if img_input is actually video output
            if img_input.get("media_type") == "video" and not vid_input:
                vid_input = img_input
                img_input = None
            else:
                if not meta_input and "metadata" in img_input:
                    meta_input = img_input.get("metadata")
                if not vid_input and "video_evidence" in img_input:
                    vid_input = img_input.get("video_evidence")

        return fuse_multimodal_evidence(
            image_evidence=img_input,
            video_evidence=vid_input,
            metadata_evidence=meta_input,
            temporal_evidence=temp_input
        )

    @staticmethod
    def fuse(
        image_evidence: Optional[Dict[str, Any]] = None,
        video_evidence: Optional[Dict[str, Any]] = None,
        metadata_evidence: Optional[Dict[str, Any]] = None,
        temporal_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Direct fusion method alias."""
        return fuse_multimodal_evidence(
            image_evidence=image_evidence,
            video_evidence=video_evidence,
            metadata_evidence=metadata_evidence,
            temporal_evidence=temporal_evidence
        )
