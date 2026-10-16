"""
DeepShield V5 - Structured Forensic Report Generator
====================================================
Generates academic and forensic grade authenticity audit reports synthesizing:
- Neural Image Analysis (EfficientNetB2 V5 spatial features & Grad-CAM)
- Temporal Video Forensics (Multi-frame consistency, volatility, & persistence)
- Video Quality & Compression Forensics (Bitrate, resolution, blur variance)
- Technical Metadata Audit (EXIF integrity, editing & generative AI signatures)
- Adaptive Multimodal Evidence Fusion & Uncertainty Analysis
"""

import datetime
from typing import Dict, Any, List, Optional

from backend.config import CALIBRATED_THRESHOLD
from backend.fusion import MultimodalFusionEngine


class ForensicReportGenerator:
    """
    Generates academic and forensic grade authenticity audit reports.
    """

    @staticmethod
    def generate_report(
        prediction_result: Dict[str, Any],
        report_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compiles the complete forensic audit report from multimodal evidence.

        Args:
            prediction_result: Prediction dictionary from ImageDetector, VideoDetector,
                               or MultimodalFusionEngine.
            report_id: Unique identifier for tracking.

        Returns:
            Structured multimodal forensic report dictionary conforming to DeepShield V5 specifications.
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rid = report_id or f"DSR-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"

        # 1. Obtain or evaluate Multimodal Fusion
        if "fusion" in prediction_result and isinstance(prediction_result["fusion"], dict):
            fusion = prediction_result["fusion"]
        elif "fusion_version" in prediction_result:
            fusion = prediction_result
        else:
            fusion = MultimodalFusionEngine.evaluate_fusion(prediction_result)

        # 2. Extract Top-level Verdict & Confidence
        verdict = fusion.get("overall_prediction", fusion.get("prediction", prediction_result.get("prediction", "Real")))
        if prediction_result.get("prediction") == "Uncertain" or prediction_result.get("verdict") == "Uncertain" or fusion.get("overall_prediction") == "Uncertain":
            verdict = "Uncertain"
        confidence = float(fusion.get("confidence", fusion.get("calibrated_confidence", prediction_result.get("confidence", 0.50))))
        fusion_score = float(fusion.get("fusion_score", fusion.get("calibrated_score", prediction_result.get("score", 0.0))))
        risk_level = "Moderate" if verdict == "Uncertain" else fusion.get("risk_level", "Moderate")

        # Qualified verdict statement adhering to DeepShield forensic guidelines
        if verdict == "Real":
            verdict_statement = "The available evidence is more consistent with authentic media."
        elif verdict in ["Likely Fake", "Fake"]:
            verdict_statement = "The available evidence contains patterns associated with manipulated or synthetic media."
        else:
            verdict_statement = (
                "The available evidence is conflicting, insufficient, degraded, or close to the decision boundary, "
                "so DeepShield does not force a binary decision."
            )
        disclaimer_text = "This is an AI-based assessment, not absolute proof of authenticity."

        # Extract underlying evidence dictionaries from fusion
        img_ev = fusion.get("image_evidence", {})
        vid_ev = fusion.get("video_evidence", {})
        meta_ev = fusion.get("metadata_evidence", {})

        # Fallback extraction from raw prediction_result if fusion was partial
        if not img_ev.get("available") and ("score" in prediction_result or "face_detected" in prediction_result):
            img_ev = {
                "available": True,
                "raw_score": prediction_result.get("score", 0.0),
                "prediction": prediction_result.get("prediction", "Real"),
                "confidence": prediction_result.get("confidence", 0.5),
                "face_detected": prediction_result.get("face_detected", False),
                "face_count": prediction_result.get("face_count", 1 if prediction_result.get("face_detected") else 0),
                "reliability_weight": 1.0,
                "findings": []
            }

        if not vid_ev.get("available") and ("frames_analyzed" in prediction_result or prediction_result.get("media_type") == "video"):
            vid_ev = {
                "available": True,
                "verdict": prediction_result.get("verdict", "Uncertain"),
                "prediction": prediction_result.get("prediction", "Uncertain"),
                "confidence": prediction_result.get("confidence", 0.50),
                "median_score": prediction_result.get("median_score", 0.0),
                "p75_score": prediction_result.get("p75_score", 0.0),
                "temporal_volatility": prediction_result.get("temporal_volatility", 0.0),
                "score_std": prediction_result.get("score_std", 0.0),
                "persistence_ratio": prediction_result.get("persistence_ratio", 0.0),
                "compression_degradation": bool(prediction_result.get("compression_degradation", prediction_result.get("compression_degraded", False))),
                "uncertainty_reason": prediction_result.get("uncertainty_reason"),
                "frames_analyzed": prediction_result.get("frames_analyzed", 0),
                "faces_detected": prediction_result.get("faces_detected", 0),
                "resolution": prediction_result.get("resolution", "Unknown"),
                "fps": prediction_result.get("fps", 0.0),
                "duration": prediction_result.get("duration", 0.0),
                "reliability_weight": 0.90,
                "findings": []
            }

        metadata_raw = prediction_result.get("metadata", prediction_result.get("video_metadata", {}))
        gradcam_raw = prediction_result.get("gradcam", {})
        faces_raw = prediction_result.get("faces", [])

        # Extract filename and format
        media_filename = metadata_raw.get("filename", prediction_result.get("filename", "analyzed_media"))
        media_format = metadata_raw.get("format", prediction_result.get("format", "Unknown"))
        media_type = prediction_result.get("media_type", "video" if vid_ev.get("available") else "image")

        # -------------------------------------------------------------
        # SECTION C: IMAGE EVIDENCE
        # -------------------------------------------------------------
        face_conf_str = "None"
        if faces_raw and len(faces_raw) > 0 and "score" in faces_raw[0]:
            face_conf_str = f"{faces_raw[0]['score'] * 100:.1f}%"
        elif img_ev.get("face_detected"):
            face_conf_str = "Detected"

        vis_paths = prediction_result.get("visual_evidence", {})
        heatmap_url = vis_paths.get("heatmap_url", gradcam_raw.get("heatmap_url"))
        overlay_url = vis_paths.get("overlay_url", gradcam_raw.get("overlay_url"))
        face_crop_url = vis_paths.get("face_crop_url")
        orig_url = vis_paths.get("original_image_url")

        sec_image_evidence = {
            "evidence_available": img_ev.get("available", False),
            "face_detected": img_ev.get("face_detected", False),
            "face_count": img_ev.get("face_count", len(faces_raw)),
            "face_confidence": face_conf_str,
            "model_score": img_ev.get("raw_score", prediction_result.get("score")),
            "image_prediction": img_ev.get("prediction", prediction_result.get("prediction")),
            "calibrated_confidence": img_ev.get("confidence", prediction_result.get("confidence")),
            "reliability_weight": img_ev.get("reliability_weight", 0.0),
            "gradcam": {
                "available": gradcam_raw.get("available", False),
                "target_layer": prediction_result.get("technical_details", {}).get("gradcam_target_layer", "efficientnetb2/top_conv"),
                "heatmap_url": heatmap_url,
                "overlay_url": overlay_url,
                "peak_intensity": gradcam_raw.get("peak_intensity", 0.0),
                "mean_intensity": gradcam_raw.get("mean_intensity", 0.0),
                "salient_ratio": gradcam_raw.get("salient_ratio", 0.0),
                "explanation": "The highlighted regions show areas that influenced the model's decision."
            },
            "visualization_paths": {
                "original_url": orig_url,
                "face_crop_url": face_crop_url,
                "heatmap_url": heatmap_url,
                "overlay_url": overlay_url
            },
            "findings": img_ev.get("findings", [])
        }

        # -------------------------------------------------------------
        # SECTION D: VIDEO EVIDENCE (Dual Evidence: V5.2 Temporal & Phase 1)
        # -------------------------------------------------------------
        suspicious_frames = prediction_result.get("suspicious_frames", [])
        v5_2_raw = (
            prediction_result.get("v5_2_temporal_evidence")
            or vid_ev.get("v5_2_temporal_evidence")
            or prediction_result.get("temporal_evidence")
            or vid_ev.get("temporal_evidence")
        )
        p1_raw = (
            prediction_result.get("phase1_video_evidence")
            or vid_ev.get("phase1_video_evidence")
        )

        conflict_status = "Standard video inspection"
        if v5_2_raw and p1_raw:
            p1_pred = p1_raw.get("prediction")
            v5_2_pred = v5_2_raw.get("prediction")
            v5_2_score = v5_2_raw.get("temporal_score")
            v5_2_frames = int(v5_2_raw.get("valid_face_frames", vid_ev.get("face_frames", 0)))
            is_v5_2_high_conf = (
                v5_2_frames >= 16 and
                v5_2_score is not None and
                (v5_2_score <= 0.22 or v5_2_score >= 0.40)
            )
            if p1_pred in ["Real", "Fake"] and v5_2_pred in ["Real", "Fake"]:
                if p1_pred == v5_2_pred:
                    conflict_status = f"Concordant with Phase 1 frame analysis ({p1_pred})"
                elif is_v5_2_high_conf:
                    conflict_status = (
                        "V5.2 temporal evidence was given higher reliability than the legacy "
                        "Phase 1 spatial heuristic because sufficient face frames were available "
                        "and the temporal score was outside the uncertainty boundary."
                    )
                else:
                    conflict_status = (
                        "Phase 1 and V5.2 provided sufficiently strong contradictory evidence; "
                        "DeepShield therefore returned Uncertain rather than forcing a binary verdict."
                    )
        elif v5_2_raw:
            conflict_status = "DeepShield V5.2 Temporal Model active"
        elif p1_raw:
            conflict_status = "Phase 1 spatial frame analysis fallback"

        temporal_summary = {
            "available": bool(v5_2_raw),
            "temporal_model": v5_2_raw.get("temporal_model", "DeepShield V5.2") if v5_2_raw else None,
            "model_file": v5_2_raw.get("model_file") if v5_2_raw else None,
            "sequence_length": v5_2_raw.get("sequence_length", 16) if v5_2_raw else 16,
            "temporal_score": v5_2_raw.get("temporal_score") if v5_2_raw else None,
            "temporal_threshold": v5_2_raw.get("temporal_threshold", 0.30) if v5_2_raw else 0.30,
            "valid_face_frames": v5_2_raw.get("valid_face_frames") if v5_2_raw else vid_ev.get("face_frames", 0),
            "total_frames": v5_2_raw.get("total_frames") if v5_2_raw else vid_ev.get("frames_analyzed", 0),
            "prediction": v5_2_raw.get("prediction") if v5_2_raw else None,
            "confidence": v5_2_raw.get("confidence") if v5_2_raw else None,
            "conflict_status": conflict_status
        }

        sec_video_evidence = {
            "evidence_available": vid_ev.get("available", False),
            "verdict": vid_ev.get("verdict", "Uncertain"),
            "prediction": vid_ev.get("prediction", "Uncertain"),
            "v5_2_temporal_evidence": temporal_summary,
            "temporal_model": temporal_summary["temporal_model"],
            "temporal_score": temporal_summary["temporal_score"],
            "temporal_threshold": temporal_summary["temporal_threshold"],
            "temporal_prediction": temporal_summary["prediction"],
            "conflict_status": conflict_status,
            "frames_analyzed": vid_ev.get("frames_analyzed", prediction_result.get("frames_analyzed", 0)),
            "total_frames_sampled": vid_ev.get("total_frames_sampled", prediction_result.get("total_frames_sampled", vid_ev.get("frames_analyzed", 0))),
            "face_frames": vid_ev.get("face_frames", prediction_result.get("face_frames", vid_ev.get("faces_detected", 0))),
            "faces_detected": vid_ev.get("faces_detected", prediction_result.get("faces_detected", 0)),
            "skipped_no_face_frames": vid_ev.get("skipped_no_face_frames", prediction_result.get("skipped_no_face_frames", 0)),
            "valid_prediction_count": vid_ev.get("valid_prediction_count", prediction_result.get("valid_prediction_count", 0)),
            "raw_model_score": vid_ev.get("raw_model_score", prediction_result.get("raw_model_score", vid_ev.get("median_score"))),
            "video_calibrated_score": vid_ev.get("video_calibrated_score", prediction_result.get("video_calibrated_score")),
            "video_score_direction": vid_ev.get("video_score_direction", prediction_result.get("video_score_direction")),
            "calibration_threshold": vid_ev.get("calibration_threshold", prediction_result.get("calibration_threshold", CALIBRATED_THRESHOLD)),
            "calibration_note": vid_ev.get("calibration_note", prediction_result.get("calibration_note")),
            "median_score": vid_ev.get("median_score"),
            "p75_score": vid_ev.get("p75_score"),
            "temporal_volatility": vid_ev.get("temporal_volatility"),
            "score_std": vid_ev.get("score_std"),
            "persistence_ratio": vid_ev.get("persistence_ratio"),
            "fake_frame_ratio": prediction_result.get("fake_frame_ratio", 0.0),
            "compression_degradation": bool(vid_ev.get("compression_degradation", False) or prediction_result.get("compression_degraded", False) or prediction_result.get("compression_degradation", False)),
            "resolution": prediction_result.get("resolution") or metadata_raw.get("resolution") or vid_ev.get("resolution", "Unknown"),
            "fps": prediction_result.get("fps") or metadata_raw.get("fps") or vid_ev.get("fps", 0.0),
            "duration": prediction_result.get("duration") or metadata_raw.get("duration") or vid_ev.get("duration", 0.0),
            "uncertainty_reason": vid_ev.get("uncertainty_reason") or prediction_result.get("uncertainty_reason"),
            "reliability_weight": vid_ev.get("reliability_weight", 0.0),
            "suspicious_keyframes": suspicious_frames,
            "keyframe_gradcam": prediction_result.get("gradcam", {}) if vid_ev.get("available") else {},
            "findings": vid_ev.get("findings", [])
        }

        # -------------------------------------------------------------
        # SECTION E: METADATA EVIDENCE
        # -------------------------------------------------------------
        forensic_notes = list(metadata_raw.get("forensic_notes", []))
        if not metadata_raw.get("has_exif", False):
            if not any("stripped" in n.lower() or "missing" in n.lower() for n in forensic_notes):
                forensic_notes.append(
                    "Note: Absence of EXIF metadata is standard for web and social media uploads and does not constitute proof of manipulation."
                )

        sec_metadata_evidence = {
            "evidence_available": meta_ev.get("available", False) or bool(metadata_raw),
            "file_format": media_format,
            "file_size": metadata_raw.get("file_size_human", "0 KB"),
            "file_size_bytes": metadata_raw.get("file_size", 0),
            "dimensions": metadata_raw.get("dimensions", metadata_raw.get("resolution", "Unknown")),
            "exif_present": metadata_raw.get("has_exif", False),
            "camera_hardware": {
                "make": metadata_raw.get("camera_make") or meta_ev.get("camera_make"),
                "model": metadata_raw.get("camera_model") or meta_ev.get("camera_model")
            },
            "editing_software": {
                "detected": metadata_raw.get("editing_software_detected", False) or meta_ev.get("editing_software_detected", False),
                "name": metadata_raw.get("editing_software_name") or metadata_raw.get("software") or meta_ev.get("editing_software_name"),
                "ai_generator_signature": metadata_raw.get("is_ai_generator_indicated", False) or meta_ev.get("is_ai_generator_indicated", False)
            },
            "forensic_notes": forensic_notes,
            "integrity_note": "Metadata is treated as supporting context only and cannot alone prove or disprove digital manipulation."
        }

        # -------------------------------------------------------------
        # SECTION F: MULTIMODAL FUSION
        # -------------------------------------------------------------
        modalities_available = []
        if sec_image_evidence["evidence_available"]:
            modalities_available.append("image")
        if sec_video_evidence["evidence_available"]:
            modalities_available.append("video")
        if sec_metadata_evidence["evidence_available"]:
            modalities_available.append("metadata")

        adaptive_reliability = {
            "image_weight": sec_image_evidence["reliability_weight"],
            "video_weight": sec_video_evidence["reliability_weight"],
            "metadata_weight": meta_ev.get("reliability_weight", 0.15 if sec_metadata_evidence["evidence_available"] else 0.0),
            "video_degraded_downweighted": sec_video_evidence["compression_degradation"],
            "volatility_penalty_applied": bool(sec_video_evidence["temporal_volatility"] and sec_video_evidence["temporal_volatility"] >= 0.20)
        }

        conflict_detected = bool(fusion.get("conflict_detected", False))
        conflict_factors = list(fusion.get("conflict_factors", fusion.get("conflicting_factors", [])))
        corroborating_factors = list(fusion.get("corroborating_factors", []))

        sec_multimodal_fusion = {
            "fusion_version": fusion.get("fusion_version", "v5-multimodal"),
            "fusion_score": fusion_score,
            "calibrated_threshold": CALIBRATED_THRESHOLD,
            "decision_threshold": CALIBRATED_THRESHOLD,
            "modalities_available": modalities_available,
            "reliability_weights": adaptive_reliability,
            "adaptive_reliability": adaptive_reliability,
            "conflict_detected": conflict_detected,
            "conflict_factors": conflict_factors,
            "corroborating_factors": corroborating_factors
        }

        # -------------------------------------------------------------
        # SECTION G: UNCERTAINTY ANALYSIS
        # -------------------------------------------------------------
        is_uncertain = verdict == "Uncertain"
        boundary_info = fusion.get("boundary_uncertainty", {})
        is_boundary_uncertain = boundary_info.get("is_boundary_uncertain", False)
        uncertainty_window = boundary_info.get("uncertainty_window", [0.16, 0.26])

        uncertainty_reasons = []
        if is_uncertain:
            if fusion.get("uncertainty_reason"):
                uncertainty_reasons.append(fusion.get("uncertainty_reason"))
            elif conflict_detected:
                uncertainty_reasons.append("Significant inter-modal conflict detected between available evidence sources.")
            elif sec_video_evidence["compression_degradation"]:
                uncertainty_reasons.append("Video quality degradation (sub-HD resolution, low bitrate, or texture blur) prevents conclusive verification.")
            elif is_boundary_uncertain:
                uncertainty_reasons.append(f"Model score ({fusion_score:.4f}) lies within decision boundary uncertainty window [{uncertainty_window[0]}, {uncertainty_window[1]}].")
            elif not modalities_available:
                uncertainty_reasons.append("Insufficient forensic evidence: no primary media stream available.")
            else:
                uncertainty_reasons.append("Ambiguous forensic indicators across analyzed modalities.")

        sec_uncertainty = {
            "is_uncertain": is_uncertain,
            "uncertainty_reason": "; ".join(uncertainty_reasons) if uncertainty_reasons else None,
            "is_within_boundary_window": is_boundary_uncertain,
            "boundary_window": uncertainty_window,
            "calibrated_threshold": CALIBRATED_THRESHOLD,
            "score_distance_from_threshold": round(abs(fusion_score - CALIBRATED_THRESHOLD), 4),
            "advisory": (
                "; ".join(uncertainty_reasons)
                if is_uncertain
                else f"Confidence level is robust. Score ({fusion_score:.4f}) is distinctly separated from threshold ({CALIBRATED_THRESHOLD})."
            )
        }

        # -------------------------------------------------------------
        # SECTION I: TECHNICAL DETAILS
        # -------------------------------------------------------------
        sec_technical_details = {
            "platform": "DeepShield Multimodal Verification Platform",
            "model_name": "DeepShield V5 Clean Best",
            "backbone_architecture": "EfficientNetB2",
            "input_resolution": "224 x 224 x 3",
            "face_detector": "OpenCV YuNet (1:1 Square Crop Geometry)",
            "calibrated_threshold": CALIBRATED_THRESHOLD,
            "fusion_version": fusion.get("fusion_version", "v5-multimodal"),
            "processing_status": "completed",
            "processing_time_seconds": prediction_result.get("technical_details", {}).get("processing_time_seconds", 0.0)
        }

        # -------------------------------------------------------------
        # SECTION H: QUALIFIED EXPLANATIONS
        # -------------------------------------------------------------
        explanation_statements = []
        explanation_statements.append(verdict_statement)

        if sec_image_evidence["evidence_available"]:
            if sec_image_evidence["face_detected"]:
                explanation_statements.append(
                    f"YuNet localized a face analyzed at model score {sec_image_evidence['model_score']:.4f}."
                )
            else:
                explanation_statements.append(
                    "No localized face was detected; neural analysis executed on full image context."
                )

        if sec_image_evidence["gradcam"]["available"]:
            explanation_statements.append(
                "The highlighted regions in the Grad-CAM heatmap show areas that influenced the model's decision."
            )

        if sec_video_evidence["evidence_available"]:
            if sec_video_evidence["compression_degradation"]:
                explanation_statements.append(
                    "Video evidence exhibits compression degradation and was down-weighted during evidence synthesis."
                )
            elif sec_video_evidence["verdict"] == "Likely Fake":
                explanation_statements.append(
                    f"Temporal analysis detected persistent manipulation patterns across keyframes (P75 score: {sec_video_evidence['p75_score']:.4f})."
                )

        if sec_metadata_evidence["editing_software"]["detected"]:
            sw_name = sec_metadata_evidence["editing_software"]["name"]
            explanation_statements.append(
                f"Metadata indicates editing or processing software signature ('{sw_name}')."
            )

        explanation_statements.append(disclaimer_text)
        summary_explanation_text = " ".join(explanation_statements[:5])

        sec_qualified_explanations = {
            "why_result": verdict_statement,
            "visual_evidence": (
                "The highlighted regions show areas that influenced the model's decision."
                if sec_image_evidence["gradcam"]["available"]
                else (
                    "Temporal and spatial facial patterns were evaluated across keyframes."
                    if sec_video_evidence["evidence_available"]
                    else "Neural facial visual features were analyzed."
                )
            ),
            "metadata_evidence": (
                f"Metadata indicates editing or processing software ('{sec_metadata_evidence['editing_software']['name']}')."
                if sec_metadata_evidence["editing_software"]["detected"]
                else "Metadata does not contain known manipulation software signatures."
            ),
            "technical_details": f"Analyzed using {sec_technical_details['model_name']} ({sec_technical_details['backbone_architecture']}) with calibrated threshold {CALIBRATED_THRESHOLD}.",
            "responsible_ai_statement": disclaimer_text,
            "text": summary_explanation_text
        }

        # -------------------------------------------------------------
        # 10-SECTION BACKWARD COMPATIBILITY MATRIX
        # -------------------------------------------------------------
        sec1_executive_summary = {
            "report_id": rid,
            "timestamp_utc": now,
            "overall_verdict": verdict,
            "deepfake_probability_score": fusion_score,
            "calibrated_threshold": CALIBRATED_THRESHOLD,
            "calibrated_confidence": confidence,
            "risk_classification": risk_level,
            "primary_rationale": verdict_statement,
            "verdict_statement": verdict_statement,
            "disclaimer": disclaimer_text
        }

        sec2_architecture = {
            "platform_name": "DeepShield Multimodal Verification Platform",
            "model_identifier": "DeepShield V5 Clean Best",
            "backbone_architecture": "EfficientNetB2 (ImageNet Transfer Pretrained)",
            "input_resolution": "224 x 224 x 3 (RGB Float32 [0, 255])",
            "classification_head": "GlobalAveragePooling2D -> Dropout(0.3) -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(1, Sigmoid)",
            "class_encoding": {0: "Real (Authentic Media)", 1: "Fake (Synthetic / Manipulated Media)"},
            "activation_interpretation": "Higher sigmoid score indicates increased likelihood of digital manipulation"
        }

        sec3_benchmark = {
            "evaluation_dataset": "DeepShield FAST10 Multi-Generator Balanced Benchmark (Test Split: 3,693 images)",
            "accuracy": "77.55%",
            "roc_auc": "83.30%",
            "precision": "87.83%",
            "recall": "80.44%",
            "calibrated_optimal_f1_threshold": CALIBRATED_THRESHOLD,
            "optimal_f1_score": 0.8771,
            "scientific_integrity_note": "Evaluated strictly on independent test split without data leakage."
        }

        sec4_face_audit = {
            "detector_algorithm": "OpenCV YuNet CNN-based Face Detector (ONNX)",
            "faces_detected": sec_image_evidence["face_count"],
            "face_detected": sec_image_evidence["face_detected"],
            "primary_face_bounding_box": faces_raw[0]["box"] if len(faces_raw) > 0 and "box" in faces_raw[0] else None,
            "detector_confidence": faces_raw[0]["score"] if len(faces_raw) > 0 and "score" in faces_raw[0] else 0.0,
            "localization_status": (
                "Primary face localized with square crop geometry for neural analysis."
                if sec_image_evidence["face_detected"]
                else "No discrete face localized. Neural analysis executed on context."
            )
        }

        sec5_explainability = {
            "methodology": "Gradient-weighted Class Activation Mapping (Grad-CAM)",
            "target_activation_layer": sec_image_evidence["gradcam"]["target_layer"],
            "gradcam_available": sec_image_evidence["gradcam"]["available"],
            "heatmap_url": heatmap_url,
            "overlay_url": overlay_url,
            "peak_activation_intensity": sec_image_evidence["gradcam"]["peak_intensity"],
            "mean_activation_intensity": sec_image_evidence["gradcam"]["mean_intensity"],
            "salient_facial_region_coverage": f"{sec_image_evidence['gradcam']['salient_ratio'] * 100:.1f}%",
            "interpretation_summary": (
                "The highlighted regions show areas that influenced the model's decision."
                if sec_image_evidence["gradcam"]["available"]
                else "Grad-CAM visualization currently unavailable for this sample."
            )
        }

        sec6_metadata = {
            "file_format": sec_metadata_evidence["file_format"],
            "dimensions": sec_metadata_evidence["dimensions"],
            "file_size": sec_metadata_evidence["file_size"],
            "exif_present": sec_metadata_evidence["exif_present"],
            "camera_hardware": sec_metadata_evidence["camera_hardware"],
            "editing_software_detected": sec_metadata_evidence["editing_software"]["detected"],
            "ai_generator_signature": sec_metadata_evidence["editing_software"]["ai_generator_signature"],
            "forensic_notes": sec_metadata_evidence["forensic_notes"]
        }

        sec7_uncertainty = {
            "calibrated_threshold": CALIBRATED_THRESHOLD,
            "score_distance_from_threshold": sec_uncertainty["score_distance_from_threshold"],
            "boundary_uncertainty_window": sec_uncertainty["boundary_window"],
            "is_within_uncertainty_boundary": sec_uncertainty["is_within_boundary_window"],
            "uncertainty_advisory": sec_uncertainty["advisory"]
        }

        sec8_evidence_matrix = {
            "discrete_evidence_items": prediction_result.get("evidence", []),
            "corroborating_factors": corroborating_factors,
            "conflicting_factors": conflict_factors
        }

        sec9_disclaimer = {
            "disclaimer_text": (
                "DeepShield is designed as a multimodal decision-support system for digital forensics and academic research. "
                "This report must be used in conjunction with qualified human forensic expertise. " + disclaimer_text
            ),
            "integrity_attestation": "Model parameters and calibrated thresholds are recorded without retrospective alteration."
        }

        sec10_recommendations = [
            "Archive this forensic report with cryptographic hash of original media for chain-of-custody preservation.",
            (
                f"Prioritize manual review: {sec_uncertainty['uncertainty_reason'] or 'Score lies near decision boundary.'}"
                if is_uncertain or is_boundary_uncertain
                else "Sample demonstrates clear separation from decision boundary; routine verification applies."
            ),
            "Cross-reference EXIF timestamps and device signatures against contextual acquisition records.",
            "Inspect high-frequency facial boundaries and temporal keyframe consistency maps for localization evidence."
        ]

        sections = {
            "1_executive_summary": sec1_executive_summary,
            "2_architecture_specifications": sec2_architecture,
            "3_benchmark_performance": sec3_benchmark,
            "4_face_localization_audit": sec4_face_audit,
            "5_explainability_attention": sec5_explainability,
            "6_metadata_forensics": sec6_metadata,
            "7_decision_boundary_uncertainty": sec7_uncertainty,
            "8_evidence_matrix": sec8_evidence_matrix,
            "9_forensic_disclaimer": sec9_disclaimer,
            "10_recommended_actions": sec10_recommendations,
            "11_multimodal_fusion": sec_multimodal_fusion,
            "12_video_forensics": sec_video_evidence
        }

        # Format full markdown text
        report_text = ForensicReportGenerator.format_markdown({
            "report_id": rid,
            "timestamp": now,
            "sections": sections,
            "verdict": verdict,
            "confidence": confidence,
            "risk_level": risk_level,
            "summary": sec_qualified_explanations,
            "evidence": {
                "image_evidence": sec_image_evidence,
                "video_evidence": sec_video_evidence,
                "metadata_evidence": sec_metadata_evidence
            },
            "multimodal_fusion": sec_multimodal_fusion,
            "uncertainty": sec_uncertainty,
            "technical_details": sec_technical_details
        })

        return {
            "report_id": rid,
            "timestamp": now,
            "timestamp_utc": now,
            "verdict": verdict,
            "overall_verdict": verdict,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "summary": sec_qualified_explanations,
            "evidence": {
                "image_evidence": sec_image_evidence,
                "video_evidence": sec_video_evidence,
                "metadata_evidence": sec_metadata_evidence
            },
            "multimodal_fusion": sec_multimodal_fusion,
            "uncertainty": sec_uncertainty,
            "technical_details": sec_technical_details,
            "report_text": report_text,
            "visualization_paths": {
                "original_url": orig_url,
                "face_crop_url": face_crop_url,
                "heatmap_url": heatmap_url,
                "overlay_url": overlay_url
            },
            "sections": sections
        }

    @staticmethod
    def format_markdown(report: Dict[str, Any]) -> str:
        """Formats the structured report into clean GitHub-flavored forensic markdown."""
        s = report.get("sections", {})
        e = s.get("1_executive_summary", {})
        b = s.get("3_benchmark_performance", {})
        u = report.get("uncertainty", s.get("7_decision_boundary_uncertainty", {}))
        f = report.get("multimodal_fusion", s.get("11_multimodal_fusion", {}))
        img = report.get("evidence", {}).get("image_evidence", {})
        vid = report.get("evidence", {}).get("video_evidence", {})
        meta = report.get("evidence", {}).get("metadata_evidence", {})

        rid = report.get("report_id", e.get("report_id", "DSR-UNKNOWN"))
        ts = report.get("timestamp", e.get("timestamp_utc", ""))
        verdict = report.get("verdict", e.get("overall_verdict", "Real"))
        conf = report.get("confidence", e.get("calibrated_confidence", 0.5))
        risk = report.get("risk_level", e.get("risk_classification", "Moderate"))
        fscore = f.get("fusion_score", e.get("deepfake_probability_score", 0.0))
        thresh = f.get("calibrated_threshold", CALIBRATED_THRESHOLD)

        md = []
        md.append(f"# DeepShield Forensic Verification Audit Report")
        md.append(f"**Report ID**: `{rid}` | **Timestamp**: `{ts}`\n")
        md.append("---")

        md.append("## 1. Executive Summary & Final Verdict")
        md.append(f"- **Final Verdict**: **{verdict}**")
        md.append(f"- **Calibrated Confidence**: `{conf * 100:.1f}%`")
        md.append(f"- **Risk Classification**: **{risk}**")
        md.append(f"- **Multimodal Fusion Score**: `{fscore:.4f}` (Decision Threshold: `{thresh}`)")
        md.append(f"- **Assessment Statement**: {e.get('verdict_statement', report.get('summary', ''))}")
        md.append(f"> *Responsible AI Disclaimer: {e.get('disclaimer', 'This is an AI-based assessment, not absolute proof of authenticity.')}*\n")

        md.append("## 2. Multimodal Evidence Synthesis")
        md.append(f"- **Active Modalities**: {', '.join(f.get('modalities_available', ['none'])).title()}")
        md.append(f"- **Inter-Modal Conflict Detected**: **{f.get('conflict_detected', False)}**")

        if f.get("conflict_factors"):
            md.append("- **Conflicting Factors**:")
            for cf in f.get("conflict_factors"):
                md.append(f"  - ⚠️ {cf}")

        if f.get("corroborating_factors"):
            md.append("- **Corroborating Factors**:")
            for corr in f.get("corroborating_factors"):
                md.append(f"  - ✓ {corr}")
        md.append("")

        if img.get("evidence_available"):
            md.append("## 3. Image Evidence Analysis")
            md.append(f"- **Face Localized**: `{img.get('face_detected')}` ({img.get('face_count', 0)} face(s) detected)")
            md.append(f"- **Image Model Score**: `{img.get('model_score', 'N/A')}` | Verdict: `{img.get('image_prediction', 'N/A')}`")
            g = img.get("gradcam", {})
            md.append(f"- **Grad-CAM Attention**: Available: `{g.get('available')}` (Layer: `{g.get('target_layer')}`)")
            if g.get("available"):
                md.append(f"  - *Interpretation*: {g.get('explanation', 'Highlighted regions show areas that influenced the decision.')}")
                md.append(f"  - *Salient Coverage*: `{g.get('salient_ratio', 0.0) * 100:.1f}%` | Peak Intensity: `{g.get('peak_intensity', 0.0):.3f}`")
            md.append("")

        if vid.get("evidence_available"):
            md.append("## 4. Video Temporal & Quality Forensics")
            tev = vid.get("v5_2_temporal_evidence", {})
            if tev and tev.get("available"):
                md.append("### DeepShield V5.2 Dedicated Temporal Video Evidence")
                t_thr = float(tev.get('temporal_threshold', 0.30))
                md.append(f"- **Temporal Score**: `{tev.get('temporal_score')}` (Calibrated Temporal Threshold: `{t_thr:.2f}`)")
                md.append(f"- **Temporal Verdict**: **{tev.get('prediction', 'N/A')}** | Valid Face Frames: `{tev.get('valid_face_frames', 'N/A')}`")
                md.append(f"- **Pipeline Concordance**: {tev.get('conflict_status', 'N/A')}")
                md.append("")
                md.append("### Phase 1 Frame-Level Forensics & Quality")
            md.append(f"- **Sampled Frames**: `{vid.get('total_frames_sampled', vid.get('frames_analyzed', 0))}` keyframes | Valid face frames: `{vid.get('face_frames', vid.get('faces_detected', 0))}` | Non-face skipped: `{vid.get('skipped_no_face_frames', 0)}`")
            md.append(f"- **Raw Face Median Score**: `{vid.get('raw_model_score', vid.get('median_score', 'N/A'))}` | Calibrated Fake Evidence Score: `{vid.get('video_calibrated_score', 'N/A')}`")
            md.append(f"- **P75 Score**: `{vid.get('p75_score', 'N/A')}` | Score Std: `{vid.get('score_std', 'N/A')}`")
            md.append(f"- **Temporal Volatility**: `{vid.get('temporal_volatility', 'N/A')}` | Persistence Ratio: `{vid.get('persistence_ratio', 'N/A')}`")
            md.append(f"- **Compression Degradation Flag**: **{vid.get('compression_degradation', False)}**")
            if vid.get("calibration_note"):
                md.append(f"- **Calibration Note**: {vid.get('calibration_note')}")
            if vid.get("uncertainty_reason"):
                md.append(f"- **Video Forensics Note**: {vid.get('uncertainty_reason')}")
            md.append("")

        if meta.get("evidence_available"):
            md.append("## 5. Metadata & Provenance Audit")
            md.append(f"- **Format**: `{meta.get('file_format')}` | Dimensions: `{meta.get('dimensions')}` | Size: `{meta.get('file_size')}`")
            md.append(f"- **EXIF Header Present**: `{meta.get('exif_present')}`")
            hw = meta.get("camera_hardware", {})
            if hw.get("make") or hw.get("model"):
                md.append(f"- **Camera Hardware**: `{hw.get('make', '')} {hw.get('model', '')}`.strip()")
            ed = meta.get("editing_software", {})
            if ed.get("detected"):
                md.append(f"- **Editing Software Detected**: `{ed.get('name')}` (AI signature: `{ed.get('ai_generator_signature')}`)")
            for note in meta.get("forensic_notes", []):
                md.append(f"- *Note*: {note}")
            md.append("")

        if u.get("is_uncertain") or u.get("is_within_boundary_window"):
            md.append("## 6. Uncertainty Analysis & Forensic Boundary")
            md.append(f"- **Uncertainty Status**: **{u.get('is_uncertain', False)}**")
            if u.get("uncertainty_reason"):
                md.append(f"- **Forensic Cause**: {u.get('uncertainty_reason')}")
            md.append(f"- **Boundary Window**: `[{u.get('boundary_window', [0.16, 0.26])[0]} - {u.get('boundary_window', [0.16, 0.26])[1]}]` around threshold `{thresh}`")
            md.append(f"- **Distance from Threshold**: `{u.get('score_distance_from_threshold', 'N/A')}`")
            md.append(f"- **Advisory**: {u.get('advisory', '')}\n")

        md.append("## 7. Technical Specifications & Architecture")
        t = report.get("technical_details", {})
        md.append(f"- **Model**: `{t.get('model_name', 'DeepShield V5')}` (`{t.get('backbone_architecture', 'EfficientNetB2')}`)")
        md.append(f"- **Face Detector**: `{t.get('face_detector', 'OpenCV YuNet')}`")
        md.append(f"- **Resolution**: `{t.get('input_resolution', '224x224x3')}`")
        md.append(f"- **Calibrated Threshold**: `{thresh}`")
        md.append(f"- **Fusion Version**: `{t.get('fusion_version', 'v5-multimodal')}`")
        md.append(f"- **Benchmark ROC-AUC**: `{b.get('roc_auc', '83.30%')}` | Optimal F1: `{b.get('optimal_f1_score', '0.8771')}`\n")

        md.append("## 10. Recommended Next Actions")
        rec = s.get("10_recommended_actions", [
            "Archive this forensic report with cryptographic hash for chain-of-custody.",
            "Cross-reference metadata and temporal metrics with acquisition timeline."
        ])
        for idx, act in enumerate(rec, 1):
            md.append(f"{idx}. {act}")

        return "\n".join(md)

