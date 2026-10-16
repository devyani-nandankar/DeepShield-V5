import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CALIBRATED_THRESHOLD, IMAGE_SIZE
from backend.face_detector import get_face_detector
from backend.image_detector import get_image_detector

face_detector = get_face_detector()
image_detector = get_image_detector()
model = image_detector.model
explainer = image_detector.explainer

diag_dir = PROJECT_ROOT / "evaluation" / "video_diagnostics"
diag_dir.mkdir(parents=True, exist_ok=True)

real_dir = PROJECT_ROOT / "backend" / "test_images" / "videos_real"
fake_dir = PROJECT_ROOT / "backend" / "test_images" / "videos_fake"

real_files = sorted(list(real_dir.glob("*.mp4")))
fake_files = sorted(list(fake_dir.glob("*.mp4")))

all_files = [(f, "Real") for f in real_files] + [(f, "Fake") for f in fake_files]

print(f"Total videos to analyze: {len(all_files)}")

results = []

for vfile, label in all_files:
    cap = cv2.VideoCapture(str(vfile))
    if not cap.isOpened():
        print(f"Could not open {vfile.name}")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(total_frames / fps, 2) if fps > 0 else 0.0

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
    if not fourcc_str:
        fourcc_str = "H264/AVC"

    target_max_frames = 40
    if total_frames > (target_max_frames * 10):
        frame_step = max(10, total_frames // target_max_frames)
    else:
        frame_step = 10

    sampled = []
    frame_idx = 0

    while cap.isOpened() and len(sampled) < target_max_frames:
        if frame_idx % frame_step != 0:
            grabbed = cap.grab()
            if not grabbed:
                break
            frame_idx += 1
            continue

        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        det = face_detector.detect_faces(frame_bgr)
        has_face = det.face_detected and (det.crop is not None) and (det.crop.size > 0)

        face_w, face_h, face_conf = 0, 0, 0.0
        if has_face:
            crop_bgr = det.crop
            face_conf = float(det.faces[0]["score"]) if det.faces else 1.0
            bbox = det.faces[0]["bbox"]
            face_w = bbox[2]
            face_h = bbox[3]
        else:
            crop_bgr = face_detector._center_crop(frame_bgr)
            face_conf = 0.0
            face_w = width
            face_h = height

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        crop_tensor = crop_resized.astype(np.float32)

        sampled.append({
            "frame_idx": frame_idx,
            "has_face": has_face,
            "face_conf": face_conf,
            "face_w": face_w,
            "face_h": face_h,
            "face_aspect": (face_w / float(face_h)) if face_h > 0 else 1.0,
            "crop_rgb": crop_rgb,
            "crop_tensor": crop_tensor
        })
        frame_idx += 1

    cap.release()

    if not sampled:
        continue

    # Batched model inference
    batch_tensor = np.stack([s["crop_tensor"] for s in sampled], axis=0)
    preds = model.predict(batch_tensor, batch_size=32, verbose=0)

    scores = [float(p[0]) for p in preds]
    for i, s in enumerate(sampled):
        s["score"] = scores[i]

    mean_s = float(np.mean(scores))
    med_s = float(np.median(scores))
    min_s = float(np.min(scores))
    max_s = float(np.max(scores))
    std_s = float(np.std(scores))

    fake_frames = [s for s in sampled if s["score"] >= CALIBRATED_THRESHOLD]
    fake_ratio = float(len(fake_frames) / len(sampled))

    faces_detected = sum(1 for s in sampled if s["has_face"])
    face_confs = [s["face_conf"] for s in sampled if s["has_face"]]
    avg_conf = float(np.mean(face_confs)) if face_confs else 0.0

    face_ws = [s["face_w"] for s in sampled if s["has_face"]]
    face_hs = [s["face_h"] for s in sampled if s["has_face"]]
    face_as = [s["face_aspect"] for s in sampled if s["has_face"]]

    avg_w = float(np.mean(face_ws)) if face_ws else float(width)
    avg_h = float(np.mean(face_hs)) if face_hs else float(height)
    avg_a = float(np.mean(face_as)) if face_as else 1.0

    rec = {
        "filename": vfile.name,
        "actual_label": label,
        "mean_score": round(mean_s, 4),
        "median_score": round(med_s, 4),
        "min_score": round(min_s, 4),
        "max_score": round(max_s, 4),
        "std_score": round(std_s, 4),
        "fake_frame_ratio": round(fake_ratio, 4),
        "frames_analyzed": len(sampled),
        "faces_detected": faces_detected,
        "avg_face_confidence": round(avg_conf, 4),
        "avg_face_width": round(avg_w, 1),
        "avg_face_height": round(avg_h, 1),
        "face_aspect_ratio": round(avg_a, 3),
        "video_resolution": f"{width}x{height}",
        "video_width": width,
        "video_height": height,
        "fps": round(fps, 2),
        "duration": duration_sec,
        "codec": fourcc_str,
        "sampled_frames": sampled
    }
    results.append(rec)
    print(f"Done: [{label}] {vfile.name[:30]} | Mean: {rec['mean_score']} | Min/Max: [{rec['min_score']}, {rec['max_score']}] | FaceSize: {rec['avg_face_width']}x{rec['avg_face_height']}")

# Save representative images for 4 selected videos:
real_res = [r for r in results if r["actual_label"] == "Real"]
fake_res = [r for r in results if r["actual_label"] == "Fake"]

real_highest = max(real_res, key=lambda x: x["mean_score"])
real_lowest = min(real_res, key=lambda x: x["mean_score"])
fake_highest = max(fake_res, key=lambda x: x["mean_score"])
fake_lowest = min(fake_res, key=lambda x: x["mean_score"])

rep_videos = [
    ("REAL_HIGHEST", real_highest),
    ("REAL_LOWEST", real_lowest),
    ("FAKE_HIGHEST", fake_highest),
    ("FAKE_LOWEST", fake_lowest)
]

rep_artifacts = {}

for role, vrec in rep_videos:
    print(f"\nProcessing representative: {role} ({vrec['filename']})")
    # Top suspicious frame (highest score)
    top_frame = max(vrec["sampled_frames"], key=lambda s: s["score"])
    
    # Save face crop
    crop_fname = f"{role}_crop_f{top_frame['frame_idx']}.jpg"
    crop_path = diag_dir / crop_fname
    cv2.imwrite(str(crop_path), cv2.cvtColor(top_frame["crop_rgb"], cv2.COLOR_RGB2BGR))
    
    # Generate Grad-CAM for this single frame
    single_tensor = tf.convert_to_tensor(np.expand_dims(top_frame["crop_tensor"], axis=0))
    gcam = explainer.generate(single_tensor, top_frame["crop_rgb"])
    
    overlay_fname = f"{role}_gradcam_overlay_f{top_frame['frame_idx']}.jpg"
    heatmap_fname = f"{role}_gradcam_heatmap_f{top_frame['frame_idx']}.jpg"
    
    # If URLs returned, copy artifacts into diag_dir
    from backend.config import GENERATED_FOLDER
    if gcam.get("overlay_filename"):
        src_over = GENERATED_FOLDER / gcam["overlay_filename"]
        if src_over.exists():
            cv2.imwrite(str(diag_dir / overlay_fname), cv2.imread(str(src_over)))
    if gcam.get("heatmap_filename"):
        src_heat = GENERATED_FOLDER / gcam["heatmap_filename"]
        if src_heat.exists():
            cv2.imwrite(str(diag_dir / heatmap_fname), cv2.imread(str(src_heat)))

    rep_artifacts[role] = {
        "filename": vrec["filename"],
        "mean_score": vrec["mean_score"],
        "top_frame_idx": top_frame["frame_idx"],
        "top_frame_score": round(top_frame["score"], 4),
        "crop_path": str(crop_path),
        "gradcam_peak": gcam.get("peak_intensity"),
        "gradcam_salient": gcam.get("salient_ratio"),
        "gradcam_overlay": str(diag_dir / overlay_fname)
    }

# Clean sampled_frames from json output to avoid huge file
for r in results:
    del r["sampled_frames"]

diag_summary = {
    "representative_artifacts": rep_artifacts,
    "all_records": results
}

out_json = PROJECT_ROOT / "evaluation" / "video_diagnostics_summary.json"
with open(out_json, "w") as f:
    json.dump(diag_summary, f, indent=2)

print("\n" + "="*70)
print("DIAGNOSTICS COMPLETE. Results saved to evaluation/video_diagnostics_summary.json")
print("="*70)
