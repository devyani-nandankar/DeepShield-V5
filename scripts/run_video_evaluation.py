import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.video_detector import get_video_detector

detector = get_video_detector()

real_dir = PROJECT_ROOT / "backend" / "test_images" / "videos_real"
fake_dir = PROJECT_ROOT / "backend" / "test_images" / "videos_fake"

real_files = sorted(list(real_dir.glob("*.mp4")))
fake_files = sorted(list(fake_dir.glob("*.mp4")))

print(f"Loaded {len(real_files)} real videos and {len(fake_files)} fake videos.")

eval_records = []

# 1. Real Videos
print("\n--- Evaluating Real Videos ---")
for idx, vfile in enumerate(real_files, 1):
    t0 = time.time()
    res = detector.predict(str(vfile))
    elapsed = round(time.time() - t0, 3)
    rec = {
        "index": idx,
        "filename": vfile.name,
        "actual_label": "Real",
        "predicted_verdict": res.get("verdict"),
        "prediction": res.get("prediction"),
        "confidence": res.get("confidence"),
        "mean_score": res.get("mean_score"),
        "median_score": res.get("median_score"),
        "fake_frame_ratio": res.get("fake_frame_ratio"),
        "frames_analyzed": res.get("frames_analyzed"),
        "faces_detected": res.get("faces_detected"),
        "processing_time": elapsed,
        "is_correct": (res.get("verdict") == "Real")
    }
    eval_records.append(rec)
    mark = "CORRECT" if rec["is_correct"] else "INCORRECT"
    print(f"[{idx}/{len(real_files)}] REAL: {vfile.name} -> {rec['predicted_verdict']} (mean={rec['mean_score']}, ratio={rec['fake_frame_ratio']}, time={elapsed}s) [{mark}]")

# 2. Fake Videos
print("\n--- Evaluating Fake Videos ---")
for idx, vfile in enumerate(fake_files, 1):
    t0 = time.time()
    res = detector.predict(str(vfile))
    elapsed = round(time.time() - t0, 3)
    rec = {
        "index": idx,
        "filename": vfile.name,
        "actual_label": "Fake",
        "predicted_verdict": res.get("verdict"),
        "prediction": res.get("prediction"),
        "confidence": res.get("confidence"),
        "mean_score": res.get("mean_score"),
        "median_score": res.get("median_score"),
        "fake_frame_ratio": res.get("fake_frame_ratio"),
        "frames_analyzed": res.get("frames_analyzed"),
        "faces_detected": res.get("faces_detected"),
        "processing_time": elapsed,
        "is_correct": (res.get("verdict") == "Likely Fake")
    }
    eval_records.append(rec)
    mark = "CORRECT" if rec["is_correct"] else "INCORRECT"
    print(f"[{idx}/{len(fake_files)}] FAKE: {vfile.name} -> {rec['predicted_verdict']} (mean={rec['mean_score']}, ratio={rec['fake_frame_ratio']}, time={elapsed}s) [{mark}]")

# Confusion Matrix
tp = sum(1 for r in eval_records if r["actual_label"] == "Fake" and r["predicted_verdict"] == "Likely Fake")
fn = sum(1 for r in eval_records if r["actual_label"] == "Fake" and r["predicted_verdict"] != "Likely Fake")
tn = sum(1 for r in eval_records if r["actual_label"] == "Real" and r["predicted_verdict"] == "Real")
fp = sum(1 for r in eval_records if r["actual_label"] == "Real" and r["predicted_verdict"] != "Real")

total = len(eval_records)
correct = tp + tn
incorrect = fp + fn
accuracy = correct / total if total > 0 else 0.0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
avg_time = sum(r["processing_time"] for r in eval_records) / total if total > 0 else 0.0

incorrect_videos = [r for r in eval_records if not r["is_correct"]]

summary = {
    "total_videos": total,
    "total_real": len(real_files),
    "total_fake": len(fake_files),
    "correct_predictions": correct,
    "incorrect_predictions": incorrect,
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    "real_video_accuracy": round(real_acc, 4),
    "fake_video_accuracy": round(fake_acc, 4),
    "average_processing_time": round(avg_time, 3),
    "incorrect_videos": incorrect_videos,
    "records": eval_records
}

out_file = PROJECT_ROOT / "evaluation" / "video_evaluation_results.json"
with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("FINAL VIDEO EVALUATION SUMMARY")
print("="*70)
print(f"Total Videos Analyzed: {total}")
print(f"Total Real Videos:     {len(real_files)}")
print(f"Total Fake Videos:     {len(fake_files)}")
print(f"Correct Predictions:   {correct}")
print(f"Incorrect Predictions: {incorrect}")
print(f"Final Video Accuracy:  {accuracy*100:.2f}%")
print(f"Precision:             {precision*100:.2f}%")
print(f"Recall:                {recall*100:.2f}%")
print(f"F1-Score:              {f1:.4f}")
print(f"Real Video Accuracy:   {real_acc*100:.2f}% ({tn}/{len(real_files)})")
print(f"Fake Video Accuracy:   {fake_acc*100:.2f}% ({tp}/{len(fake_files)})")
print(f"Confusion Matrix:      TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Average Processing:    {avg_time:.3f}s per video")
print("="*70)
print(f"Incorrectly Classified Videos ({len(incorrect_videos)}):")
for iv in incorrect_videos:
    print(f"  * {iv['filename']}: Actual={iv['actual_label']}, Predicted={iv['predicted_verdict']}, Mean={iv['mean_score']}, Ratio={iv['fake_frame_ratio']}")
print(f"\nSaved detailed results to {out_file}")
