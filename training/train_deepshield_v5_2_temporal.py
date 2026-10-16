"""
DeepShield V5.2 - Large-Scale Temporal Video Model Training Pipeline
====================================================================

Trains the Dedicated BiLSTM Temporal Classifier on the Verified V5.2 Dataset.

Requirements & Integrity Protocols:
- Reuses V5 EfficientNetB2 spatial weights (models/deepshield_v5_clean_best.keras) as frozen feature extractor
- Does NOT overwrite V5 or V5.1 model weights
- Saves exclusively to models/v5_2/
- Trains two experimental configurations:
    Model A: No compression augmentation
    Model B: With compression augmentation (p=0.60: JPEG, blur, downsampling, noise)
- Manifest-driven: Uses Metadata/v5_2_sequence_manifest.csv & Metadata/v5_2_video_manifest.csv
- Threshold calibration performed strictly on Validation videos (F1 maximization)
- Evaluates official held-out test (4 videos) and quarantined WhatsApp benchmark (20 videos)
- Logs complete training history, confusion matrices, and inference timing
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, roc_curve

workspace = r"c:\Users\ADMIN\Desktop\DeepShield_V5_Final"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

from backend.config import YUNET_MODEL_PATH, IMAGE_SIZE
from backend.face_detector import YuNetFaceDetector
from training.video_sequence_dataset import VideoSequenceDataset, CompressionAugmenter

# Configuration Constants
T = 16
FEATURE_DIM = 1408
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-3

V5_BASE_MODEL_PATH = os.path.join(workspace, "models", "deepshield_v5_clean_best.keras")
V5_2_OUTPUT_DIR = os.path.join(workspace, "models", "v5_2")
EVAL_DIR = os.path.join(workspace, "evaluation")
DOCS_DIR = os.path.join(workspace, "docs")
METADATA_DIR = os.path.join(workspace, "Metadata")

os.makedirs(V5_2_OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


def build_bilstm_temporal_model(sequence_length=16, feature_dim=1408, learning_rate=1e-3, name="DeepShield_V5_2_BiLSTM"):
    """
    Constructs the V5.2 BiLSTM Temporal Deepfake Classifier.
    Face frame -> EfficientNetB2 -> GAP -> 1408-d -> Masking -> BiLSTM(128) -> Drop(0.3) -> BiLSTM(64) -> Drop(0.3) -> Dense(64, relu) -> Drop(0.3) -> Dense(1, sigmoid)
    """
    inputs = layers.Input(shape=(sequence_length, feature_dim), name="temporal_features_input")
    
    # Mask zero-padded frames
    masked = layers.Masking(mask_value=0.0, name="temporal_masking")(inputs)
    
    # BiLSTM Layer 1 (128 units, sequence return)
    lstm1 = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0),
        name="bilstm_layer_1"
    )(masked)
    drop1 = layers.Dropout(0.3, name="dropout_1")(lstm1)
    
    # BiLSTM Layer 2 (64 units, final temporal summary)
    lstm2 = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.0),
        name="bilstm_layer_2"
    )(drop1)
    drop2 = layers.Dropout(0.3, name="dropout_2")(lstm2)
    
    # Dense Projection
    dense = layers.Dense(64, activation="relu", name="dense_projection")(drop2)
    drop3 = layers.Dropout(0.3, name="dropout_3")(dense)
    
    # Final Output Sigmoid
    outputs = layers.Dense(1, activation="sigmoid", name="video_fake_probability")(drop3)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def load_v5_feature_extractor(v5_path: str):
    """Loads the V5 base model and creates a frozen feature extractor (layer 2: GlobalAveragePooling2D)."""
    print(f"[+] Loading frozen V5 spatial feature extractor from: {v5_path}")
    v5_base = tf.keras.models.load_model(v5_path)
    # Output of layer 2 is 1408-dimensional GlobalAveragePooling2D
    feature_extractor = tf.keras.Model(
        inputs=v5_base.input,
        outputs=v5_base.layers[2].output,
        name="V5_Spatial_Feature_Extractor"
    )
    feature_extractor.trainable = False
    return feature_extractor


def extract_features_from_manifest(
    manifest_path: str,
    feature_extractor,
    dataset_builder: VideoSequenceDataset,
    cache_path: Optional[str] = None
):
    """
    Extracts features for all sequences in the manifest.
    If cached .npz file exists, loads from cache for speed.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[+] Loading cached features from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["X_train"], data["y_train"], data["train_meta"].tolist(),
            data["X_val"], data["y_val"], data["val_meta"].tolist(),
            data["X_test"], data["y_test"], data["test_meta"].tolist(),
            data["X_wa"], data["y_wa"], data["wa_meta"].tolist(),
        )

    df_seq = pd.read_csv(manifest_path)
    print(f"[+] Processing {len(df_seq)} sequences from manifest: {manifest_path}")

    splits_data = {
        "train": {"X": [], "y": [], "meta": []},
        "validation": {"X": [], "y": [], "meta": []},
        "test": {"X": [], "y": [], "meta": []},
        "benchmark_calibration": {"X": [], "y": [], "meta": []}
    }

    # Group sequences by video to read each video in a single sequential pass
    video_sequences = {}
    for _, row in df_seq.iterrows():
        items = row["frame_paths"].split(";")
        v_relpath = items[0].split(":")[0]
        frame_indices = [int(it.split(":")[1].replace("frame_", "")) for it in items]
        video_sequences.setdefault(v_relpath, []).append({
            "row": row,
            "ordered_indices": frame_indices,
            "needed_set": set(frame_indices)
        })

    for v_relpath, seq_items in video_sequences.items():
        v_abspath = os.path.join(workspace, v_relpath)
        if not os.path.exists(v_abspath):
            print(f"[!] Warning: video not found: {v_abspath}")
            continue

        all_target_frames = set()
        for s in seq_items:
            all_target_frames.update(s["needed_set"])
        max_frame = max(all_target_frames) if all_target_frames else 0

        # Read frames sequentially
        cap = cv2.VideoCapture(v_abspath)
        frame_cache = {}
        curr_idx = 0
        while cap.isOpened() and curr_idx <= max_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if curr_idx in all_target_frames:
                frame_cache[curr_idx] = frame
            curr_idx += 1
        cap.release()

        # Now construct each sequence for this video
        for s in seq_items:
            row = s["row"]
            seq_id = row["sequence_id"]
            vid_id = row["video_id"]
            label = int(row["label"])
            split = row["split"]
            source = row["source_dataset"]

            crops = []
            for f_idx in s["ordered_indices"]:
                frame_bgr = frame_cache.get(f_idx)
                if frame_bgr is None:
                    continue

                if dataset_builder.use_aug and dataset_builder.augmenter:
                    frame_bgr = dataset_builder.augmenter.augment_frame(frame_bgr)

                det_res = dataset_builder.face_detector.detect_faces(frame_bgr)
                if det_res.face_detected and det_res.crop is not None and det_res.crop.size > 0:
                    c_bgr = det_res.crop
                    c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
                    c_res = cv2.resize(c_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                    crops.append(c_res.astype(np.float32))

            valid_count = len(crops)
            if valid_count == 0:
                padded = np.zeros((T, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
                seq_tensor = padded
            elif valid_count < T:
                padded = np.zeros((T, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
                for i, c in enumerate(crops):
                    padded[i] = c
                seq_tensor = padded
            else:
                seq_tensor = np.stack(crops[:T], axis=0)

            feats = dataset_builder.extract_features_for_sequence(seq_tensor, feature_extractor)
            splits_data[split]["X"].append(feats)
            splits_data[split]["y"].append(label)
            splits_data[split]["meta"].append({
                "sequence_id": seq_id,
                "video_id": vid_id,
                "filepath": v_relpath,
                "label": label,
                "source": source,
                "valid_crops": valid_count
            })

    def pack_split(key):
        X = np.stack(splits_data[key]["X"], axis=0).astype(np.float32) if splits_data[key]["X"] else np.zeros((0, T, FEATURE_DIM), dtype=np.float32)
        y = np.array(splits_data[key]["y"], dtype=np.float32)
        meta = splits_data[key]["meta"]
        return X, y, meta

    X_train, y_train, train_meta = pack_split("train")
    X_val, y_val, val_meta = pack_split("validation")
    X_test, y_test, test_meta = pack_split("test")
    X_wa, y_wa, wa_meta = pack_split("benchmark_calibration")

    if cache_path:
        np.savez_compressed(
            cache_path,
            X_train=X_train, y_train=y_train, train_meta=np.array(train_meta, dtype=object),
            X_val=X_val, y_val=y_val, val_meta=np.array(val_meta, dtype=object),
            X_test=X_test, y_test=y_test, test_meta=np.array(test_meta, dtype=object),
            X_wa=X_wa, y_wa=y_wa, wa_meta=np.array(wa_meta, dtype=object)
        )
        print(f"[+] Saved feature cache to: {cache_path}")

    return X_train, y_train, train_meta, X_val, y_val, val_meta, X_test, y_test, test_meta, X_wa, y_wa, wa_meta


def calibrate_validation_threshold(val_y, val_probs):
    """
    Determines optimal decision threshold purely on validation set via F1 maximization.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1 = -1.0
    best_thresh = 0.50
    
    for th in thresholds:
        preds = (val_probs >= th).astype(int)
        tp = np.sum((val_y == 1) & (preds == 1))
        fp = np.sum((val_y == 0) & (preds == 1))
        fn = np.sum((val_y == 1) & (preds == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(th)
            
    return round(best_thresh, 3), round(best_f1, 4)


def compute_metrics(y_true, y_probs, threshold=0.50):
    """Computes comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, CM."""
    preds = (y_probs >= threshold).astype(int)
    
    tp = int(np.sum((y_true == 1) & (preds == 1)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    
    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    try:
        if len(np.unique(y_true)) > 1:
            auc_val = float(roc_auc_score(y_true, y_probs))
        else:
            auc_val = 0.50
    except Exception:
        auc_val = 0.50
        
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc_val, 4),
        "real_accuracy": round(real_acc, 4),
        "fake_accuracy": round(fake_acc, 4),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "sample_count": total
    }


def aggregate_video_level_predictions(meta_list, seq_probs, threshold=0.50):
    """
    Aggregates sequence predictions to video level via mean probability and majority voting.
    """
    video_map = {}
    for meta, prob in zip(meta_list, seq_probs):
        vid_id = meta["video_id"]
        video_map.setdefault(vid_id, {
            "probs": [],
            "label": meta["label"],
            "filepath": meta["filepath"]
        })
        video_map[vid_id]["probs"].append(float(prob))

    v_true = []
    v_probs = []
    for vid_id, data in video_map.items():
        mean_p = float(np.mean(data["probs"]))
        v_true.append(data["label"])
        v_probs.append(mean_p)

    v_true = np.array(v_true, dtype=np.float32)
    v_probs = np.array(v_probs, dtype=np.float32)
    metrics = compute_metrics(v_true, v_probs, threshold)
    return metrics, video_map


def run_experiment(
    exp_name: str,
    X_train, y_train, train_meta,
    X_val, y_val, val_meta,
    X_test, y_test, test_meta,
    X_wa, y_wa, wa_meta,
    save_path: str
):
    print("\n" + "="*70)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print("="*70)
    print(f"Train sequences: {len(y_train)} (Real: {int(np.sum(y_train==0))}, Fake: {int(np.sum(y_train==1))})")
    print(f"Val sequences:   {len(y_val)} (Real: {int(np.sum(y_val==0))}, Fake: {int(np.sum(y_val==1))})")
    print(f"Test sequences:  {len(y_test)} (Real: {int(np.sum(y_test==0))}, Fake: {int(np.sum(y_test==1))})")
    print(f"Benchmark seqs:  {len(y_wa)} (Real: {int(np.sum(y_wa==0))}, Fake: {int(np.sum(y_wa==1))})")

    # Compute class weights for balanced gradients
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    weight_0 = (1.0 / neg_count) * (len(y_train) / 2.0) if neg_count > 0 else 1.0
    weight_1 = (1.0 / pos_count) * (len(y_train) / 2.0) if pos_count > 0 else 1.0
    class_weight = {0: weight_0, 1: weight_1}

    model = build_bilstm_temporal_model(sequence_length=T, feature_dim=FEATURE_DIM, learning_rate=LR, name=exp_name)

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=14,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    training_time = time.time() - t0

    # Ensure best model is loaded
    if os.path.exists(save_path):
        best_model = tf.keras.models.load_model(save_path)
    else:
        best_model = model

    # 1. Validation Threshold Calibration (F1 maximization)
    val_probs = best_model.predict(X_val, verbose=0).flatten()
    val_metrics_raw, _ = aggregate_video_level_predictions(val_meta, val_probs, threshold=0.50)
    
    # Calibrate at video level
    video_map_val = {}
    for meta, prob in zip(val_meta, val_probs):
        vid_id = meta["video_id"]
        video_map_val.setdefault(vid_id, {"probs": [], "label": meta["label"]})
        video_map_val[vid_id]["probs"].append(float(prob))
    v_val_true = np.array([v["label"] for v in video_map_val.values()])
    v_val_probs = np.array([np.mean(v["probs"]) for v in video_map_val.values()])

    calibrated_threshold, val_f1 = calibrate_validation_threshold(v_val_true, v_val_probs)
    print(f"[+] Calibrated Validation Threshold: {calibrated_threshold} (Validation F1: {val_f1})")

    # Evaluate validation with calibrated threshold
    val_metrics_cal, _ = aggregate_video_level_predictions(val_meta, val_probs, threshold=calibrated_threshold)

    # 2. Official Held-Out Test Evaluation (Evaluated ONCE with frozen threshold)
    test_probs = best_model.predict(X_test, verbose=0).flatten()
    test_metrics_video, _ = aggregate_video_level_predictions(test_meta, test_probs, threshold=calibrated_threshold)
    test_metrics_seq = compute_metrics(y_test, test_probs, threshold=calibrated_threshold)

    # 3. Quarantined WhatsApp Benchmark Evaluation (Evaluated with frozen threshold)
    wa_probs = best_model.predict(X_wa, verbose=0).flatten()
    wa_metrics_video, _ = aggregate_video_level_predictions(wa_meta, wa_probs, threshold=calibrated_threshold)
    wa_metrics_seq = compute_metrics(y_wa, wa_probs, threshold=calibrated_threshold)

    results = {
        "experiment": exp_name,
        "model_path": save_path,
        "training_time_seconds": round(training_time, 2),
        "epochs_trained": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "calibrated_threshold": calibrated_threshold,
        "validation_metrics_video": val_metrics_cal,
        "official_test_metrics_video": test_metrics_video,
        "official_test_metrics_sequence": test_metrics_seq,
        "whatsapp_benchmark_metrics_video": wa_metrics_video,
        "whatsapp_benchmark_metrics_sequence": wa_metrics_seq,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()}
    }

    print(f"\n--- {exp_name} RESULTS SUMMARY ---")
    print(f"Calibrated Threshold: {calibrated_threshold}")
    print(f"Official Test (n=4 videos): Acc={test_metrics_video['accuracy']*100:.1f}%, F1={test_metrics_video['f1']:.4f}, AUC={test_metrics_video['roc_auc']:.4f}")
    print(f"WhatsApp Benchmark (n=20 videos): Acc={wa_metrics_video['accuracy']*100:.1f}%, F1={wa_metrics_video['f1']:.4f}, AUC={wa_metrics_video['roc_auc']:.4f}")

    return results, best_model


def measure_inference_performance(feature_extractor, temporal_model, sample_video_path: str):
    """
    Measures component and end-to-end inference latencies on the current system (CPU).
    """
    print("\n[+] Measuring inference performance...")
    face_detector = YuNetFaceDetector()

    # 1. Feature extraction latency
    dummy_crop = np.random.uniform(0, 255, (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)).astype(np.float32)
    # Warmup
    _ = feature_extractor(dummy_crop, training=False)
    
    t_feat_start = time.time()
    for _ in range(30):
        _ = feature_extractor(dummy_crop, training=False)
    feat_time_per_frame = (time.time() - t_feat_start) / 30.0

    # 2. BiLSTM temporal inference latency
    dummy_seq = np.random.uniform(-1, 1, (1, T, FEATURE_DIM)).astype(np.float32)
    # Warmup
    _ = temporal_model(dummy_seq, training=False)

    t_bilstm_start = time.time()
    for _ in range(50):
        _ = temporal_model(dummy_seq, training=False)
    bilstm_time = (time.time() - t_bilstm_start) / 50.0

    # 3. End-to-end video inference on actual sample video
    cap = cv2.VideoCapture(sample_video_path)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_f // T)
    
    t_e2e_start = time.time()
    crops = []
    curr = 0
    while cap.isOpened() and len(crops) < T:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
        if curr % step == 0:
            det = face_detector.detect_faces(frame_bgr)
            if det.face_detected and det.crop is not None:
                cr = cv2.resize(cv2.cvtColor(det.crop, cv2.COLOR_BGR2RGB), IMAGE_SIZE)
                crops.append(cr.astype(np.float32))
        curr += 1
    cap.release()

    if len(crops) < T:
        padded = np.zeros((T, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
        for i, c in enumerate(crops):
            padded[i] = c
        seq = padded
    else:
        seq = np.stack(crops[:T], axis=0)

    # Spatial features
    spatial_feats = feature_extractor(seq, training=False)  # (T, 1408)
    spatial_feats = tf.expand_dims(spatial_feats, axis=0)   # (1, T, 1408)
    pred = temporal_model(spatial_feats, training=False)
    total_e2e_time = time.time() - t_e2e_start

    bench_results = {
        "hardware_device": "CPU (Intel/AMD x86_64, Native Windows)",
        "feature_extraction_ms_per_frame": round(feat_time_per_frame * 1000, 2),
        "bilstm_inference_ms_per_sequence": round(bilstm_time * 1000, 2),
        "total_video_inference_seconds": round(total_e2e_time, 3),
        "real_time_capable": bool(total_e2e_time < 3.0)
    }
    print(f"  Feature extraction per frame: {bench_results['feature_extraction_ms_per_frame']} ms")
    print(f"  BiLSTM sequence inference:    {bench_results['bilstm_inference_ms_per_sequence']} ms")
    print(f"  Complete video analysis time: {bench_results['total_video_inference_seconds']} s")
    return bench_results


def generate_training_plots(results_a, results_b, out_path: str):
    """Generates comparison plots of Loss and ROC-AUC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Plot
    axes[0].plot(results_a["history"]["loss"], label="Model A Train Loss (No Aug)", color="#1f77b4", linestyle="--")
    axes[0].plot(results_a["history"]["val_loss"], label="Model A Val Loss (No Aug)", color="#1f77b4")
    axes[0].plot(results_b["history"]["loss"], label="Model B Train Loss (With Aug)", color="#ff7f0e", linestyle="--")
    axes[0].plot(results_b["history"]["val_loss"], label="Model B Val Loss (With Aug)", color="#ff7f0e")
    axes[0].set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # AUC Plot
    axes[1].plot(results_a["history"]["auc"], label="Model A Train AUC", color="#1f77b4", linestyle="--")
    axes[1].plot(results_a["history"]["val_auc"], label="Model A Val AUC", color="#1f77b4")
    axes[1].plot(results_b["history"]["auc"], label="Model B Train AUC", color="#ff7f0e", linestyle="--")
    axes[1].plot(results_b["history"]["val_auc"], label="Model B Val AUC", color="#ff7f0e")
    axes[1].set_title("Training and Validation ROC-AUC", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[+] Saved training plots to: {out_path}")


def main():
    print("="*70)
    print("DEEPSHIELD V5.2 - TEMPORAL VIDEO MODEL TRAINING")
    print("="*70)

    feature_extractor = load_v5_feature_extractor(V5_BASE_MODEL_PATH)
    manifest_path = os.path.join(METADATA_DIR, "v5_2_sequence_manifest.csv")

    # Feature caching paths
    cache_no_aug = os.path.join(V5_2_OUTPUT_DIR, "features_v5_2_no_aug.npz")
    cache_with_aug = os.path.join(V5_2_OUTPUT_DIR, "features_v5_2_with_aug.npz")

    # 1. Prepare Feature Sets for Model A (No Augmentation)
    print("\n[STEP 1] Extracting Features for Model A (No Augmentation)...")
    dataset_no_aug = VideoSequenceDataset(sequence_length=T, use_compression_augmentation=False)
    X_train_a, y_train_a, meta_train_a, X_val_a, y_val_a, meta_val_a, X_test_a, y_test_a, meta_test_a, X_wa_a, y_wa_a, meta_wa_a = extract_features_from_manifest(
        manifest_path, feature_extractor, dataset_no_aug, cache_path=cache_no_aug
    )

    # 2. Prepare Feature Sets for Model B (With Augmentation)
    print("\n[STEP 2] Extracting Features for Model B (With Compression Augmentation)...")
    dataset_with_aug = VideoSequenceDataset(sequence_length=T, use_compression_augmentation=True)
    X_train_b, y_train_b, meta_train_b, X_val_b, y_val_b, meta_val_b, X_test_b, y_test_b, meta_test_b, X_wa_b, y_wa_b, meta_wa_b = extract_features_from_manifest(
        manifest_path, feature_extractor, dataset_with_aug, cache_path=cache_with_aug
    )

    # 3. Train Model A (No Augmentation)
    model_a_path = os.path.join(V5_2_OUTPUT_DIR, "deepshield_v5_2_temporal_no_aug.keras")
    results_a, model_a = run_experiment(
        "DeepShield_V5_2_Model_A_NoAug",
        X_train_a, y_train_a, meta_train_a,
        X_val_a, y_val_a, meta_val_a,
        X_test_a, y_test_a, meta_test_a,
        X_wa_a, y_wa_a, meta_wa_a,
        model_a_path
    )

    # 4. Train Model B (With Compression Augmentation)
    model_b_path = os.path.join(V5_2_OUTPUT_DIR, "deepshield_v5_2_temporal.keras")
    results_b, model_b = run_experiment(
        "DeepShield_V5_2_Model_B_WithAug",
        X_train_b, y_train_b, meta_train_b,
        X_val_b, y_val_b, meta_val_b,
        X_test_b, y_test_b, meta_test_b,
        X_wa_b, y_wa_b, meta_wa_b,
        model_b_path
    )

    # 5. Inference Benchmarking
    sample_vid = os.path.join(workspace, "backend", "test_images", "unseen_videos", "real", "WhatsApp Video 2026-09-15 at 12.49.50 PM.mp4")
    if not os.path.exists(sample_vid):
        sample_vid = os.path.join(workspace, "backend", "test_images", "real_check.mp4")
    inference_metrics = measure_inference_performance(feature_extractor, model_b, sample_vid)

    # 6. Generate Comparison Plots
    plot_path = os.path.join(EVAL_DIR, "v5_2_training_loss_curves.png")
    generate_training_plots(results_a, results_b, plot_path)

    # 7. Save Comprehensive Results JSON
    final_output = {
        "model_a_no_aug": results_a,
        "model_b_with_aug": results_b,
        "inference_benchmarks": inference_metrics,
        "phase1_baseline": {
            "dataset": "WhatsApp Benchmark (n=20)",
            "accuracy": 0.8500,
            "precision": 0.8182,
            "recall": 0.9000,
            "f1": 0.8571,
            "roc_auc": 0.9000
        },
        "v5_1_benchmarks": {
            "official_test_model_a": {"accuracy": 0.7500, "f1": 0.6667, "roc_auc": 0.8000},
            "official_test_model_b": {"accuracy": 0.7500, "f1": 0.6667, "roc_auc": 0.7333},
            "whatsapp_model_a": {"accuracy": 0.7500, "f1": 0.7059, "roc_auc": 0.9200},
            "whatsapp_model_b": {"accuracy": 0.7000, "f1": 0.6250, "roc_auc": 0.8100}
        }
    }

    eval_json_path = os.path.join(EVAL_DIR, "v5_2_evaluation_results.json")
    with open(eval_json_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[+] Saved full evaluation results to: {eval_json_path}")


if __name__ == "__main__":
    main()
