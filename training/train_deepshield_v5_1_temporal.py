"""
DeepShield V5.1 - Dedicated Temporal Video Model Training Pipeline
==================================================================

Trains the EfficientNetB2 + BiLSTM Temporal Deepfake Classifier.

Key Requirements:
- V5 model weights preserved unmodified (models/deepshield_v5_clean_best.keras)
- Sequence length: T = 16
- Video-group split: Strict isolation between Train, Validation, and Test
- Ablation: Model A (without compression aug) vs Model B (with compression aug)
- Calibrated video threshold computed ONLY from Validation set
- Independent evaluation on held-out test set and 20-video WhatsApp set
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix

workspace = r"c:\Users\ADMIN\Desktop\DeepShield_V5_Final"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

from training.video_sequence_dataset import VideoSequenceDataset

# Configuration
T = 16
FEATURE_DIM = 1408
BATCH_SIZE = 8
EPOCHS = 35
LR = 1e-3
V5_MODEL_PATH = os.path.join(workspace, "models", "deepshield_v5_clean_best.keras")
OUTPUT_DIR = os.path.join(workspace, "models", "v5_1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. DEFINE STRICT VIDEO GROUP SPLITS
# -------------------------------------------------------------
# Unseen videos directory:
unseen_real_dir = os.path.join(workspace, "backend", "test_images", "unseen_videos", "real")
unseen_fake_dir = os.path.join(workspace, "backend", "test_images", "unseen_videos", "fake")

unseen_real_files = sorted([os.path.join(unseen_real_dir, f) for f in os.listdir(unseen_real_dir) if f.endswith(".mp4")])
unseen_fake_files = sorted([os.path.join(unseen_fake_dir, f) for f in os.listdir(unseen_fake_dir) if f.endswith(".mp4")])

# Additional test videos
real_check = os.path.join(workspace, "backend", "test_images", "real_check.mp4")
fake_check = os.path.join(workspace, "backend", "test_images", "fake_check.mp4")
test_real = os.path.join(workspace, "backend", "test_images", "test.mp4")
test_real_1 = os.path.join(workspace, "backend", "test_images", "test1.mp4")

# Strict partition by video identity:
# Train: 5 Real, 5 Fake
TRAIN_VIDEOS = [
    (unseen_real_files[0], 0, "unseen_real_0"),
    (unseen_real_files[1], 0, "unseen_real_1"),
    (unseen_real_files[2], 0, "unseen_real_2"),
    (real_check, 0, "real_check"),
    (test_real, 0, "test_real"),
    (unseen_fake_files[0], 1, "unseen_fake_0"),
    (unseen_fake_files[1], 1, "unseen_fake_1"),
    (unseen_fake_files[2], 1, "unseen_fake_2"),
    (unseen_fake_files[3], 1, "unseen_fake_3"),
    (fake_check, 1, "fake_check"),
]

# Validation: 3 Real, 3 Fake (used for early stopping & threshold calibration)
VAL_VIDEOS = [
    (unseen_real_files[3], 0, "unseen_real_3"),
    (unseen_real_files[4], 0, "unseen_real_4"),
    (test_real_1, 0, "test_real_1"),
    (unseen_fake_files[4], 1, "unseen_fake_4"),
    (unseen_fake_files[5], 1, "unseen_fake_5"),
    (unseen_fake_files[6], 1, "unseen_fake_6"),
]

# Official Held-Out Test: 4 Real, 4 Fake (completely isolated from training & validation)
TEST_VIDEOS = [
    (unseen_real_files[5], 0, "unseen_real_5"),
    (unseen_real_files[6], 0, "unseen_real_6"),
    (unseen_real_files[7], 0, "unseen_real_7"),
    (unseen_real_files[8], 0, "unseen_real_8"),
    (unseen_fake_files[7], 1, "unseen_fake_7"),
    (unseen_fake_files[8], 1, "unseen_fake_8"),
    (unseen_fake_files[9], 1, "unseen_fake_9"),
    (unseen_real_files[9], 0, "unseen_real_9"), # 5 real, 3 fake or balance
]

# 20-video WhatsApp Calibration Set (External Generalization Check)
wa_real_dir = os.path.join(workspace, "backend", "test_images", "videos_real")
wa_fake_dir = os.path.join(workspace, "backend", "test_images", "videos_fake")
WHATSAPP_CALIBRATION_VIDEOS = (
    [(os.path.join(wa_real_dir, f), 0, f"wa_real_{i}") for i, f in enumerate(sorted(os.listdir(wa_real_dir))) if f.endswith(".mp4")] +
    [(os.path.join(wa_fake_dir, f), 1, f"wa_fake_{i}") for i, f in enumerate(sorted(os.listdir(wa_fake_dir))) if f.endswith(".mp4")]
)


def build_bilstm_temporal_model(sequence_length=16, feature_dim=1408, learning_rate=1e-3):
    """
    Constructs the V5.1 BiLSTM Temporal Deepfake Classifier.
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
    
    # BiLSTM Layer 2 (64 units, final representation)
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
    
    model = models.Model(inputs=inputs, outputs=outputs, name="DeepShield_V5_1_BiLSTM")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def extract_features_for_video_list(video_list, dataset_builder, feature_extractor, num_windows_per_video=3):
    """
    Extracts multiple sliding-window temporal sequences per video.
    """
    all_features = []
    all_labels = []
    video_records = []
    
    for v_path, label, vid_id in video_list:
        if not os.path.exists(v_path):
            continue
            
        cap = cv2.VideoCapture(v_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Determine start offsets for multiple windows
        offsets = [0]
        if total_frames > 30 and num_windows_per_video >= 2:
            offsets.append(max(5, total_frames // 4))
        if total_frames > 60 and num_windows_per_video >= 3:
            offsets.append(max(10, total_frames // 2))
            
        for offset in offsets:
            seq_tensor, valid_faces, total_sampled, confs = dataset_builder.extract_face_sequence_from_video(
                v_path, start_frame=offset
            )
            if seq_tensor is not None:
                feats = dataset_builder.extract_features_for_sequence(seq_tensor, feature_extractor)
                all_features.append(feats)
                all_labels.append(label)
                video_records.append({
                    "video_path": v_path,
                    "video_id": vid_id,
                    "offset": offset,
                    "label": label,
                    "valid_faces": valid_faces
                })
                
    if not all_features:
        return np.zeros((0, T, FEATURE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
        
    X = np.stack(all_features, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.float32)
    return X, y, video_records


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
    """
    Computes standard academic and forensic verification metrics.
    """
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
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.50
        
    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = 0.50
        
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "real_accuracy": round(real_acc, 4),
        "fake_accuracy": round(fake_acc, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total
    }


def main():
    print("="*80)
    print("DEEPSHIELD V5.1 - TEMPORAL VIDEO MODEL TRAINING & ABLATION")
    print("="*80)
    
    # 1. Load V5 Model & Build Spatial Feature Extractor
    print("\n[1/6] Loading V5 Base Model and constructing Feature Extractor...")
    v5_model = tf.keras.models.load_model(V5_MODEL_PATH)
    # Output of GlobalAveragePooling2D is layer index 2
    feature_extractor = tf.keras.Model(inputs=v5_model.input, outputs=v5_model.layers[2].output)
    feature_extractor.trainable = False
    print("Feature Extractor ready. Output feature dimension:", FEATURE_DIM)
    
    # 2. Extract Data for Ablation:
    # Model A: WITHOUT compression augmentation
    # Model B: WITH compression augmentation
    print("\n[2/6] Extracting Temporal Feature Sequences for Model A (NO Augmentation)...")
    dataset_no_aug = VideoSequenceDataset(sequence_length=T, use_compression_augmentation=False)
    
    t0 = time.time()
    X_train_a, y_train_a, _ = extract_features_for_video_list(TRAIN_VIDEOS, dataset_no_aug, feature_extractor, num_windows_per_video=3)
    X_val, y_val, _ = extract_features_for_video_list(VAL_VIDEOS, dataset_no_aug, feature_extractor, num_windows_per_video=2)
    X_test, y_test, test_recs = extract_features_for_video_list(TEST_VIDEOS, dataset_no_aug, feature_extractor, num_windows_per_video=1)
    print(f"Model A Extraction complete in {time.time()-t0:.1f}s. Train: {X_train_a.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    print("\n[3/6] Extracting Temporal Feature Sequences for Model B (WITH Compression Augmentation)...")
    dataset_with_aug = VideoSequenceDataset(sequence_length=T, use_compression_augmentation=True)
    t0 = time.time()
    X_train_b, y_train_b, _ = extract_features_for_video_list(TRAIN_VIDEOS, dataset_with_aug, feature_extractor, num_windows_per_video=3)
    print(f"Model B Extraction complete in {time.time()-t0:.1f}s. Train: {X_train_b.shape}")
    
    # Extract 20-video WhatsApp Calibration Set (Separately evaluated)
    print("\nExtracting 20-video WhatsApp Calibration Set...")
    X_wa, y_wa, wa_recs = extract_features_for_video_list(WHATSAPP_CALIBRATION_VIDEOS, dataset_no_aug, feature_extractor, num_windows_per_video=1)
    print(f"WhatsApp Benchmark Extraction complete. Shape: {X_wa.shape}")
    
    # 3. Train Model A (Without Compression Augmentation)
    print("\n[4/6] Training Model A (EfficientNetB2 + BiLSTM - NO Augmentation)...")
    model_a = build_bilstm_temporal_model(sequence_length=T, feature_dim=FEATURE_DIM, learning_rate=LR)
    
    cb_a = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=12, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-5)
    ]
    t_train_a0 = time.time()
    hist_a = model_a.fit(
        X_train_a, y_train_a,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_a,
        verbose=1
    )
    time_train_a = round(time.time() - t_train_a0, 2)
    model_a_path = os.path.join(OUTPUT_DIR, "deepshield_v5_1_temporal_no_aug.keras")
    model_a.save(model_a_path)
    print(f"Model A saved to {model_a_path} (Training time: {time_train_a}s)")
    
    # 4. Train Model B (With Compression Augmentation)
    print("\n[5/6] Training Model B (EfficientNetB2 + BiLSTM - WITH Compression Augmentation)...")
    model_b = build_bilstm_temporal_model(sequence_length=T, feature_dim=FEATURE_DIM, learning_rate=LR)
    
    cb_b = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=12, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-5)
    ]
    t_train_b0 = time.time()
    hist_b = model_b.fit(
        X_train_b, y_train_b,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_b,
        verbose=1
    )
    time_train_b = round(time.time() - t_train_b0, 2)
    model_b_path = os.path.join(OUTPUT_DIR, "deepshield_v5_1_temporal.keras")
    model_b.save(model_b_path)
    print(f"Model B saved to {model_b_path} (Training time: {time_train_b}s)")
    
    # 5. Calibration & Evaluation
    print("\n[6/6] Threshold Calibration & Multi-Dataset Evaluation...")
    
    # Determine optimal threshold ONLY on validation set
    val_probs_a = model_a.predict(X_val).flatten()
    val_thresh_a, val_f1_a = calibrate_validation_threshold(y_val, val_probs_a)
    
    val_probs_b = model_b.predict(X_val).flatten()
    val_thresh_b, val_f1_b = calibrate_validation_threshold(y_val, val_probs_b)
    
    print(f"\nValidation Optimal Thresholds:")
    print(f"  Model A: {val_thresh_a} (Val F1: {val_f1_a})")
    print(f"  Model B: {val_thresh_b} (Val F1: {val_f1_b})")
    
    # Evaluate on Held-Out Test Set
    t_inf0 = time.time()
    test_probs_a = model_a.predict(X_test).flatten()
    inf_time_per_seq = round((time.time() - t_inf0) / len(X_test), 4) if len(X_test) > 0 else 0.0
    test_metrics_a = compute_metrics(y_test, test_probs_a, threshold=val_thresh_a)
    
    test_probs_b = model_b.predict(X_test).flatten()
    test_metrics_b = compute_metrics(y_test, test_probs_b, threshold=val_thresh_b)
    
    # Evaluate on WhatsApp Calibration Benchmark (External Generalization Check)
    wa_probs_a = model_a.predict(X_wa).flatten()
    wa_metrics_a = compute_metrics(y_wa, wa_probs_a, threshold=val_thresh_a)
    
    wa_probs_b = model_b.predict(X_wa).flatten()
    wa_metrics_b = compute_metrics(y_wa, wa_probs_b, threshold=val_thresh_b)
    
    results_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "CPU (AVX2/FMA Optimized)",
        "feature_dim": FEATURE_DIM,
        "sequence_length": T,
        "inference_time_per_sequence_sec": inf_time_per_seq,
        "splits": {
            "train_video_count": len(TRAIN_VIDEOS),
            "train_sequences_a": len(X_train_a),
            "train_sequences_b": len(X_train_b),
            "val_video_count": len(VAL_VIDEOS),
            "val_sequences": len(X_val),
            "test_video_count": len(TEST_VIDEOS),
            "test_sequences": len(X_test),
            "whatsapp_benchmark_videos": len(WHATSAPP_CALIBRATION_VIDEOS)
        },
        "model_a_no_aug": {
            "model_path": model_a_path,
            "training_time_sec": time_train_a,
            "val_optimal_threshold": val_thresh_a,
            "val_f1": val_f1_a,
            "held_out_test": test_metrics_a,
            "whatsapp_benchmark": wa_metrics_a
        },
        "model_b_with_aug": {
            "model_path": model_b_path,
            "training_time_sec": time_train_b,
            "val_optimal_threshold": val_thresh_b,
            "val_f1": val_f1_b,
            "held_out_test": test_metrics_b,
            "whatsapp_benchmark": wa_metrics_b
        }
    }
    
    out_json = os.path.join(workspace, "docs", "v5_1_ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
        
    print("\n" + "="*80)
    print("RESULTS COMPARISON:")
    print("="*80)
    print("HELD-OUT OFFICIAL TEST SET:")
    print(f"  Model A (No Aug):   Acc: {test_metrics_a['accuracy']*100:.1f}%, F1: {test_metrics_a['f1']}, AUC: {test_metrics_a['roc_auc']}")
    print(f"  Model B (With Aug): Acc: {test_metrics_b['accuracy']*100:.1f}%, F1: {test_metrics_b['f1']}, AUC: {test_metrics_b['roc_auc']}")
    print("\nWHATSAPP 20-VIDEO CALIBRATION/GENERALIZATION SET:")
    print(f"  Model A (No Aug):   Acc: {wa_metrics_a['accuracy']*100:.1f}%, F1: {wa_metrics_a['f1']}, AUC: {wa_metrics_a['roc_auc']}")
    print(f"  Model B (With Aug): Acc: {wa_metrics_b['accuracy']*100:.1f}%, F1: {wa_metrics_b['f1']}, AUC: {wa_metrics_b['roc_auc']}")
    print(f"\nSaved full results to {out_json}")

if __name__ == "__main__":
    main()
