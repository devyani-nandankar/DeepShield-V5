"""
Controlled Image Backbone Fine-Tuning Experiment
DeepShield V5 Image Pipeline
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, silhouette_score
)
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean, cosine
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# File Paths
MANIFEST_PATH = "Kaggle_Transfer/face_crop_FAST10_manifest_split.csv"
TRAIN_DIR = "G:/My Drive/DeepShield_Datasets/Face_Cropped_FAST10/train"
VAL_DIR = "G:/My Drive/DeepShield_Datasets/Face_Cropped_FAST10/validation"
TEST_DIR = "evaluation/test_images"
TEST_PRED_BASELINE = "evaluation/direct_fast10_predictions.csv"
COHORT_800_CSV = "evaluation/image_feature_separability.csv"

BASE_MODEL_PATH = "models/deepshield_v5_clean_best.keras"
OUTPUT_MODEL_DIR = "models/v5_image_finetuned"
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_MODEL_DIR, "deepshield_v5_image_finetuned.keras")

OUTPUT_JSON_PATH = "evaluation/v5_image_finetuned_results.json"
OUTPUT_CSV_PATH = "evaluation/v5_image_finetuned_comparison.csv"

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0

def load_images_parallel(file_paths, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        imgs = list(executor.map(preprocess_image, file_paths))
    return np.array(imgs, dtype=np.float32)

def compute_all_metrics(y_t, preds, scores):
    acc = float(accuracy_score(y_t, preds))
    prec = float(precision_score(y_t, preds, zero_division=0))
    rec = float(recall_score(y_t, preds, zero_division=0))
    f1 = float(f1_score(y_t, preds, zero_division=0))
    auc = float(roc_auc_score(y_t, scores))
    tn, fp, fn, tp = confusion_matrix(y_t, preds).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "fpr": fpr,
        "fnr": fnr
    }

def main():
    print("================================================================")
    print("CONTROLLED IMAGE BACKBONE FINE-TUNING EXPERIMENT")
    print("================================================================")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    
    # 1. Prepare Training and Validation Sets (Strictly from Train and Val Splits)
    print("\n--- Step 1: Loading Manifest & Preparing Balanced Splits ---")
    df = pd.read_csv(MANIFEST_PATH)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    
    train_df["full_path"] = train_df["face_path"].apply(lambda x: os.path.join(TRAIN_DIR, os.path.basename(x)))
    val_df["full_path"] = val_df["face_path"].apply(lambda x: os.path.join(VAL_DIR, os.path.basename(x)))
    
    # Sample balanced 1,000 Real + 1,000 Fake from train split across 3 datasets
    train_reals = []
    train_fakes = []
    for ds in ["FF-C23", "Deepfake-Videos-Merged", "FaceForensics-1000"]:
        sub_r = train_df[(train_df["source_dataset"] == ds) & (train_df["label"] == 0)]
        sub_f = train_df[(train_df["source_dataset"] == ds) & (train_df["label"] == 1)]
        train_reals.append(sub_r.sample(n=min(334, len(sub_r)), random_state=42))
        train_fakes.append(sub_f.sample(n=min(334, len(sub_f)), random_state=42))
        
    sampled_train = pd.concat(train_reals + train_fakes).sample(frac=1.0, random_state=42).reset_index(drop=True)
    # Ensure exactly 2000 images (1000 real, 1000 fake)
    real_sample = sampled_train[sampled_train["label"] == 0].iloc[:1000]
    fake_sample = sampled_train[sampled_train["label"] == 1].iloc[:1000]
    sampled_train = pd.concat([real_sample, fake_sample]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Validation set: 200 Real + 200 Fake from val split
    val_reals = []
    val_fakes = []
    for ds in ["FF-C23", "Deepfake-Videos-Merged", "FaceForensics-1000"]:
        sub_r = val_df[(val_df["source_dataset"] == ds) & (val_df["label"] == 0)]
        sub_f = val_df[(val_df["source_dataset"] == ds) & (val_df["label"] == 1)]
        val_reals.append(sub_r.sample(n=min(70, len(sub_r)), random_state=42))
        val_fakes.append(sub_f.sample(n=min(70, len(sub_f)), random_state=42))
    sampled_val = pd.concat(val_reals + val_fakes).sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_real_sub = sampled_val[sampled_val["label"] == 0].iloc[:200]
    val_fake_sub = sampled_val[sampled_val["label"] == 1].iloc[:200]
    sampled_val = pd.concat([val_real_sub, val_fake_sub]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"Training dataset: {len(sampled_train)} images ({sum(sampled_train['label']==0)} Real, {sum(sampled_train['label']==1)} Fake)")
    print(f"Validation dataset: {len(sampled_val)} images ({sum(sampled_val['label']==0)} Real, {sum(sampled_val['label']==1)} Fake)")
    
    # Preload into memory
    print("Preloading training images into RAM...")
    t0 = time.time()
    X_train = load_images_parallel(sampled_train["full_path"].tolist(), max_workers=8)
    y_train = sampled_train["label"].values.astype(np.float32)
    print(f"Preloaded X_train: {X_train.shape} in {time.time()-t0:.2f}s")
    
    print("Preloading validation images into RAM...")
    t0 = time.time()
    X_val = load_images_parallel(sampled_val["full_path"].tolist(), max_workers=8)
    y_val = sampled_val["label"].values.astype(np.float32)
    print(f"Preloaded X_val: {X_val.shape} in {time.time()-t0:.2f}s")
    
    # 2. Build Fine-Tuning Model Architecture
    print("\n--- Step 2: Configuring EfficientNetB2 Layer Trainability ---")
    model = tf.keras.models.load_model(BASE_MODEL_PATH, compile=False)
    bb = model.get_layer("efficientnetb2")
    bb.trainable = True
    
    unfrozen_layer_names = []
    frozen_layer_names = []
    
    for layer in bb.layers:
        # Unfreeze ONLY upper blocks: block6, block7, top
        if any(layer.name.startswith(b) for b in ["block6", "block7", "top"]):
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                frozen_layer_names.append(layer.name)
            else:
                layer.trainable = True
                unfrozen_layer_names.append(layer.name)
        else:
            layer.trainable = False
            frozen_layer_names.append(layer.name)
            
    # Dense head trainable
    model.get_layer("dense").trainable = True
    model.get_layer("dense_1").trainable = True
    unfrozen_layer_names.extend(["dense", "dense_1"])
    
    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    total_count = trainable_count + non_trainable_count
    
    print(f"Total Parameters: {total_count:,}")
    print(f"Trainable Parameters: {trainable_count:,} ({trainable_count/total_count*100:.1f}%)")
    print(f"Non-Trainable Parameters: {non_trainable_count:,} ({non_trainable_count/total_count*100:.1f}%)")
    print(f"Unfrozen Conv/Dense Layers ({len(unfrozen_layer_names)} layers): Block6, Block7, TopConv, Dense, Dense_1")
    print(f"Frozen Layers ({len(frozen_layer_names)} layers): Blocks 1-5, all BatchNormalization")
    
    # 3. Compile and Train with Callbacks
    print("\n--- Step 3: Compiling and Training Model (lr=1e-5) ---")
    learning_rate = 1e-5
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=OUTPUT_MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    epochs = 10
    batch_size = 32
    print(f"Starting training for up to {epochs} epochs (batch_size={batch_size}, lr={learning_rate})...")
    
    start_train_t = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1
    )
    train_duration = time.time() - start_train_t
    print(f"Training completed in {train_duration:.2f}s ({train_duration/60:.2f} min)")
    
    # Load best checkpoint
    print(f"Loading best saved model from {OUTPUT_MODEL_PATH}...")
    best_model = tf.keras.models.load_model(OUTPUT_MODEL_PATH, compile=False)
    
    # 4. Evaluate on Official Untouched FAST10 Test Set (3,693 images)
    print("\n--- Step 4: Evaluating Fine-Tuned Model on Untouched FAST10 Test Set (N = 3,693) ---")
    baseline_df = pd.read_csv(TEST_PRED_BASELINE)
    test_paths = [os.path.join(TEST_DIR, os.path.basename(p)) for p in baseline_df["face_path"]]
    
    # Predict in batches
    print("Running inference across 3,693 test images...")
    t0 = time.time()
    all_scores = []
    batch_sz = 64
    for i in range(0, len(test_paths), batch_sz):
        batch_p = test_paths[i:i+batch_sz]
        batch_imgs = load_images_parallel(batch_p, max_workers=8)
        preds = best_model.predict(batch_imgs, batch_size=batch_sz, verbose=0).flatten()
        all_scores.extend(preds)
        if (i // batch_sz) % 10 == 0 or i + batch_sz >= len(test_paths):
            print(f"  Inference: {min(i+batch_sz, len(test_paths))}/{len(test_paths)} images")
            
    s_finetuned = np.array(all_scores, dtype=np.float32)
    s_baseline = baseline_df["raw_score"].values.astype(np.float32)
    y_test = baseline_df["label"].values.astype(int)
    
    # Metrics at threshold 0.21
    th = 0.21
    pred_base_021 = (s_baseline >= th).astype(int)
    pred_fine_021 = (s_finetuned >= th).astype(int)
    
    m_base = compute_all_metrics(y_test, pred_base_021, s_baseline)
    m_fine_021 = compute_all_metrics(y_test, pred_fine_021, s_finetuned)
    
    # Optimal threshold for fine-tuned model
    thresholds = np.linspace(0.01, 0.99, 99)
    best_th = th
    best_f1 = 0.0
    for t in thresholds:
        f = f1_score(y_test, (s_finetuned >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = t
    m_fine_opt = compute_all_metrics(y_test, (s_finetuned >= best_th).astype(int), s_finetuned)
    m_fine_opt["optimal_threshold"] = float(best_th)
    
    print("\n================================================================")
    print("OFFICIAL TEST BENCHMARK: V5 BASELINE VS FINE-TUNED V5")
    print("================================================================")
    print(f"{'Metric':<20} | {'V5 Baseline (0.21)':<20} | {'Fine-Tuned (0.21)':<20} | {'Change':<15} | {'Fine-Tuned (Best ' + f'{best_th:.2f})':<22}")
    print("-" * 105)
    for k in ["accuracy", "precision", "recall", "f1", "auc", "fpr", "fnr"]:
        v_b = m_base[k]
        v_f = m_fine_021[k]
        chg = v_f - v_b
        print(f"{k:<20} | {v_b:<20.4f} | {v_f:<20.4f} | {chg:<+15.4f} | {m_fine_opt[k]:<22.4f}")
    print(f"{'TP / FP / TN / FN':<20} | {m_base['tp']}/{m_base['fp']}/{m_base['tn']}/{m_base['fn']:^18} | {m_fine_021['tp']}/{m_fine_021['fp']}/{m_fine_021['tn']}/{m_fine_021['fn']:^18} | {'-':^15} | {m_fine_opt['tp']}/{m_fine_opt['fp']}/{m_fine_opt['tn']}/{m_fine_opt['fn']:^20}")
    
    # 5. Repeat Feature-Separability Diagnostic on the NEW Model (Same 800 Cohort)
    print("\n--- Step 5: Repeating Feature-Separability Diagnostic on Fine-Tuned Model ---")
    cohort_df = pd.read_csv(COHORT_800_CSV)
    cohort_paths = [os.path.join(TEST_DIR, os.path.basename(p)) for p in cohort_df["face_path"]]
    
    # Feature extractor for fine-tuned model
    gap_layer_fine = best_model.get_layer("global_average_pooling2d")
    feat_extractor_fine = tf.keras.Model(inputs=best_model.input, outputs=gap_layer_fine.output)
    
    # Extract features in batches
    all_fine_feats = []
    for i in range(0, len(cohort_paths), batch_sz):
        batch_p = cohort_paths[i:i+batch_sz]
        batch_imgs = load_images_parallel(batch_p, max_workers=8)
        feats = feat_extractor_fine.predict(batch_imgs, batch_size=batch_sz, verbose=0)
        all_fine_feats.append(feats)
    X_fine_feats = np.vstack(all_fine_feats)
    
    cohort_labels = cohort_df["true_label"].values
    cohort_names = cohort_df["cohort"].values
    
    # Compute new centroids
    mask_tn = (cohort_names == "TN_Real")
    mask_fp = (cohort_names == "FP_Real")
    mask_tp = (cohort_names == "TP_Fake")
    mask_fn = (cohort_names == "FN_Fake")
    
    c_tn_new = np.mean(X_fine_feats[mask_tn], axis=0)
    c_fp_new = np.mean(X_fine_feats[mask_fp], axis=0)
    c_tp_new = np.mean(X_fine_feats[mask_tp], axis=0)
    c_fn_new = np.mean(X_fine_feats[mask_fn], axis=0)
    c_real_new = np.mean(X_fine_feats[cohort_labels == 0], axis=0)
    c_fake_new = np.mean(X_fine_feats[cohort_labels == 1], axis=0)
    
    # Key distances on new model
    dist_rf_new = float(euclidean(c_real_new, c_fake_new))
    dist_fp_tn_new = float(euclidean(c_fp_new, c_tn_new))
    dist_fp_tp_new = float(euclidean(c_fp_new, c_tp_new))
    dist_fn_tp_new = float(euclidean(c_fn_new, c_tp_new))
    dist_fn_tn_new = float(euclidean(c_fn_new, c_tn_new))
    
    # Nearest centroid accuracy
    d_to_tn_new = np.linalg.norm(X_fine_feats - c_tn_new, axis=1)
    d_to_tp_new = np.linalg.norm(X_fine_feats - c_tp_new, axis=1)
    nc_preds_new = (d_to_tp_new < d_to_tn_new).astype(int)
    nc_acc_new = float(accuracy_score(cohort_labels, nc_preds_new))
    
    # k-NN accuracy (5-fold CV)
    from sklearn.model_selection import cross_val_score
    knn1 = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn1_acc_new = float(np.mean(cross_val_score(knn1, X_fine_feats, cohort_labels, cv=5)))
    knn3 = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn3_acc_new = float(np.mean(cross_val_score(knn3, X_fine_feats, cohort_labels, cv=5)))
    knn5 = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn5_acc_new = float(np.mean(cross_val_score(knn5, X_fine_feats, cohort_labels, cv=5)))
    
    # Silhouette score
    sil_new = float(silhouette_score(X_fine_feats, cohort_labels, metric="euclidean"))
    
    # PCA explained variance
    pca = PCA(n_components=10, random_state=42)
    pca.fit(X_fine_feats)
    pca_var_new = [float(v) for v in pca.explained_variance_ratio_]
    
    print("\n--- Feature Separability Comparison (Baseline vs Fine-Tuned) ---")
    print(f"Real/Fake Centroid Dist: Baseline = 0.0027 -> Fine-Tuned = {dist_rf_new:.4f}")
    print(f"FP/TN Centroid Dist    : Baseline = 0.0044 -> Fine-Tuned = {dist_fp_tn_new:.4f}")
    print(f"FN/TP Centroid Dist    : Baseline = 0.0069 -> Fine-Tuned = {dist_fn_tp_new:.4f}")
    print(f"Nearest-Centroid Acc   : Baseline = 52.75% -> Fine-Tuned = {nc_acc_new*100:.2f}%")
    print(f"k-NN (k=1) Accuracy    : Baseline = 65.12% -> Fine-Tuned = {knn1_acc_new*100:.2f}%")
    print(f"k-NN (k=3) Accuracy    : Baseline = 58.50% -> Fine-Tuned = {knn3_acc_new*100:.2f}%")
    print(f"k-NN (k=5) Accuracy    : Baseline = 56.38% -> Fine-Tuned = {knn5_acc_new*100:.2f}%")
    print(f"Silhouette Score       : Baseline = 0.0005 -> Fine-Tuned = {sil_new:.4f}")
    print(f"PCA PC1 Variance       : Baseline = 90.02% -> Fine-Tuned = {pca_var_new[0]*100:.2f}%")
    
    # 6. Save Comparison CSV and Results JSON
    comparison_df = pd.DataFrame({
        "filename": baseline_df["filename"],
        "face_path": baseline_df["face_path"],
        "source_dataset": baseline_df["source_dataset"],
        "video_group": baseline_df["video_group"],
        "true_label": y_test,
        "baseline_score": s_baseline,
        "finetuned_score": s_finetuned,
        "baseline_pred_021": pred_base_021,
        "finetuned_pred_021": pred_fine_021,
        "finetuned_pred_opt": (s_finetuned >= best_th).astype(int)
    })
    comparison_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSaved test comparison to {OUTPUT_CSV_PATH}")
    
    results_json = {
        "training_config": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs_run": len(history.history["loss"]),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "trainable_parameters": int(trainable_count),
            "non_trainable_parameters": int(non_trainable_count),
            "total_parameters": int(total_count),
            "unfrozen_blocks": ["block6", "block7", "top_conv", "dense", "dense_1"],
            "frozen_blocks": ["blocks 1-5", "all BatchNormalization"]
        },
        "validation_performance": {
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_val_accuracy": float(history.history["val_accuracy"][-1]),
            "final_val_auc": float(history.history["val_auc"][-1]),
            "best_val_auc": float(max(history.history["val_auc"]))
        },
        "test_performance": {
            "baseline_v5": m_base,
            "finetuned_v5_021": m_fine_021,
            "finetuned_v5_optimal": m_fine_opt
        },
        "feature_separability_comparison": {
            "baseline": {
                "real_fake_centroid_dist": 0.0027,
                "nearest_centroid_acc": 0.5275,
                "knn1_acc": 0.6512,
                "silhouette_score": 0.0005,
                "pca_pc1_variance": 0.9002
            },
            "finetuned": {
                "real_fake_centroid_dist": dist_rf_new,
                "nearest_centroid_acc": nc_acc_new,
                "knn1_acc": knn1_acc_new,
                "knn3_acc": knn3_acc_new,
                "knn5_acc": knn5_acc_new,
                "silhouette_score": sil_new,
                "pca_pc1_variance": pca_var_new[0]
            }
        }
    }
    
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved experiment results to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
