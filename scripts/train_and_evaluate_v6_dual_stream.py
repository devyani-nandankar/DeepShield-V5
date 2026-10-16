"""
Dual-Stream Spatial + Forensic Image Model Experiment (DeepShield V6)
Strict Scope: Image Pipeline Only. Freeze Video Pipeline.
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
OUTPUT_MODEL_DIR = "models/v6_dual_stream"
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_MODEL_DIR, "deepshield_v6_dual_stream.keras")

OUTPUT_JSON_PATH = "evaluation/v6_dual_stream_results.json"
OUTPUT_CSV_PATH = "evaluation/v6_dual_stream_comparison.csv"

def preprocess_single_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_spatial = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    
    # Forensic Stream: High-frequency residual = image - GaussianBlur(image)
    blurred = cv2.GaussianBlur(img_spatial, (5, 5), sigmaX=1.0)
    residual = img_spatial - blurred
    # Safe normalization: bounded scaling to [-1.0, 1.0] preserving relative energy
    img_forensic = np.clip(residual * 4.0, -1.0, 1.0).astype(np.float32)
    
    return img_spatial, img_forensic

def load_images_parallel(file_paths, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(preprocess_single_image, file_paths))
    spatial_list = [r[0] for r in results]
    forensic_list = [r[1] for r in results]
    return np.array(spatial_list, dtype=np.float32), np.array(forensic_list, dtype=np.float32)

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
    print("DUAL-STREAM SPATIAL + FORENSIC IMAGE MODEL EXPERIMENT (V6)")
    print("================================================================")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    
    # 1. Load Manifest & Prepare Balanced Splits
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
    print("Preloading dual-stream training images into RAM...")
    t0 = time.time()
    X_train_spatial, X_train_forensic = load_images_parallel(sampled_train["full_path"].tolist(), max_workers=8)
    y_train = sampled_train["label"].values.astype(np.float32)
    print(f"Preloaded Train: Spatial {X_train_spatial.shape}, Forensic {X_train_forensic.shape} in {time.time()-t0:.2f}s")
    
    print("Preloading dual-stream validation images into RAM...")
    t0 = time.time()
    X_val_spatial, X_val_forensic = load_images_parallel(sampled_val["full_path"].tolist(), max_workers=8)
    y_val = sampled_val["label"].values.astype(np.float32)
    print(f"Preloaded Val: Spatial {X_val_spatial.shape}, Forensic {X_val_forensic.shape} in {time.time()-t0:.2f}s")
    
    # 2. Build Dual-Stream Architecture
    print("\n--- Step 2: Constructing Dual-Stream Model Architecture ---")
    base_model = tf.keras.models.load_model(BASE_MODEL_PATH, compile=False)
    base_backbone = base_model.get_layer("efficientnetb2")
    base_backbone.trainable = False  # Initially freeze backbone
    
    spatial_input = tf.keras.Input(shape=(224, 224, 3), name="spatial_input")
    forensic_input = tf.keras.Input(shape=(224, 224, 3), name="forensic_input")
    
    # Stream 1: Spatial Stream
    x_spatial = base_backbone(spatial_input)
    spatial_features = tf.keras.layers.GlobalAveragePooling2D(name="spatial_gap")(x_spatial)
    
    # Stream 2: Forensic/Frequency Stream
    x_forensic = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="forensic_conv1")(forensic_input)
    x_forensic = tf.keras.layers.BatchNormalization(name="forensic_bn1")(x_forensic)
    x_forensic = tf.keras.layers.MaxPooling2D((2, 2), name="forensic_pool1")(x_forensic)
    
    x_forensic = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="forensic_conv2")(x_forensic)
    x_forensic = tf.keras.layers.BatchNormalization(name="forensic_bn2")(x_forensic)
    x_forensic = tf.keras.layers.MaxPooling2D((2, 2), name="forensic_pool2")(x_forensic)
    
    x_forensic = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="forensic_conv3")(x_forensic)
    x_forensic = tf.keras.layers.BatchNormalization(name="forensic_bn3")(x_forensic)
    forensic_features = tf.keras.layers.GlobalAveragePooling2D(name="forensic_gap")(x_forensic)
    
    # Fusion Layer
    fused_features = tf.keras.layers.Concatenate(name="fusion_concat")([spatial_features, forensic_features])
    x = tf.keras.layers.Dense(256, activation="relu", name="fusion_dense")(fused_features)
    x = tf.keras.layers.Dropout(0.4, name="fusion_dropout")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="fusion_output")(x)
    
    v6_model = tf.keras.Model(inputs=[spatial_input, forensic_input], outputs=output, name="deepshield_v6_dual_stream")
    
    trainable_count = sum(tf.keras.backend.count_params(w) for w in v6_model.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w) for w in v6_model.non_trainable_weights)
    total_count = trainable_count + non_trainable_count
    
    print(f"Total Parameters: {total_count:,}")
    print(f"Trainable Parameters: {trainable_count:,} ({trainable_count/total_count*100:.1f}%)")
    print(f"Non-Trainable Parameters: {non_trainable_count:,} ({non_trainable_count/total_count*100:.1f}%)")
    
    # 3. Compile and Train Model
    print("\n--- Step 3: Compiling and Training V6 Dual-Stream (lr=1e-4) ---")
    learning_rate = 1e-4
    v6_model.compile(
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
        patience=4,
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
    
    epochs = 12
    batch_size = 32
    print(f"Training up to {epochs} epochs (batch_size={batch_size}, lr={learning_rate})...")
    
    start_train_t = time.time()
    history = v6_model.fit(
        [X_train_spatial, X_train_forensic], y_train,
        validation_data=([X_val_spatial, X_val_forensic], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1
    )
    train_duration = time.time() - start_train_t
    print(f"Training completed in {train_duration:.2f}s ({train_duration/60:.2f} min)")
    
    # Load best checkpoint
    print(f"\nLoading best checkpoint from {OUTPUT_MODEL_PATH}...")
    best_model = tf.keras.models.load_model(OUTPUT_MODEL_PATH, compile=False)
    
    # 4. Evaluate on Official Untouched FAST10 Test Set (3,693 images)
    print("\n--- Step 4: Evaluating V6 Dual-Stream on FAST10 Test Set (N = 3,693) ---")
    baseline_df = pd.read_csv(TEST_PRED_BASELINE)
    test_paths = [os.path.join(TEST_DIR, os.path.basename(p)) for p in baseline_df["face_path"]]
    
    all_scores = []
    batch_sz = 64
    t0 = time.time()
    for i in range(0, len(test_paths), batch_sz):
        batch_p = test_paths[i:i+batch_sz]
        batch_spatial, batch_forensic = load_images_parallel(batch_p, max_workers=8)
        preds = best_model.predict([batch_spatial, batch_forensic], batch_size=batch_sz, verbose=0).flatten()
        all_scores.extend(preds)
        if (i // batch_sz) % 10 == 0 or i + batch_sz >= len(test_paths):
            print(f"  Test Inference: {min(i+batch_sz, len(test_paths))}/{len(test_paths)} images")
            
    s_v6 = np.array(all_scores, dtype=np.float32)
    s_baseline = baseline_df["raw_score"].values.astype(np.float32)
    y_test = baseline_df["label"].values.astype(int)
    
    # Metrics at baseline threshold 0.21
    th = 0.21
    pred_base_021 = (s_baseline >= th).astype(int)
    pred_v6_021 = (s_v6 >= th).astype(int)
    
    m_base = compute_all_metrics(y_test, pred_base_021, s_baseline)
    m_v6_021 = compute_all_metrics(y_test, pred_v6_021, s_v6)
    
    # Optimal threshold for V6 model
    thresholds = np.linspace(0.01, 0.99, 99)
    best_th = th
    best_f1 = 0.0
    for t in thresholds:
        f = f1_score(y_test, (s_v6 >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = t
    m_v6_opt = compute_all_metrics(y_test, (s_v6 >= best_th).astype(int), s_v6)
    m_v6_opt["optimal_threshold"] = float(best_th)
    
    print("\n================================================================")
    print("OFFICIAL TEST BENCHMARK: V5 BASELINE VS V6 DUAL-STREAM")
    print("================================================================")
    print(f"{'Metric':<20} | {'V5 Baseline (0.21)':<20} | {'V6 Dual-Stream (0.21)':<22} | {'Change':<15} | {'V6 (Best ' + f'{best_th:.2f})':<20}")
    print("-" * 105)
    for k in ["accuracy", "precision", "recall", "f1", "auc", "fpr", "fnr"]:
        v_b = m_base[k]
        v_v6 = m_v6_021[k]
        chg = v_v6 - v_b
        print(f"{k:<20} | {v_b:<20.4f} | {v_v6:<22.4f} | {chg:<+15.4f} | {m_v6_opt[k]:<20.4f}")
    print(f"{'TP / FP / TN / FN':<20} | {m_base['tp']}/{m_base['fp']}/{m_base['tn']}/{m_base['fn']:^18} | {m_v6_021['tp']}/{m_v6_021['fp']}/{m_v6_021['tn']}/{m_v6_021['fn']:^20} | {'-':^15} | {m_v6_opt['tp']}/{m_v6_opt['fp']}/{m_v6_opt['tn']}/{m_v6_opt['fn']:^18}")
    
    # 5. Feature Separability Analysis on the 800-Image Diagnostic Cohort
    print("\n--- Step 5: Feature Separability Analysis on the 800 Diagnostic Cohort ---")
    cohort_df = pd.read_csv(COHORT_800_CSV)
    cohort_paths = [os.path.join(TEST_DIR, os.path.basename(p)) for p in cohort_df["face_path"]]
    
    # Feature extractor for V6 model (fusion_dense: 256-d latent representation)
    fusion_dense_layer = best_model.get_layer("fusion_dense")
    feat_extractor_v6 = tf.keras.Model(inputs=best_model.inputs, outputs=fusion_dense_layer.output)
    
    all_v6_feats = []
    for i in range(0, len(cohort_paths), batch_sz):
        batch_p = cohort_paths[i:i+batch_sz]
        batch_s, batch_f = load_images_parallel(batch_p, max_workers=8)
        feats = feat_extractor_v6.predict([batch_s, batch_f], batch_size=batch_sz, verbose=0)
        all_v6_feats.append(feats)
    X_v6_feats = np.vstack(all_v6_feats)
    
    cohort_labels = cohort_df["label"].values
    cohort_names = cohort_df["cohort"].values
    
    # Centroids
    mask_tn = (cohort_names == "TN_Real")
    mask_fp = (cohort_names == "FP_Real")
    mask_tp = (cohort_names == "TP_Fake")
    mask_fn = (cohort_names == "FN_Fake")
    
    c_tn = np.mean(X_v6_feats[mask_tn], axis=0)
    c_fp = np.mean(X_v6_feats[mask_fp], axis=0)
    c_tp = np.mean(X_v6_feats[mask_tp], axis=0)
    c_fn = np.mean(X_v6_feats[mask_fn], axis=0)
    c_real = np.mean(X_v6_feats[cohort_labels == 0], axis=0)
    c_fake = np.mean(X_v6_feats[cohort_labels == 1], axis=0)
    
    dist_rf = float(euclidean(c_real, c_fake))
    dist_fp_tn = float(euclidean(c_fp, c_tn))
    dist_fp_tp = float(euclidean(c_fp, c_tp))
    dist_fn_tp = float(euclidean(c_fn, c_tp))
    dist_fn_tn = float(euclidean(c_fn, c_tn))
    
    # Nearest centroid accuracy
    d_to_tn = np.linalg.norm(X_v6_feats - c_tn, axis=1)
    d_to_tp = np.linalg.norm(X_v6_feats - c_tp, axis=1)
    nc_preds = (d_to_tp < d_to_tn).astype(int)
    nc_acc = float(accuracy_score(cohort_labels, nc_preds))
    
    # k-NN accuracy (5-fold CV)
    from sklearn.model_selection import cross_val_score
    knn1 = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn1_acc = float(np.mean(cross_val_score(knn1, X_v6_feats, cohort_labels, cv=5)))
    knn3 = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn3_acc = float(np.mean(cross_val_score(knn3, X_v6_feats, cohort_labels, cv=5)))
    knn5 = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn5_acc = float(np.mean(cross_val_score(knn5, X_v6_feats, cohort_labels, cv=5)))
    
    sil = float(silhouette_score(X_v6_feats, cohort_labels, metric="euclidean"))
    
    pca = PCA(n_components=10, random_state=42)
    pca.fit(X_v6_feats)
    pca_var = [float(v) for v in pca.explained_variance_ratio_]
    
    print("\n--- Feature Separability Comparison (Baseline V5 vs V6 Dual-Stream) ---")
    print(f"Real/Fake Centroid Dist: Baseline = 0.0027 -> V6 = {dist_rf:.4f}")
    print(f"FP/TN Centroid Dist    : Baseline = 0.0044 -> V6 = {dist_fp_tn:.4f}")
    print(f"FN/TP Centroid Dist    : Baseline = 0.0069 -> V6 = {dist_fn_tp:.4f}")
    print(f"Nearest-Centroid Acc   : Baseline = 52.75% -> V6 = {nc_acc*100:.2f}%")
    print(f"k-NN (k=1) Accuracy    : Baseline = 65.12% -> V6 = {knn1_acc*100:.2f}%")
    print(f"k-NN (k=3) Accuracy    : Baseline = 58.50% -> V6 = {knn3_acc*100:.2f}%")
    print(f"k-NN (k=5) Accuracy    : Baseline = 56.38% -> V6 = {knn5_acc*100:.2f}%")
    print(f"Silhouette Score       : Baseline = 0.0005 -> V6 = {sil:.4f}")
    print(f"PCA PC1 Variance       : Baseline = 90.02% -> V6 = {pca_var[0]*100:.2f}%")
    
    # 6. Read-Only Robustness Evaluation on Diagnostic Subset
    print("\n--- Step 6: Read-Only Robustness Evaluation ---")
    # Take 200 balanced images from cohort (100 Real, 100 Fake)
    rob_reals = cohort_df[cohort_df["label"] == 0].head(100)
    rob_fakes = cohort_df[cohort_df["label"] == 1].head(100)
    rob_df = pd.concat([rob_reals, rob_fakes]).reset_index(drop=True)
    rob_paths = [os.path.join(TEST_DIR, os.path.basename(p)) for p in rob_df["face_path"]]
    y_rob = rob_df["label"].values
    
    def evaluate_perturbed_stream(transform_fn, label_str):
        imgs_sp = []
        imgs_fo = []
        for p in rob_paths:
            raw = cv2.imread(p)
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            raw = transform_fn(raw)
            sp = cv2.resize(raw, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            blur = cv2.GaussianBlur(sp, (5, 5), sigmaX=1.0)
            res = sp - blur
            fo = np.clip(res * 4.0, -1.0, 1.0).astype(np.float32)
            imgs_sp.append(sp)
            imgs_fo.append(fo)
        X_sp = np.array(imgs_sp, dtype=np.float32)
        X_fo = np.array(imgs_fo, dtype=np.float32)
        v6_preds = best_model.predict([X_sp, X_fo], batch_size=32, verbose=0).flatten()
        v5_preds = base_model.predict(X_sp, batch_size=32, verbose=0).flatten()
        m_v6 = compute_all_metrics(y_rob, (v6_preds >= th).astype(int), v6_preds)
        m_v5 = compute_all_metrics(y_rob, (v5_preds >= th).astype(int), v5_preds)
        print(f"Robustness [{label_str:<12}]: V5 AUC={m_v5['auc']:.4f}, Acc={m_v5['accuracy']*100:.1f}% | V6 AUC={m_v6['auc']:.4f}, Acc={m_v6['accuracy']*100:.1f}%")
        return {"v5": m_v5, "v6": m_v6}
    
    # 1. Clean
    res_clean = evaluate_perturbed_stream(lambda img: img, "Clean")
    # 2. JPEG compressed (Q=50)
    def apply_jpeg(img):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    res_jpeg = evaluate_perturbed_stream(apply_jpeg, "JPEG_Q50")
    # 3. Resized (downsampled 2x, upsampled)
    def apply_resize(img):
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    res_resize = evaluate_perturbed_stream(apply_resize, "Resized_0.5x")
    # 4. Mild blur (3x3, sigma=0.8)
    def apply_blur(img):
        return cv2.GaussianBlur(img, (3, 3), 0.8)
    res_blur = evaluate_perturbed_stream(apply_blur, "Mild_Blur")
    
    # 7. Save Comparison CSV and Results JSON
    print("\n--- Step 7: Saving Outputs ---")
    comparison_df = pd.DataFrame({
        "filename": baseline_df["filename"],
        "face_path": baseline_df["face_path"],
        "source_dataset": baseline_df["source_dataset"],
        "video_group": baseline_df["video_group"],
        "true_label": y_test,
        "baseline_score": s_baseline,
        "v6_score": s_v6,
        "baseline_pred_021": pred_base_021,
        "v6_pred_021": pred_v6_021,
        "v6_pred_opt": (s_v6 >= best_th).astype(int)
    })
    comparison_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved test comparison to {OUTPUT_CSV_PATH}")
    
    results_json = {
        "training_config": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs_run": len(history.history["loss"]),
            "train_samples": len(X_train_spatial),
            "val_samples": len(X_val_spatial),
            "trainable_parameters": int(trainable_count),
            "non_trainable_parameters": int(non_trainable_count),
            "total_parameters": int(total_count),
            "architecture": "Dual-Stream (Frozen EfficientNetB2 + 3-layer Forensic Residual CNN + Concat + Dense(256))"
        },
        "validation_performance": {
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_val_accuracy": float(history.history["val_accuracy"][-1]),
            "final_val_auc": float(history.history["val_auc"][-1]),
            "best_val_auc": float(max(history.history["val_auc"]))
        },
        "test_performance": {
            "baseline_v5": m_base,
            "v6_dual_stream_021": m_v6_021,
            "v6_dual_stream_optimal": m_v6_opt
        },
        "feature_separability_comparison": {
            "baseline_v5": {
                "real_fake_centroid_dist": 0.0027,
                "nearest_centroid_acc": 0.5275,
                "knn1_acc": 0.6512,
                "silhouette_score": 0.0005,
                "pca_pc1_variance": 0.9002
            },
            "v6_dual_stream": {
                "real_fake_centroid_dist": dist_rf,
                "nearest_centroid_acc": nc_acc,
                "knn1_acc": knn1_acc,
                "knn3_acc": knn3_acc,
                "knn5_acc": knn5_acc,
                "silhouette_score": sil,
                "pca_pc1_variance": pca_var[0]
            }
        },
        "robustness_evaluation": {
            "clean": res_clean,
            "jpeg_q50": res_jpeg,
            "resized_0.5x": res_resize,
            "mild_blur": res_blur
        }
    }
    
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved experiment results to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
