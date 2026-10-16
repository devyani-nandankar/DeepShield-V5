"""
Evaluation and Feature-Separability Diagnostic for Fine-Tuned V5 Image Model
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

MODEL_PATH = "models/v5_image_finetuned/deepshield_v5_image_finetuned.keras"
TEST_PRED_BASELINE = "evaluation/direct_fast10_predictions.csv"
TEST_DIR = "evaluation/test_images"
COHORT_800_CSV = "evaluation/image_feature_separability.csv"

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
    print("EVALUATING FINE-TUNED MODEL & FEATURE SEPARABILITY")
    print("================================================================")
    
    print(f"Loading fine-tuned model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # 1. Full FAST10 Test Set Evaluation (N = 3,693)
    print("\n--- Step 1: Evaluating on Official Untouched FAST10 Test Set (N = 3,693) ---")
    baseline_df = pd.read_csv(TEST_PRED_BASELINE)
    test_paths = [os.path.join(TEST_DIR, os.path.basename(p)) for p in baseline_df["face_path"]]
    
    all_scores = []
    batch_sz = 64
    t0 = time.time()
    for i in range(0, len(test_paths), batch_sz):
        batch_p = test_paths[i:i+batch_sz]
        batch_imgs = load_images_parallel(batch_p, max_workers=8)
        preds = model.predict(batch_imgs, batch_size=batch_sz, verbose=0).flatten()
        all_scores.extend(preds)
        if (i // batch_sz) % 10 == 0 or i + batch_sz >= len(test_paths):
            print(f"  Inference: {min(i+batch_sz, len(test_paths))}/{len(test_paths)} images")
            
    s_finetuned = np.array(all_scores, dtype=np.float32)
    s_baseline = baseline_df["raw_score"].values.astype(np.float32)
    y_test = baseline_df["label"].values.astype(int)
    
    th = 0.21
    pred_base_021 = (s_baseline >= th).astype(int)
    pred_fine_021 = (s_finetuned >= th).astype(int)
    
    m_base = compute_all_metrics(y_test, pred_base_021, s_baseline)
    m_fine_021 = compute_all_metrics(y_test, pred_fine_021, s_finetuned)
    
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
    
    # 2. Feature Separability Diagnostic on the NEW Model (Same 800 Cohort)
    print("\n--- Step 2: Feature-Separability Diagnostic on Fine-Tuned Model (N = 800) ---")
    cohort_df = pd.read_csv(COHORT_800_CSV)
    cohort_paths = [os.path.join(TEST_DIR, os.path.basename(p)) for p in cohort_df["face_path"]]
    
    gap_layer_fine = model.get_layer("global_average_pooling2d")
    feat_extractor_fine = tf.keras.Model(inputs=model.input, outputs=gap_layer_fine.output)
    
    all_fine_feats = []
    for i in range(0, len(cohort_paths), batch_sz):
        batch_p = cohort_paths[i:i+batch_sz]
        batch_imgs = load_images_parallel(batch_p, max_workers=8)
        feats = feat_extractor_fine.predict(batch_imgs, batch_size=batch_sz, verbose=0)
        all_fine_feats.append(feats)
    X_fine_feats = np.vstack(all_fine_feats)
    
    cohort_labels = cohort_df["label"].values
    cohort_names = cohort_df["cohort"].values
    
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
    
    dist_rf_new = float(euclidean(c_real_new, c_fake_new))
    dist_fp_tn_new = float(euclidean(c_fp_new, c_tn_new))
    dist_fp_tp_new = float(euclidean(c_fp_new, c_tp_new))
    dist_fn_tp_new = float(euclidean(c_fn_new, c_tp_new))
    dist_fn_tn_new = float(euclidean(c_fn_new, c_tn_new))
    
    d_to_tn_new = np.linalg.norm(X_fine_feats - c_tn_new, axis=1)
    d_to_tp_new = np.linalg.norm(X_fine_feats - c_tp_new, axis=1)
    nc_preds_new = (d_to_tp_new < d_to_tn_new).astype(int)
    nc_acc_new = float(accuracy_score(cohort_labels, nc_preds_new))
    
    from sklearn.model_selection import cross_val_score
    knn1 = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn1_acc_new = float(np.mean(cross_val_score(knn1, X_fine_feats, cohort_labels, cv=5)))
    knn3 = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn3_acc_new = float(np.mean(cross_val_score(knn3, X_fine_feats, cohort_labels, cv=5)))
    knn5 = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn5_acc_new = float(np.mean(cross_val_score(knn5, X_fine_feats, cohort_labels, cv=5)))
    
    sil_new = float(silhouette_score(X_fine_feats, cohort_labels, metric="euclidean"))
    
    pca = PCA(n_components=10, random_state=42)
    pca.fit(X_fine_feats)
    pca_var_new = [float(v) for v in pca.explained_variance_ratio_]
    
    print("\n--- Feature Separability Comparison (Baseline vs Fine-Tuned) ---")
    print(f"Real/Fake Centroid Dist: Baseline = 0.0027 -> Fine-Tuned = {dist_rf_new:.4f}")
    print(f"FP/TN Centroid Dist    : Baseline = 0.0044 -> Fine-Tuned = {dist_fp_tn_new:.4f}")
    print(f"FP/TP Centroid Dist    : Baseline = 0.0042 -> Fine-Tuned = {dist_fp_tp_new:.4f}")
    print(f"FN/TP Centroid Dist    : Baseline = 0.0069 -> Fine-Tuned = {dist_fn_tp_new:.4f}")
    print(f"FN/TN Centroid Dist    : Baseline = 0.0025 -> Fine-Tuned = {dist_fn_tn_new:.4f}")
    print(f"Nearest-Centroid Acc   : Baseline = 52.75% -> Fine-Tuned = {nc_acc_new*100:.2f}%")
    print(f"k-NN (k=1) Accuracy    : Baseline = 65.12% -> Fine-Tuned = {knn1_acc_new*100:.2f}%")
    print(f"k-NN (k=3) Accuracy    : Baseline = 58.50% -> Fine-Tuned = {knn3_acc_new*100:.2f}%")
    print(f"k-NN (k=5) Accuracy    : Baseline = 56.38% -> Fine-Tuned = {knn5_acc_new*100:.2f}%")
    print(f"Silhouette Score       : Baseline = 0.0005 -> Fine-Tuned = {sil_new:.4f}")
    print(f"PCA PC1 Variance       : Baseline = 90.02% -> Fine-Tuned = {pca_var_new[0]*100:.2f}%")
    
    # 3. Save Comparison CSV and Results JSON
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
            "learning_rate": 1e-5,
            "batch_size": 32,
            "epochs_run": 4,
            "train_samples": 2000,
            "val_samples": 400,
            "trainable_parameters": 6601499,
            "non_trainable_parameters": 1528031,
            "total_parameters": 8129530,
            "unfrozen_blocks": ["block6", "block7", "top_conv", "dense", "dense_1"],
            "frozen_blocks": ["blocks 1-5", "all BatchNormalization"]
        },
        "validation_performance": {
            "best_epoch": 1,
            "val_auc_epoch1": 0.5000,
            "val_loss_epoch1": 0.6932,
            "val_accuracy_epoch1": 0.5000
        },
        "test_performance": {
            "baseline_v5": m_base,
            "finetuned_v5_021": m_fine_021,
            "finetuned_v5_optimal": m_fine_opt
        },
        "feature_separability_comparison": {
            "baseline": {
                "real_fake_centroid_dist": 0.0027,
                "fp_tn_centroid_dist": 0.0044,
                "fp_tp_centroid_dist": 0.0042,
                "fn_tp_centroid_dist": 0.0069,
                "fn_tn_centroid_dist": 0.0025,
                "nearest_centroid_acc": 0.5275,
                "knn1_acc": 0.6512,
                "knn3_acc": 0.5850,
                "knn5_acc": 0.5638,
                "silhouette_score": 0.0005,
                "pca_pc1_variance": 0.9002
            },
            "finetuned": {
                "real_fake_centroid_dist": dist_rf_new,
                "fp_tn_centroid_dist": dist_fp_tn_new,
                "fp_tp_centroid_dist": dist_fp_tp_new,
                "fn_tp_centroid_dist": dist_fn_tp_new,
                "fn_tn_centroid_dist": dist_fn_tn_new,
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
