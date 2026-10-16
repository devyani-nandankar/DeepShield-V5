"""
Feature-Separability Diagnostic
DeepShield V5 Image Pipeline
Analyzes 1408-D pooled feature vectors from global_average_pooling2d across 800 test images (200 TN, 200 FP, 200 TP, 200 FN).
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
    silhouette_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy.spatial.distance import cdist, cosine, euclidean
import tensorflow as tf

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

BASE_MODEL_PATH = "models/deepshield_v5_clean_best.keras"
TEST_PRED_BASELINE = "evaluation/direct_fast10_predictions.csv"
TEST_DIR = "evaluation/test_images"

OUTPUT_CSV_PATH = "evaluation/image_feature_separability.csv"
OUTPUT_JSON_PATH = "evaluation/image_feature_separability_summary.json"

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

def extract_features_in_batches(feature_extractor, file_paths, batch_size=64, max_workers=8):
    n_files = len(file_paths)
    all_features = []
    print(f"Extracting features for {n_files} images in batches of {batch_size}...")
    start_t = time.time()
    for i in range(0, n_files, batch_size):
        batch_paths = file_paths[i:i+batch_size]
        batch_imgs = load_images_parallel(batch_paths, max_workers=max_workers)
        feats = feature_extractor.predict(batch_imgs, batch_size=batch_size, verbose=0)
        all_features.append(feats)
        elapsed = time.time() - start_t
        rate = (i + len(batch_paths)) / max(elapsed, 0.001)
        print(f"  Processed {i + len(batch_paths)}/{n_files} ({rate:.1f} imgs/s)")
    return np.vstack(all_features)

def cosine_similarity_vec(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))

def main():
    print("================================================================")
    print("STARTING FEATURE-SEPARABILITY DIAGNOSTIC")
    print("================================================================")
    
    # 1. Load Baseline Predictions and Sample Cohorts
    print("\n--- Step 1: Loading Predictions & Sampling Cohorts ---")
    df = pd.read_csv(TEST_PRED_BASELINE)
    th = 0.21
    
    # Filter cohorts
    tn_df = df[(df["label"] == 0) & (df["raw_score"] < th)].sample(n=200, random_state=42).copy()
    tn_df["cohort"] = "TN_Real"
    
    fp_df = df[(df["label"] == 0) & (df["raw_score"] >= th)].sample(n=200, random_state=42).copy()
    fp_df["cohort"] = "FP_Real"
    
    tp_df = df[(df["label"] == 1) & (df["raw_score"] >= th)].sample(n=200, random_state=42).copy()
    tp_df["cohort"] = "TP_Fake"
    
    fn_df = df[(df["label"] == 1) & (df["raw_score"] < th)].sample(n=200, random_state=42).copy()
    fn_df["cohort"] = "FN_Fake"
    
    sample_df = pd.concat([tn_df, fp_df, tp_df, fn_df]).reset_index(drop=True)
    print(f"Total sampled images: {len(sample_df)}")
    print("Cohort breakdown:")
    print(sample_df["cohort"].value_counts())
    
    # Build local file paths
    sample_df["local_path"] = sample_df["face_path"].apply(lambda p: os.path.join(TEST_DIR, os.path.basename(p)))
    
    # Verify local files exist
    for p in sample_df["local_path"]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing image file: {p}")
    print("All 800 image files verified on disk.")
    
    # 2. Extract 1408-D Feature Vectors
    print("\n--- Step 2: Extracting 1408-D Features from global_average_pooling2d ---")
    base_model = tf.keras.models.load_model(BASE_MODEL_PATH, compile=False)
    gap_layer = base_model.get_layer("global_average_pooling2d")
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=gap_layer.output)
    
    X_features = extract_features_in_batches(feature_extractor, sample_df["local_path"].tolist(), batch_size=64)
    print(f"Extracted feature matrix shape: {X_features.shape}")
    
    y_true = sample_df["label"].values.astype(int)
    cohorts = sample_df["cohort"].values
    
    # 3. Centroids Computation
    print("\n--- Step 3: Centroid Analysis ---")
    # Mask per cohort
    mask_tn = (cohorts == "TN_Real")
    mask_fp = (cohorts == "FP_Real")
    mask_tp = (cohorts == "TP_Fake")
    mask_fn = (cohorts == "FN_Fake")
    mask_real = (y_true == 0)
    mask_fake = (y_true == 1)
    
    c_tn = np.mean(X_features[mask_tn], axis=0)
    c_fp = np.mean(X_features[mask_fp], axis=0)
    c_tp = np.mean(X_features[mask_tp], axis=0)
    c_fn = np.mean(X_features[mask_fn], axis=0)
    c_real = np.mean(X_features[mask_real], axis=0)
    c_fake = np.mean(X_features[mask_fake], axis=0)
    
    centroids = {
        "TN_Real": c_tn,
        "FP_Real": c_fp,
        "TP_Fake": c_tp,
        "FN_Fake": c_fn,
        "All_Real": c_real,
        "All_Fake": c_fake
    }
    
    # Centroid Pairwise Distances (Euclidean & Cosine)
    centroid_names = list(centroids.keys())
    centroid_dist_matrix = {}
    centroid_cos_matrix = {}
    
    print("\nPairwise Centroid Euclidean Distances:")
    print(f"{'':<12} " + " ".join([f"{name:>12}" for name in centroid_names]))
    for name1 in centroid_names:
        centroid_dist_matrix[name1] = {}
        centroid_cos_matrix[name1] = {}
        row_str = f"{name1:<12} "
        for name2 in centroid_names:
            d_euc = float(euclidean(centroids[name1], centroids[name2]))
            d_cos = float(cosine(centroids[name1], centroids[name2]))
            sim_cos = 1.0 - d_cos
            centroid_dist_matrix[name1][name2] = d_euc
            centroid_cos_matrix[name1][name2] = sim_cos
            row_str += f"{d_euc:>12.4f} "
        print(row_str)
        
    print("\nPairwise Centroid Cosine Similarities:")
    print(f"{'':<12} " + " ".join([f"{name:>12}" for name in centroid_names]))
    for name1 in centroid_names:
        row_str = f"{name1:<12} "
        for name2 in centroid_names:
            sim_cos = centroid_cos_matrix[name1][name2]
            row_str += f"{sim_cos:>12.4f} "
        print(row_str)
        
    # Crucial Centroid Proximity Comparisons
    print("\n--- Key Centroid Distances & Proximities ---")
    dist_real_fake = euclidean(c_real, c_fake)
    dist_tn_tp = euclidean(c_tn, c_tp)
    dist_fp_tn = euclidean(c_fp, c_tn)
    dist_fp_tp = euclidean(c_fp, c_tp)
    dist_fn_tp = euclidean(c_fn, c_tp)
    dist_fn_tn = euclidean(c_fn, c_tn)
    
    sim_real_fake = centroid_cos_matrix["All_Real"]["All_Fake"]
    sim_tn_tp = centroid_cos_matrix["TN_Real"]["TP_Fake"]
    sim_fp_tn = centroid_cos_matrix["FP_Real"]["TN_Real"]
    sim_fp_tp = centroid_cos_matrix["FP_Real"]["TP_Fake"]
    sim_fn_tp = centroid_cos_matrix["FN_Fake"]["TP_Fake"]
    sim_fn_tn = centroid_cos_matrix["FN_Fake"]["TN_Real"]
    
    print(f"All_Real vs All_Fake: Euclidean = {dist_real_fake:.4f}, Cosine Sim = {sim_real_fake:.4f}")
    print(f"TN_Real vs TP_Fake (Clean Prototypes): Euclidean = {dist_tn_tp:.4f}, Cosine Sim = {sim_tn_tp:.4f}")
    print(f"FP_Real vs TN_Real: Euclidean = {dist_fp_tn:.4f}, Cosine Sim = {sim_fp_tn:.4f}")
    print(f"FP_Real vs TP_Fake: Euclidean = {dist_fp_tp:.4f}, Cosine Sim = {sim_fp_tp:.4f}")
    print(f"FN_Fake vs TP_Fake: Euclidean = {dist_fn_tp:.4f}, Cosine Sim = {sim_fn_tp:.4f}")
    print(f"FN_Fake vs TN_Real: Euclidean = {dist_fn_tn:.4f}, Cosine Sim = {sim_fn_tn:.4f}")
    
    # 4. Individual Sample Proximities to Canonical Prototypes (c_tn vs c_tp)
    print("\n--- Step 4: Individual Sample Distances to Canonical Prototypes (TN vs TP) ---")
    d_to_c_tn = np.linalg.norm(X_features - c_tn, axis=1)
    d_to_c_tp = np.linalg.norm(X_features - c_tp, axis=1)
    
    # Cosine similarities
    norm_X = np.linalg.norm(X_features, axis=1, keepdims=True)
    norm_c_tn = np.linalg.norm(c_tn)
    norm_c_tp = np.linalg.norm(c_tp)
    
    cos_sim_tn = np.dot(X_features, c_tn) / (norm_X.flatten() * norm_c_tn + 1e-12)
    cos_sim_tp = np.dot(X_features, c_tp) / (norm_X.flatten() * norm_c_tp + 1e-12)
    
    # Nearest centroid prediction: 0 if closer to c_tn, 1 if closer to c_tp
    pred_nearest_centroid = (d_to_c_tp < d_to_c_tn).astype(int)
    
    sample_df["dist_to_c_tn"] = d_to_c_tn
    sample_df["dist_to_c_tp"] = d_to_c_tp
    sample_df["cos_sim_c_tn"] = cos_sim_tn
    sample_df["cos_sim_c_tp"] = cos_sim_tp
    sample_df["nearest_centroid_pred"] = pred_nearest_centroid
    
    # Per-cohort breakdown of nearest centroid predictions
    for c_name in ["TN_Real", "FP_Real", "TP_Fake", "FN_Fake"]:
        c_sub = sample_df[sample_df["cohort"] == c_name]
        pct_closer_tp = (c_sub["nearest_centroid_pred"] == 1).mean() * 100
        mean_d_tn = c_sub["dist_to_c_tn"].mean()
        mean_d_tp = c_sub["dist_to_c_tp"].mean()
        mean_cos_tn = c_sub["cos_sim_c_tn"].mean()
        mean_cos_tp = c_sub["cos_sim_c_tp"].mean()
        print(f"Cohort {c_name:<8}: Closer to TP={pct_closer_tp:5.1f}%, Closer to TN={100-pct_closer_tp:5.1f}% | Dist(TN)={mean_d_tn:.4f}, Dist(TP)={mean_d_tp:.4f} | CosSim(TN)={mean_cos_tn:.4f}, CosSim(TP)={mean_cos_tp:.4f}")
        
    # Nearest Centroid Accuracy on True Labels
    nc_acc_true = accuracy_score(y_true, pred_nearest_centroid)
    print(f"\nNearest-Centroid Overall Accuracy vs True Labels: {nc_acc_true*100:.2f}%")
    
    # 5. k-NN Feature-Space Classification
    print("\n--- Step 5: k-NN Feature-Space Classification ---")
    knn_results = {}
    for k in [1, 3, 5, 7, 15]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        # Leave-one-out cross validation
        from sklearn.model_selection import LeaveOneOut, cross_val_score
        scores = cross_val_score(knn, X_features, y_true, cv=5)
        knn_results[f"k={k}"] = float(np.mean(scores))
        print(f"  k-NN (k={k:2d}) 5-Fold CV Accuracy: {np.mean(scores)*100:.2f}%")
        
    # Nearest-neighbor class agreement (k=1)
    nn_finder = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn_finder.fit(X_features)
    distances, indices = nn_finder.kneighbors(X_features)
    
    # Index of closest distinct neighbor (indices[:, 1])
    closest_neighbor_idx = indices[:, 1]
    neighbor_true_label = y_true[closest_neighbor_idx]
    neighbor_cohort = cohorts[closest_neighbor_idx]
    
    sample_df["1nn_neighbor_label"] = neighbor_true_label
    sample_df["1nn_neighbor_cohort"] = neighbor_cohort
    sample_df["1nn_neighbor_dist"] = distances[:, 1]
    
    nn_agreement_true = (neighbor_true_label == y_true).mean()
    print(f"\n1-NN Nearest-Neighbor Class Agreement (True Label): {nn_agreement_true*100:.2f}%")
    
    for c_name in ["TN_Real", "FP_Real", "TP_Fake", "FN_Fake"]:
        c_sub = sample_df[sample_df["cohort"] == c_name]
        pct_neighbor_fake = (c_sub["1nn_neighbor_label"] == 1).mean() * 100
        pct_neighbor_real = 100.0 - pct_neighbor_fake
        print(f"Cohort {c_name:<8}: 1-NN is Real={pct_neighbor_real:5.1f}%, 1-NN is Fake={pct_neighbor_fake:5.1f}%")
        
    # 6. PCA (2 Dimensions & 10 Dimensions)
    print("\n--- Step 6: PCA Analysis (2-D and 10-D) ---")
    pca_10 = PCA(n_components=10, random_state=42)
    X_pca_10 = pca_10.fit_transform(X_features)
    exp_var_10 = pca_10.explained_variance_ratio_
    cum_var_10 = np.cumsum(exp_var_10)
    
    print("PCA Top 10 Components Explained Variance:")
    for idx, (ev, cv) in enumerate(zip(exp_var_10, cum_var_10), start=1):
        print(f"  PC{idx:2d}: {ev*100:5.2f}% (Cumulative: {cv*100:5.2f}%)")
        
    sample_df["PC1"] = X_pca_10[:, 0]
    sample_df["PC2"] = X_pca_10[:, 1]
    for i in range(1, 11):
        sample_df[f"PC{i}"] = X_pca_10[:, i-1]
        
    # 7. Cluster Separation & Silhouette Scores
    print("\n--- Step 7: Clustering Quality Metrics ---")
    sil_true = float(silhouette_score(X_features, y_true, metric="euclidean"))
    sil_cohort = float(silhouette_score(X_features, [0 if 'Real' in c else 1 for c in cohorts], metric="euclidean"))
    ch_score = float(calinski_harabasz_score(X_features, y_true))
    db_score = float(davies_bouldin_score(X_features, y_true))
    
    print(f"Silhouette Score (True Real vs Fake): {sil_true:.4f}")
    print(f"Calinski-Harabasz Index: {ch_score:.2f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    
    # 8. Save CSV and JSON
    print(f"\n--- Step 8: Saving Deliverables ---")
    # Drop local_path before saving
    export_df = sample_df.drop(columns=["local_path"])
    export_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved {OUTPUT_CSV_PATH} ({len(export_df)} rows)")
    
    summary_data = {
        "dataset_size": len(sample_df),
        "cohorts": {
            "TN_Real": 200,
            "FP_Real": 200,
            "TP_Fake": 200,
            "FN_Fake": 200
        },
        "centroid_distances_euclidean": centroid_dist_matrix,
        "centroid_cosine_similarities": centroid_cos_matrix,
        "key_distances": {
            "dist_real_fake": float(dist_real_fake),
            "sim_real_fake": float(sim_real_fake),
            "dist_tn_tp": float(dist_tn_tp),
            "sim_tn_tp": float(sim_tn_tp),
            "dist_fp_tn": float(dist_fp_tn),
            "sim_fp_tn": float(sim_fp_tn),
            "dist_fp_tp": float(dist_fp_tp),
            "sim_fp_tp": float(sim_fp_tp),
            "dist_fn_tp": float(dist_fn_tp),
            "sim_fn_tp": float(sim_fn_tp),
            "dist_fn_tn": float(dist_fn_tn),
            "sim_fn_tn": float(sim_fn_tn)
        },
        "cohort_proximities": {
            c_name: {
                "pct_closer_to_tp": float((sample_df[sample_df["cohort"] == c_name]["nearest_centroid_pred"] == 1).mean() * 100),
                "pct_closer_to_tn": float((sample_df[sample_df["cohort"] == c_name]["nearest_centroid_pred"] == 0).mean() * 100),
                "mean_dist_to_c_tn": float(sample_df[sample_df["cohort"] == c_name]["dist_to_c_tn"].mean()),
                "mean_dist_to_c_tp": float(sample_df[sample_df["cohort"] == c_name]["dist_to_c_tp"].mean()),
                "mean_cos_sim_c_tn": float(sample_df[sample_df["cohort"] == c_name]["cos_sim_c_tn"].mean()),
                "mean_cos_sim_c_tp": float(sample_df[sample_df["cohort"] == c_name]["cos_sim_c_tp"].mean()),
                "1nn_pct_neighbor_fake": float((sample_df[sample_df["cohort"] == c_name]["1nn_neighbor_label"] == 1).mean() * 100),
                "1nn_pct_neighbor_real": float((sample_df[sample_df["cohort"] == c_name]["1nn_neighbor_label"] == 0).mean() * 100)
            }
            for c_name in ["TN_Real", "FP_Real", "TP_Fake", "FN_Fake"]
        },
        "nearest_centroid_accuracy": float(nc_acc_true),
        "knn_5fold_cv_accuracy": knn_results,
        "1nn_class_agreement": float(nn_agreement_true),
        "pca_explained_variance_10": [float(x) for x in exp_var_10],
        "pca_cumulative_variance_10": [float(x) for x in cum_var_10],
        "clustering_metrics": {
            "silhouette_score_true": sil_true,
            "calinski_harabasz": ch_score,
            "davies_bouldin": db_score
        }
    }
    
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
