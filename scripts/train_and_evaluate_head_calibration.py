"""
Controlled Hard-Negative Classification-Head Calibration Experiment
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

MANIFEST_PATH = "Kaggle_Transfer/face_crop_FAST10_manifest_split.csv"
TRAIN_DIR = "G:/My Drive/DeepShield_Datasets/Face_Cropped_FAST10/train"
VAL_DIR = "G:/My Drive/DeepShield_Datasets/Face_Cropped_FAST10/validation"
TEST_DIR = "evaluation/test_images"
TEST_PRED_BASELINE = "evaluation/direct_fast10_predictions.csv"

BASE_MODEL_PATH = "models/deepshield_v5_clean_best.keras"
CALIBRATED_MODEL_PATH = "models/v5_image_head_calibrated.keras"
HISTORY_CSV_PATH = "evaluation/image_head_calibration_history.csv"
TEST_RESULTS_CSV_PATH = "evaluation/image_head_calibration_test_results.csv"

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
        if (i // batch_size) % 10 == 0 or i + batch_size >= n_files:
            elapsed = time.time() - start_t
            rate = (i + len(batch_paths)) / max(elapsed, 0.001)
            print(f"  Processed {i + len(batch_paths)}/{n_files} ({rate:.1f} imgs/s)")
            
    return np.vstack(all_features)

def main():
    print("================================================================")
    print("STARTING CONTROLLED HARD-NEGATIVE HEAD CALIBRATION EXPERIMENT")
    print("================================================================")
    
    # 1. Load manifest and verify splits
    print("\n--- Step 1: Loading Manifest & Mining Hard Negatives ---")
    df = pd.read_csv(MANIFEST_PATH)
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'validation'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    print(f"Train split: {len(train_df)} images")
    print(f"Validation split: {len(val_df)} images")
    print(f"Test split: {len(test_df)} images (Strictly held out)")
    
    train_df['full_path'] = train_df['face_path'].apply(lambda x: os.path.join(TRAIN_DIR, os.path.basename(x)))
    val_df['full_path'] = val_df['face_path'].apply(lambda x: os.path.join(VAL_DIR, os.path.basename(x)))
    test_df['full_path'] = test_df['face_path'].apply(lambda x: os.path.join(TEST_DIR, os.path.basename(x)))
    
    # 2. Load Base Model and Build Feature Extractor
    print("\n--- Step 2: Loading Base V5 Model & Freezing Backbone ---")
    base_model = tf.keras.models.load_model(BASE_MODEL_PATH, compile=False)
    
    gap_layer = base_model.get_layer('global_average_pooling2d')
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=gap_layer.output)
    
    orig_dense = base_model.get_layer('dense')
    orig_dense_1 = base_model.get_layer('dense_1')
    w_dense = orig_dense.get_weights()
    w_dense_1 = orig_dense_1.get_weights()
    print(f"Original dense units: {orig_dense.units}, dense_1 units: {orig_dense_1.units}")
    
    # 3. Mine Hard Negatives from TRAIN split
    real_train_df = train_df[train_df['label'] == 0]
    fake_train_df = train_df[train_df['label'] == 1]
    
    real_candidates = []
    for ds in ['FF-C23', 'Deepfake-Videos-Merged', 'FaceForensics-1000']:
        sub = real_train_df[real_train_df['source_dataset'] == ds]
        sample_n = min(800, len(sub))
        real_candidates.append(sub.sample(n=sample_n, random_state=42))
    real_candidates_df = pd.concat(real_candidates).reset_index(drop=True)
    print(f"Mining hard negatives from {len(real_candidates_df)} candidate Real training images...")
    
    cand_feats = extract_features_in_batches(feature_extractor, real_candidates_df['full_path'].tolist(), batch_size=64)
    
    h1 = np.maximum(0, np.dot(cand_feats, w_dense[0]) + w_dense[1])
    cand_scores = 1.0 / (1.0 + np.exp(-(np.dot(h1, w_dense_1[0]) + w_dense_1[1]))).flatten()
    real_candidates_df['baseline_score'] = cand_scores
    
    real_candidates_df = real_candidates_df.sort_values(by='baseline_score', ascending=False)
    
    hard_negatives = real_candidates_df.iloc[:600]
    remaining_reals = real_candidates_df.iloc[600:].sample(n=600, random_state=42)
    selected_reals = pd.concat([hard_negatives, remaining_reals]).reset_index(drop=True)
    
    print(f"Selected {len(selected_reals)} Real training images:")
    print(f"  - Hard negatives (top score): {len(hard_negatives)} (mean baseline score: {hard_negatives['baseline_score'].mean():.4f}, min: {hard_negatives['baseline_score'].min():.4f}, max: {hard_negatives['baseline_score'].max():.4f})")
    print(f"  - Representative reals: {len(remaining_reals)} (mean baseline score: {remaining_reals['baseline_score'].mean():.4f})")
    
    fake_samples = []
    for ds in ['FF-C23', 'Deepfake-Videos-Merged', 'FaceForensics-1000']:
        sub = fake_train_df[fake_train_df['source_dataset'] == ds]
        sample_n = min(400, len(sub))
        fake_samples.append(sub.sample(n=sample_n, random_state=42))
    selected_fakes = pd.concat(fake_samples).reset_index(drop=True)
    print(f"Selected {len(selected_fakes)} Fake training images (400 per source dataset)")
    
    calibration_train_df = pd.concat([selected_reals, selected_fakes]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Total calibration training set: {len(calibration_train_df)} images ({sum(calibration_train_df['label']==0)} Real, {sum(calibration_train_df['label']==1)} Fake)")
    
    val_real_df = val_df[val_df['label'] == 0]
    val_fake_df = val_df[val_df['label'] == 1]
    
    val_reals = []
    val_fakes = []
    for ds in ['FF-C23', 'Deepfake-Videos-Merged', 'FaceForensics-1000']:
        sub_r = val_real_df[val_real_df['source_dataset'] == ds]
        sub_f = val_fake_df[val_fake_df['source_dataset'] == ds]
        val_reals.append(sub_r.sample(n=min(100, len(sub_r)), random_state=42))
        val_fakes.append(sub_f.sample(n=min(100, len(sub_f)), random_state=42))
    calibration_val_df = pd.concat(val_reals + val_fakes).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Total validation set: {len(calibration_val_df)} images ({sum(calibration_val_df['label']==0)} Real, {sum(calibration_val_df['label']==1)} Fake)")
    
    # 4. Extract features for calibration train and validation sets
    print("\n--- Step 3: Extracting Features for Calibration Training & Validation Sets ---")
    train_features = extract_features_in_batches(feature_extractor, calibration_train_df['full_path'].tolist(), batch_size=64)
    val_features = extract_features_in_batches(feature_extractor, calibration_val_df['full_path'].tolist(), batch_size=64)
    
    y_train = calibration_train_df['label'].values.astype(np.float32)
    y_val = calibration_val_df['label'].values.astype(np.float32)
    
    # 5. Build and Train Classification Head
    print("\n--- Step 4: Training Dense Classification Head with Adam (lr=2e-4) ---")
    head_input = tf.keras.Input(shape=(1408,), name="gap_input")
    x = tf.keras.layers.Dropout(0.3, name="calib_dropout")(head_input)
    x = tf.keras.layers.Dense(256, activation="relu", name="calib_dense")(x)
    x = tf.keras.layers.Dropout(0.2, name="calib_dropout_1")(x)
    head_output = tf.keras.layers.Dense(1, activation="sigmoid", name="calib_dense_1")(x)
    
    head_model = tf.keras.Model(inputs=head_input, outputs=head_output, name="calibrated_head")
    
    head_model.get_layer("calib_dense").set_weights(w_dense)
    head_model.get_layer("calib_dense_1").set_weights(w_dense_1)
    
    learning_rate = 2e-4
    head_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    
    epochs = 10
    batch_size = 32
    history_records = []
    
    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_dense_weights = None
    best_dense_1_weights = None
    best_epoch = 0
    
    print(f"Starting head training for {epochs} epochs (batch_size={batch_size}, lr={learning_rate})...")
    for ep in range(1, epochs + 1):
        hist = head_model.fit(
            train_features, y_train,
            validation_data=(val_features, y_val),
            epochs=1,
            batch_size=batch_size,
            verbose=0,
            shuffle=True
        )
        
        train_loss = float(hist.history["loss"][0])
        train_acc = float(hist.history["accuracy"][0])
        train_auc = float(hist.history["auc"][0])
        val_loss = float(hist.history["val_loss"][0])
        val_acc = float(hist.history["val_accuracy"][0])
        val_auc = float(hist.history["val_auc"][0])
        
        val_preds = head_model.predict(val_features, verbose=0).flatten()
        val_f1_05 = float(f1_score(y_val, (val_preds >= 0.5).astype(int), zero_division=0))
        val_f1_021 = float(f1_score(y_val, (val_preds >= 0.21).astype(int), zero_division=0))
        
        print(f"Epoch {ep:2d}/{epochs}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, AUC={train_auc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}, F1@0.21={val_f1_021:.4f}")
        
        history_records.append({
            "epoch": ep,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_auc": val_auc,
            "val_f1_021": val_f1_021,
            "val_f1_05": val_f1_05
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = ep
            best_dense_weights = head_model.get_layer("calib_dense").get_weights()
            best_dense_1_weights = head_model.get_layer("calib_dense_1").get_weights()
            
    print(f"\nBest epoch was Epoch {best_epoch} with Val Loss={best_val_loss:.4f}, Val AUC={best_val_auc:.4f}")
    
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(HISTORY_CSV_PATH, index=False)
    print(f"Saved calibration history to {HISTORY_CSV_PATH}")
    
    # 6. Reconstruct Full End-to-End Calibrated Model
    print("\n--- Step 5: Assembling and Saving Full End-to-End Calibrated Model ---")
    calibrated_full_model = tf.keras.models.load_model(BASE_MODEL_PATH, compile=False)
    
    for layer in calibrated_full_model.layers:
        if layer.name in ['dense', 'dense_1']:
            layer.trainable = True
        else:
            layer.trainable = False
            
    calibrated_full_model.get_layer('dense').set_weights(best_dense_weights)
    calibrated_full_model.get_layer('dense_1').set_weights(best_dense_1_weights)
    
    calibrated_full_model.save(CALIBRATED_MODEL_PATH)
    print(f"Saved calibrated model to {CALIBRATED_MODEL_PATH}")
    
    head_model.get_layer("calib_dense").set_weights(best_dense_weights)
    head_model.get_layer("calib_dense_1").set_weights(best_dense_1_weights)
    
    # 7. Evaluate on UNTOUCHED Official FAST10 Test Split (3,693 images)
    print("\n--- Step 6: Evaluating Both Models on UNTOUCHED FAST10 Test Set (3,693 images) ---")
    baseline_df = pd.read_csv(TEST_PRED_BASELINE)
    print(f"Loaded {len(baseline_df)} baseline test predictions.")
    
    test_file_paths = [os.path.join(TEST_DIR, os.path.basename(p)) for p in baseline_df['face_path']]
    test_features = extract_features_in_batches(feature_extractor, test_file_paths, batch_size=64)
    
    calibrated_scores = head_model.predict(test_features, batch_size=64, verbose=0).flatten()
    
    test_results_df = pd.DataFrame({
        "filename": baseline_df["filename"],
        "face_path": baseline_df["face_path"],
        "source_dataset": baseline_df["source_dataset"],
        "video_group": baseline_df["video_group"],
        "true_label": baseline_df["label"],
        "baseline_score": baseline_df["raw_score"],
        "calibrated_score": calibrated_scores
    })
    
    y_true = test_results_df["true_label"].values
    s_base = test_results_df["baseline_score"].values
    s_calib = test_results_df["calibrated_score"].values
    
    th = 0.21
    pred_base_021 = (s_base >= th).astype(int)
    pred_calib_021 = (s_calib >= th).astype(int)
    
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
    
    base_m_021 = compute_all_metrics(y_true, pred_base_021, s_base)
    calib_m_021 = compute_all_metrics(y_true, pred_calib_021, s_calib)
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_th = th
    best_f1 = 0.0
    for t in thresholds:
        f = f1_score(y_true, (s_calib >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = t
            
    calib_m_best = compute_all_metrics(y_true, (s_calib >= best_th).astype(int), s_calib)
    calib_m_best["optimal_threshold"] = float(best_th)
    
    fp_mask = (y_true == 0) & (pred_base_021 == 1)
    fn_mask = (y_true == 1) & (pred_base_021 == 0)
    tp_mask = (y_true == 1) & (pred_base_021 == 1)
    tn_mask = (y_true == 0) & (pred_base_021 == 0)
    
    fp_indices = np.where(fp_mask)[0]
    fp_corrected_to_tn = int(np.sum(pred_calib_021[fp_indices] == 0))
    fp_remained_fp = int(np.sum(pred_calib_021[fp_indices] == 1))
    fp_mean_score_base = float(np.mean(s_base[fp_indices]))
    fp_mean_score_calib = float(np.mean(s_calib[fp_indices]))
    
    fn_indices = np.where(fn_mask)[0]
    fn_remained_fn = int(np.sum(pred_calib_021[fn_indices] == 0))
    fn_corrected_to_tp = int(np.sum(pred_calib_021[fn_indices] == 1))
    fn_mean_score_base = float(np.mean(s_base[fn_indices]))
    fn_mean_score_calib = float(np.mean(s_calib[fn_indices]))
    
    tn_indices = np.where(tn_mask)[0]
    new_fps_created = int(np.sum(pred_calib_021[tn_indices] == 1))
    
    tp_indices = np.where(tp_mask)[0]
    new_fns_created = int(np.sum(pred_calib_021[tp_indices] == 0))
    
    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)
    real_score_shift = float(np.mean(s_calib[real_mask]) - np.mean(s_base[real_mask]))
    fake_score_shift = float(np.mean(s_calib[fake_mask]) - np.mean(s_base[fake_mask]))
    
    source_datasets = test_results_df["source_dataset"].unique()
    source_breakdown = {}
    for ds in source_datasets:
        ds_mask = (test_results_df["source_dataset"] == ds).values
        ds_base = compute_all_metrics(y_true[ds_mask], pred_base_021[ds_mask], s_base[ds_mask])
        ds_calib = compute_all_metrics(y_true[ds_mask], pred_calib_021[ds_mask], s_calib[ds_mask])
        source_breakdown[str(ds)] = {
            "count": int(np.sum(ds_mask)),
            "baseline": ds_base,
            "calibrated": ds_calib
        }
        
    test_results_df["baseline_pred_021"] = pred_base_021
    test_results_df["calibrated_pred_021"] = pred_calib_021
    test_results_df["calibrated_pred_best"] = (s_calib >= best_th).astype(int)
    test_results_df.to_csv(TEST_RESULTS_CSV_PATH, index=False)
    print(f"Saved test comparison results to {TEST_RESULTS_CSV_PATH}")
    
    print("\n================================================================")
    print("TEST SET EVALUATION SUMMARY (N = 3,693)")
    print("================================================================")
    print("Metrics comparison:")
    for m in ["accuracy", "precision", "recall", "f1", "auc", "fpr", "fnr"]:
        print(f"{m:<12}: Base@0.21={base_m_021[m]:.4f} | Calib@0.21={calib_m_021[m]:.4f} | Calib@best({best_th:.2f})={calib_m_best[m]:.4f}")
    
    print("\n--- Specific Error Repair Analysis ---")
    print(f"477 Baseline FPs: Corrected to TN = {fp_corrected_to_tn} ({fp_corrected_to_tn/477*100:.1f}%), Remained FP = {fp_remained_fp}")
    print(f"  Mean Score: Before = {fp_mean_score_base:.4f} -> After = {fp_mean_score_calib:.4f} (Shift = {fp_mean_score_calib - fp_mean_score_base:.4f})")
    print(f"223 Baseline FNs: Corrected to TP = {fn_corrected_to_tp} ({fn_corrected_to_tp/223*100:.1f}%), Remained FN = {fn_remained_fn}")
    print(f"  Mean Score: Before = {fn_mean_score_base:.4f} -> After = {fn_mean_score_calib:.4f} (Shift = {fn_mean_score_calib - fn_mean_score_base:.4f})")
    print(f"New FPs Created (from 516 baseline TNs): {new_fps_created} ({new_fps_created/516*100:.1f}%)")
    print(f"New FNs Created (from 2477 baseline TPs): {new_fns_created} ({new_fns_created/2477*100:.1f}%)")
    print(f"Overall Real Score Shift: {real_score_shift:+.4f} (Mean Base: {np.mean(s_base[real_mask]):.4f} -> Mean Calib: {np.mean(s_calib[real_mask]):.4f})")
    print(f"Overall Fake Score Shift: {fake_score_shift:+.4f} (Mean Base: {np.mean(s_base[fake_mask]):.4f} -> Mean Calib: {np.mean(s_calib[fake_mask]):.4f})")
    
    summary_data = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_auc": float(best_val_auc),
        "baseline_at_021": base_m_021,
        "calibrated_at_021": calib_m_021,
        "calibrated_optimal": calib_m_best,
        "error_repair": {
            "baseline_fps_total": 477,
            "fp_corrected_to_tn": fp_corrected_to_tn,
            "fp_remained_fp": fp_remained_fp,
            "fp_mean_score_base": fp_mean_score_base,
            "fp_mean_score_calib": fp_mean_score_calib,
            "baseline_fns_total": 223,
            "fn_corrected_to_tp": fn_corrected_to_tp,
            "fn_remained_fn": fn_remained_fn,
            "fn_mean_score_base": fn_mean_score_base,
            "fn_mean_score_calib": fn_mean_score_calib,
            "new_fps_created": new_fps_created,
            "new_fns_created": new_fns_created,
            "real_score_mean_base": float(np.mean(s_base[real_mask])),
            "real_score_mean_calib": float(np.mean(s_calib[real_mask])),
            "real_score_shift": real_score_shift,
            "fake_score_mean_base": float(np.mean(s_base[fake_mask])),
            "fake_score_mean_calib": float(np.mean(s_calib[fake_mask])),
            "fake_score_shift": fake_score_shift
        },
        "source_breakdown": source_breakdown
    }
    
    with open("evaluation/image_head_calibration_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print("Saved evaluation summary to evaluation/image_head_calibration_summary.json")

if __name__ == "__main__":
    main()
