"""
DeepShield V5 - Two-Stage Training Pipeline
===========================================

Project Purpose:
Explainable multimodal deepfake detection and digital media authenticity verification.

Model Architecture:
- Backbone: EfficientNetB2 (ImageNet pretrained)
- Input: 224 x 224 x 3
- Head: GlobalAveragePooling2D -> Dropout -> Dense -> Dropout -> Dense(1, activation="sigmoid")
- Class Mapping: 0 = Real, 1 = Fake (higher score = Fake, lower score = Real)

Training Strategy:
- Stage 1: Freeze EfficientNetB2 backbone, train classification head only.
- Stage 2: Load best Stage 1 checkpoint, unfreeze top ~30% layers (BatchNorm layers kept frozen),
           fine-tune at reduced learning rate.
- Metric to monitor: Validation AUC (val_auc)

Environment: Suitable for Google Colab, Kaggle, or local GPU execution.
"""

import os
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, callbacks

# ==============================================================================
# SCRIPT VERSION & INTEGRITY IDENTIFIER
# ==============================================================================
SCRIPT_VERSION = "5.0.3-face-path-fix"

# ==============================================================================
# INITIAL CONFIGURATION (Adjustable hyperparameters)
# ==============================================================================
# NOTE: 90%+ accuracy is a project benchmark TARGET, not an automatic guarantee.
# All hyperparameters are baseline starting values and can be tuned as needed.
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 25
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
SEED = 42
FINE_TUNE_PERCENTAGE = 30

# ==============================================================================
# ENVIRONMENT & PATH CONFIGURATION (Colab vs Kaggle)
# ==============================================================================
# Auto-detect ENVIRONMENT or allow override via environment variable
if os.environ.get("ENVIRONMENT"):
    ENVIRONMENT = os.environ.get("ENVIRONMENT").lower()
elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
    ENVIRONMENT = "kaggle"
elif Path("/content").exists():
    ENVIRONMENT = "colab"
else:
    ENVIRONMENT = "colab"  # Options: "colab" | "kaggle"

# ----------------- Colab Environment Configuration -----------------
# High-speed local Colab storage paths (preferred to avoid Drive FUSE latency)
COLAB_LOCAL_DATASET_ROOT = Path("/content/DeepShield_FAST10")
COLAB_LOCAL_MANIFEST_PATH = Path("/content/DeepShield_FAST10_manifest_split.csv")

# Persistent Google Drive source & output paths
COLAB_DRIVE_DATASET_ROOT = Path("/content/drive/MyDrive/DeepShield_Datasets/Face_Cropped_FAST10")
COLAB_DRIVE_MANIFEST_PATH = Path("/content/drive/MyDrive/DeepShield_Datasets/Metadata/face_crop_FAST10_manifest_split.csv")
COLAB_MODEL_OUTPUT_ROOT = Path("/content/drive/MyDrive/DeepShield_V5")

# ----------------- Kaggle Environment Configuration ----------------
# In Kaggle, input datasets are mounted read-only under /kaggle/input/<dataset_name>/
KAGGLE_DATASET_ROOT = os.environ.get("KAGGLE_DATASET_ROOT", "")
KAGGLE_MANIFEST_PATH = os.environ.get("KAGGLE_MANIFEST_PATH", "")
KAGGLE_MODEL_OUTPUT_ROOT = Path("/kaggle/working/DeepShield_V5")

def auto_detect_kaggle_paths():
    """Auto-discovers Face_Cropped_FAST10 and manifest inside /kaggle/input."""
    ds_root = None
    manifest = None

    # Priority 1: Known exact dataset candidates on Kaggle
    known_ds_candidates = [
        Path("/kaggle/input/datasets/devyaninandankar/deepshield-face-dataset/Kaggle_Transfer/Face_Cropped_FAST10"),
        Path("/kaggle/input/deepshield-face-dataset/Kaggle_Transfer/Face_Cropped_FAST10"),
        Path("/kaggle/input/face_cropped_fast10/Face_Cropped_FAST10"),
        Path("/kaggle/input/Face_Cropped_FAST10")
    ]
    for cand in known_ds_candidates:
        if cand.is_dir():
            ds_root = cand
            break

    # Priority 1: Known exact manifest candidates on Kaggle
    known_mf_candidates = [
        Path("/kaggle/input/datasets/devyaninandankar/deepshield-face-dataset/Kaggle_Transfer/face_crop_FAST10_manifest_split.csv"),
        Path("/kaggle/input/deepshield-face-dataset/Kaggle_Transfer/face_crop_FAST10_manifest_split.csv"),
        Path("/kaggle/input/face_cropped_fast10/face_crop_FAST10_manifest_split.csv"),
        Path("/kaggle/input/face_crop_FAST10_manifest_split.csv")
    ]
    for cand in known_mf_candidates:
        if cand.is_file():
            manifest = cand
            break

    # Priority 2: Recursive scan of /kaggle/input
    input_base = Path("/kaggle/input")
    if input_base.exists():
        for root, dirs, files in os.walk(str(input_base)):
            if not ds_root:
                for d in dirs:
                    if d == "Face_Cropped_FAST10" or "fast10" in d.lower():
                        cand = Path(root) / d
                        if (cand / "train").is_dir():
                            ds_root = cand
                            break
            if not manifest:
                for f in files:
                    if ("manifest_split.csv" in f or "face_crop" in f) and f.endswith(".csv"):
                        manifest = Path(root) / f
                        break
            if ds_root and manifest:
                break

    return ds_root, manifest

# ----------------- Dynamic Path Resolution based on ENVIRONMENT ----
if ENVIRONMENT.lower() == "colab":
    # Prefer local fast storage if available, otherwise fallback to Google Drive
    if COLAB_LOCAL_DATASET_ROOT.is_dir() and COLAB_LOCAL_MANIFEST_PATH.is_file():
        print(f"[+] Using high-speed local Colab dataset storage: {COLAB_LOCAL_DATASET_ROOT}")
        DATASET_ROOT = COLAB_LOCAL_DATASET_ROOT
        MANIFEST_PATH = COLAB_LOCAL_MANIFEST_PATH
    else:
        print(f"[*] Local Colab dataset not detected, reading directly from Google Drive: {COLAB_DRIVE_DATASET_ROOT}")
        DATASET_ROOT = COLAB_DRIVE_DATASET_ROOT
        MANIFEST_PATH = COLAB_DRIVE_MANIFEST_PATH

    MODEL_OUTPUT_ROOT = COLAB_MODEL_OUTPUT_ROOT
    # Checkpoints & models save directly to Google Drive to protect against runtime disconnects
    if Path("/content/drive/MyDrive").exists():
        CHECKPOINT_DIR = MODEL_OUTPUT_ROOT / "checkpoints"
        LOCAL_MODEL_DIR = MODEL_OUTPUT_ROOT
        HISTORY_DIR = MODEL_OUTPUT_ROOT / "training"
    else:
        CHECKPOINT_DIR = Path("models/checkpoints")
        LOCAL_MODEL_DIR = Path("models")
        HISTORY_DIR = Path("training")
elif ENVIRONMENT.lower() == "kaggle":
    detected_ds, detected_mf = auto_detect_kaggle_paths()
    resolved_ds = Path(KAGGLE_DATASET_ROOT) if KAGGLE_DATASET_ROOT and Path(KAGGLE_DATASET_ROOT).is_dir() else detected_ds
    resolved_mf = Path(KAGGLE_MANIFEST_PATH) if KAGGLE_MANIFEST_PATH and Path(KAGGLE_MANIFEST_PATH).is_file() else detected_mf

    if not resolved_ds or not resolved_mf:
        print("[!] NOTICE for Kaggle: Verify dataset is attached or set KAGGLE_DATASET_ROOT and KAGGLE_MANIFEST_PATH.")

    DATASET_ROOT = resolved_ds if resolved_ds else Path("/kaggle/input")
    MANIFEST_PATH = resolved_mf if resolved_mf else Path("/kaggle/input/manifest.csv")
    MODEL_OUTPUT_ROOT = KAGGLE_MODEL_OUTPUT_ROOT
    CHECKPOINT_DIR = MODEL_OUTPUT_ROOT / "checkpoints"
    LOCAL_MODEL_DIR = MODEL_OUTPUT_ROOT
    HISTORY_DIR = MODEL_OUTPUT_ROOT / "training"
    print(f"[+] Kaggle Environment Detected:")
    print(f"    - Dataset Root:    {DATASET_ROOT}")
    print(f"    - Manifest Path:   {MANIFEST_PATH}")
    print(f"    - Output Root:     {MODEL_OUTPUT_ROOT}")
else:
    raise ValueError(f"Unsupported ENVIRONMENT '{ENVIRONMENT}'. Must be 'colab' or 'kaggle'.")

# Expected counts per split (used for sanity checks)
EXPECTED_COUNTS = {
    "train": 17208,
    "validation": 3688,
    "test": 3693
}

# Checkpoint paths and history output paths
STAGE1_CHECKPOINT = CHECKPOINT_DIR / "deepshield_v5_stage1_best.keras"
STAGE2_CHECKPOINT = CHECKPOINT_DIR / "deepshield_v5_stage2_best.keras"
FINAL_MODEL_PATH = LOCAL_MODEL_DIR / "deepshield_v5_best.keras"

HISTORY_STAGE1_CSV = HISTORY_DIR / "history_stage1.csv"
HISTORY_STAGE2_CSV = HISTORY_DIR / "history_stage2.csv"


# ==============================================================================
# 1. REPRODUCIBILITY & HARDWARE SETUP
# ==============================================================================
def set_seed(seed_value: int = SEED):
    """Sets deterministic seeds across Python, NumPy, and TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print(f"[+] Random seed set to: {seed_value}")


def check_environment_and_gpu():
    """Verifies Environment, Python, TensorFlow versions, and checks GPU availability."""
    print("=" * 70)
    print("DEEPSHIELD V5: ENVIRONMENT & HARDWARE CHECK")
    print("=" * 70)
    print(f"Environment:        {ENVIRONMENT.upper()}")
    print(f"TensorFlow Version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Availability:   True ({len(gpus)} device(s) found)")
        for i, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            device_name = details.get('device_name', gpu.name)
            print(f"GPU Device [{i}]:     {device_name}")
        print("[+] Hardware Check: GPU READY")
    else:
        print("GPU Availability:   False")
        print("\nGPU NOT AVAILABLE")
        print("[!] Do NOT run long training jobs on CPU.")
        if ENVIRONMENT.lower() == "colab":
            print("[!] On Colab: Select Runtime -> Change runtime type -> Hardware accelerator: T4 GPU.")
        else:
            print("[!] On Kaggle: Enable GPU in Settings -> Accelerator -> GPU T4 x2 or P100.")
        sys.exit(1)
    print("=" * 70)


def mount_google_drive(mount_point: str = "/content/drive"):
    """Mounts Google Drive when running in Colab environment."""
    if ENVIRONMENT.lower() != "colab":
        # Google Drive mounting is not performed in Kaggle
        return

    # In Google Colab, /content/drive/MyDrive exists only after drive is mounted.
    mydrive_path = Path(mount_point) / "MyDrive"
    if mydrive_path.exists():
        print(f"[+] Google Drive already mounted at '{mount_point}'.")
        return

    try:
        from google.colab import drive
        print("[*] Mounting Google Drive to '/content/drive'...")
        drive.mount(mount_point)
        print("[+] Google Drive successfully mounted.")
    except ImportError:
        print("[*] google.colab not available; running in local or standalone environment.")
    except Exception as exc:
        print(f"[!] Warning: Failed to mount Google Drive: {exc}")


# ==============================================================================
# 2. DATASET AUDITING & RESOLUTION
# ==============================================================================
def load_and_verify_manifest(manifest_path: Path) -> pd.DataFrame:
    """Loads manifest CSV, validates columns, and splits without re-randomizing."""
    if not manifest_path.exists():
        print(f"[!] Critical Error: Manifest not found at: {manifest_path}")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    print(f"[+] Loaded manifest with {len(df):,} records.")

    required_cols = {'label', 'split', 'video_group'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[!] Critical Error: Manifest missing required columns: {missing}")
        sys.exit(1)

    return df


def verify_video_group_leakage(df: pd.DataFrame):
    """Ensures zero overlap of video groups across train, validation, and test splits."""
    print("\n[*] Auditing video_group leakage across splits...")
    train_groups = set(df[df['split'] == 'train']['video_group'].dropna().unique())
    val_groups = set(df[df['split'] == 'validation']['video_group'].dropna().unique())
    test_groups = set(df[df['split'] == 'test']['video_group'].dropna().unique())

    train_val_leak = train_groups.intersection(val_groups)
    train_test_leak = train_groups.intersection(test_groups)
    val_test_leak = val_groups.intersection(test_groups)

    leakage_detected = False
    if train_val_leak:
        print(f"[!] LEAKAGE: {len(train_val_leak)} video groups shared between Train and Validation!")
        leakage_detected = True
    if train_test_leak:
        print(f"[!] LEAKAGE: {len(train_test_leak)} video groups shared between Train and Test!")
        leakage_detected = True
    if val_test_leak:
        print(f"[!] LEAKAGE: {len(val_test_leak)} video groups shared between Validation and Test!")
        leakage_detected = True

    if leakage_detected:
        print("[!] Data leakage detected. Aborting training to preserve evaluation integrity.")
        sys.exit(1)
    else:
        print("[+] PASS: Zero video group leakage found across train, validation, and test splits.")


def resolve_split_filepaths(df: pd.DataFrame, split_name: str, dataset_root: Path):
    """
    Robustly resolves actual image paths on disk for a given split using Path(row['face_path']).name.
    Populates 'resolved_path' in the training dataframe.
    """
    subset = df[df['split'] == split_name].copy()

    # Determine split directory, case-insensitively if needed
    split_dir = dataset_root / split_name
    if not split_dir.is_dir():
        for cand_name in [split_name.lower(), split_name.capitalize(), split_name.upper()]:
            cand_p = dataset_root / cand_name
            if cand_p.is_dir():
                split_dir = cand_p
                break

    # Build: basename -> actual Kaggle path (recursive scan)
    basename_to_path = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if split_dir.is_dir():
        for root, dirs, files in os.walk(str(split_dir)):
            for f in files:
                if Path(f).suffix.lower() in valid_exts:
                    full_p = os.path.join(root, f)
                    basename_to_path[f] = full_p
                    basename_to_path[f.lower()] = full_p

    resolved_paths = []
    labels = []
    missing_records = []

    for idx, row in subset.iterrows():
        label_val = int(row['label'])
        resolved = None

        # For every manifest row, use: Path(row["face_path"]).name as the image basename
        if 'face_path' in row and pd.notna(row['face_path']):
            bname = Path(str(row['face_path']).replace("\\", "/")).name.strip()
            p_flat = split_dir / bname
            if p_flat.is_file():
                resolved = str(p_flat.resolve())
            elif bname in basename_to_path:
                resolved = basename_to_path[bname]
            elif bname.lower() in basename_to_path:
                resolved = basename_to_path[bname.lower()]

        # Verification: ensure resolved path actually exists on disk
        if resolved and os.path.isfile(resolved):
            resolved_paths.append(resolved)
            labels.append(label_val)
        else:
            resolved_paths.append(None)
            missing_records.append({
                "index": idx,
                "face_path": row.get('face_path', ''),
                "bname": Path(str(row.get('face_path', ''))).name
            })

    # The training dataframe must contain: resolved_path
    subset['resolved_path'] = resolved_paths

    valid_subset = subset[subset['resolved_path'].notna()].copy()
    valid_paths = valid_subset['resolved_path'].tolist()
    valid_labels = valid_subset['label'].astype(int).tolist()
    missing_count = len(missing_records)

    if missing_count > 0:
        print(f"[!] Critical Error: {missing_count:,} missing image file(s) in split '{split_name}'.")
        print("First 20 missing face_path basenames:")
        for rec in missing_records[:20]:
            print(f"  - [Index {rec['index']}] face_path basename: {rec['bname']} (raw: {rec['face_path']})")
        print(f"    Split directory searched:  {split_dir}")
        sys.exit(1)

    return valid_subset, valid_paths, valid_labels, missing_count


def compute_training_class_weights(train_labels: list) -> dict:
    """
    Calculates balanced class weights ONLY from the training split labels.
    Formula: total_samples / (num_classes * count_for_class)
    """
    train_labels = np.array(train_labels)
    total_samples = len(train_labels)
    real_count = int(np.sum(train_labels == 0))
    fake_count = int(np.sum(train_labels == 1))

    # Avoid division by zero
    weight_real = total_samples / (2.0 * real_count) if real_count > 0 else 1.0
    weight_fake = total_samples / (2.0 * fake_count) if fake_count > 0 else 1.0

    class_weights = {0: float(weight_real), 1: float(weight_fake)}

    print("\n[*] Training Class Distribution & Weights (Calculated exclusively from train split):")
    print(f"    Real (0): {real_count:,} samples -> Class Weight: {weight_real:.4f}")
    print(f"    Fake (1): {fake_count:,} samples -> Class Weight: {weight_fake:.4f}")
    return class_weights


# ==============================================================================
# 3. PREPROCESSING & TF.DATA PIPELINES
# ==============================================================================
# PREPROCESSING DOCUMENTATION (Crucial for Inference Consistency):
# 1. Images are already face-cropped (FAST10); no YuNet face detection during training.
# 2. Images are loaded as raw RGB bytes via tf.io.read_file and tf.image.decode_jpeg/png.
# 3. Resized to 224 x 224 with bilinear interpolation.
# 4. EfficientNetB2 expects input in range [0, 255] as tf.keras.applications.efficientnet.preprocess_input
#    applies its own scaling internally or models include normalization layers. Here we pass float32 [0, 255].
# 5. NO double-normalization or BGR/RGB conversion mismatch is applied.
# 6. Training split receives moderate random augmentations; validation/test receive NO augmentation.

def create_data_augmentation_layers():
    """Moderate, forensic-safe data augmentations for training data."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.08, fill_mode="nearest", seed=SEED),
        layers.RandomZoom(0.08, fill_mode="nearest", seed=SEED),
        layers.RandomTranslation(0.05, 0.05, fill_mode="nearest", seed=SEED),
        layers.RandomContrast(0.1, seed=SEED),
    ], name="data_augmentation")


def parse_and_preprocess_image(file_path: tf.Tensor, label: tf.Tensor, is_training: bool = False, augment_layer=None):
    """Loads, decodes (JPEG optimized), resizes, and optionally augments an image."""
    img_bytes = tf.io.read_file(file_path)
    # decode_jpeg is significantly faster than generic decode_image
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img.set_shape((*IMAGE_SIZE, 3))
    img = tf.cast(img, tf.float32)  # Range [0, 255] float32 for EfficientNet

    if is_training and augment_layer is not None:
        img = augment_layer(img, training=True)

    return img, label


def build_tf_dataset(file_paths: list, labels: list, is_training: bool = False, batch_size: int = BATCH_SIZE, cache_file: str = "") -> tf.data.Dataset:
    """Builds a high-performance tf.data pipeline using prefetching, autotuning, and optional disk caching."""
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    augment_layer = create_data_augmentation_layers() if is_training else None

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), 4096), seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, lbl: parse_and_preprocess_image(path, lbl, is_training=is_training, augment_layer=augment_layer),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Safe validation disk caching to avoid redundant image reads and decoding
    if cache_file:
        dataset = dataset.cache(cache_file)
    elif not is_training and len(file_paths) <= 5000:
        # If no explicit file path, cache to local temp disk
        dataset = dataset.cache("/content/val_cache" if ENVIRONMENT.lower() == "colab" else "")

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# ==============================================================================
# 4. MODEL ARCHITECTURE
# ==============================================================================
def build_deepshield_v5(input_shape=(224, 224, 3)) -> tf.keras.Model:
    """
    Constructs the DeepShield V5 architecture:
    Input -> Preprocessing -> EfficientNetB2 -> GAP -> Dropout -> Dense -> Dropout -> Dense(1, sigmoid)
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # EfficientNet internal preprocessing: scales [0, 255] inputs appropriately
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    # Base pretrained backbone
    base_model = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )
    base_model.trainable = False  # Freeze backbone for Stage 1

    # Feature extraction & classification head
    features = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    x = layers.Dropout(0.3, name="head_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_intermediate")(x)
    x = layers.Dropout(0.2, name="head_dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction_sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DeepShield_V5")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float):
    """Compiles the model with BinaryCrossentropy and evaluation metrics."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc")
        ]
    )


# ==============================================================================
# 5. TRAINING STAGES & RESUME CAPABILITIES
# ==============================================================================
def setup_directories():
    """Ensures required model checkpoint and training directories exist."""
    STAGE1_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    STAGE2_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_STAGE1_CSV.parent.mkdir(parents=True, exist_ok=True)


def get_callbacks(checkpoint_path: Path, csv_log_path: Path):
    """Creates training callbacks monitoring validation AUC."""
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    csv_logger_cb = callbacks.CSVLogger(
        filename=str(csv_log_path),
        append=True
    )
    return [checkpoint_cb, early_stop_cb, reduce_lr_cb, csv_logger_cb]


def run_stage_1(model, train_ds, val_ds, class_weights):
    """STAGE 1: Train classification head with frozen backbone."""
    print("\n" + "=" * 70)
    print("STAGE 1: TRAINING CLASSIFICATION HEAD (Backbone Frozen)")
    print("=" * 70)

    # Resume check: if Stage 1 best checkpoint already exists, load and notify
    if STAGE1_CHECKPOINT.exists():
        print(f"[*] Found existing Stage 1 checkpoint at: {STAGE1_CHECKPOINT}")
        print("[*] Loading existing Stage 1 best model to allow resumption...")
        model = tf.keras.models.load_model(str(STAGE1_CHECKPOINT))
        return model

    compile_model(model, learning_rate=INITIAL_LEARNING_RATE)
    stage1_callbacks = get_callbacks(STAGE1_CHECKPOINT, HISTORY_STAGE1_CSV)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        class_weight=class_weights,
        callbacks=stage1_callbacks,
        verbose=1
    )
    return model


def run_stage_2(train_ds, val_ds, class_weights):
    """
    STAGE 2: Fine-tuning the top ~30% of EfficientNetB2 layers.
    BatchNormalization layers are explicitly kept frozen.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: FINE-TUNING (Unfreezing Upper EfficientNetB2 Layers)")
    print("=" * 70)

    # Resume check: if Stage 2 best checkpoint already exists, load and notify
    if STAGE2_CHECKPOINT.exists():
        print(f"[*] Found existing Stage 2 checkpoint at: {STAGE2_CHECKPOINT}")
        print("[*] Loading existing Stage 2 model...")
        model = tf.keras.models.load_model(str(STAGE2_CHECKPOINT))
        return model

    # Load best checkpoint from Stage 1
    if not STAGE1_CHECKPOINT.exists():
        print(f"[!] Error: Cannot start Stage 2 without Stage 1 checkpoint: {STAGE1_CHECKPOINT}")
        sys.exit(1)

    print(f"[+] Loading best Stage 1 checkpoint: {STAGE1_CHECKPOINT}")
    model = tf.keras.models.load_model(str(STAGE1_CHECKPOINT))

    # Retrieve EfficientNetB2 backbone inside model
    backbone = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        print("[!] Warning: Backbone layer not found by name, traversing model layers directly.")
        target_layers = model.layers
    else:
        backbone.trainable = True
        target_layers = backbone.layers

    total_layers = len(target_layers)
    num_unfreeze = int(total_layers * (FINE_TUNE_PERCENTAGE / 100.0))
    unfreeze_from = total_layers - num_unfreeze

    print(f"[+] Backbone total layers: {total_layers}")
    print(f"[+] Unfreezing top {FINE_TUNE_PERCENTAGE}% (~{num_unfreeze} layers) from layer index {unfreeze_from}...")

    # Freeze earlier layers and keep ALL BatchNormalization layers frozen
    for i, layer in enumerate(target_layers):
        if i < unfreeze_from or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Re-compile with reduced fine-tuning learning rate
    compile_model(model, learning_rate=FINE_TUNE_LEARNING_RATE)
    print(f"[+] Recompiled model with fine-tuning learning rate: {FINE_TUNE_LEARNING_RATE}")

    stage2_callbacks = get_callbacks(STAGE2_CHECKPOINT, HISTORY_STAGE2_CSV)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE2_EPOCHS,
        class_weight=class_weights,
        callbacks=stage2_callbacks,
        verbose=1
    )
    return model


# ==============================================================================
# 6. MAIN PIPELINE WORKFLOW
# ==============================================================================
def main():
    # 1. Setup deterministic seed and hardware
    set_seed(SEED)
    check_environment_and_gpu()

    # 2. Mount Google Drive if in Colab
    mount_google_drive()

    # Dynamically bind checkpoint and model paths to Google Drive once mounted
    global CHECKPOINT_DIR, LOCAL_MODEL_DIR, HISTORY_DIR, STAGE1_CHECKPOINT, STAGE2_CHECKPOINT, FINAL_MODEL_PATH, HISTORY_STAGE1_CSV, HISTORY_STAGE2_CSV
    if ENVIRONMENT.lower() == "colab" and Path("/content/drive/MyDrive").exists():
        CHECKPOINT_DIR = COLAB_MODEL_OUTPUT_ROOT / "checkpoints"
        LOCAL_MODEL_DIR = COLAB_MODEL_OUTPUT_ROOT
        HISTORY_DIR = COLAB_MODEL_OUTPUT_ROOT / "training"
        STAGE1_CHECKPOINT = CHECKPOINT_DIR / "deepshield_v5_stage1_best.keras"
        STAGE2_CHECKPOINT = CHECKPOINT_DIR / "deepshield_v5_stage2_best.keras"
        FINAL_MODEL_PATH = LOCAL_MODEL_DIR / "deepshield_v5_best.keras"
        HISTORY_STAGE1_CSV = HISTORY_DIR / "history_stage1.csv"
        HISTORY_STAGE2_CSV = HISTORY_DIR / "history_stage2.csv"
        print(f"[+] Output & checkpoints configured to persist directly on Google Drive: {COLAB_MODEL_OUTPUT_ROOT}")

    # 3. Ensure required directories exist
    setup_directories()

    # 3. Verify dataset root exists
    if not DATASET_ROOT.exists():
        print(f"[!] Critical Error: Dataset directory not found at: {DATASET_ROOT}")
        print("[!] Verify that your Google Drive is properly connected and paths are correct.")
        sys.exit(1)

    # 4. Load manifest and verify split / columns
    manifest_df = load_and_verify_manifest(MANIFEST_PATH)

    # 5. Check video group leakage across splits
    verify_video_group_leakage(manifest_df)

    # 6. Resolve and verify actual image paths on disk across all three splits using face_path
    print("\n[*] Auditing image path resolution across all dataset splits...")
    train_df, train_paths, train_labels, tr_missing = resolve_split_filepaths(manifest_df, "train", DATASET_ROOT)
    val_df, val_paths, val_labels, va_missing = resolve_split_filepaths(manifest_df, "validation", DATASET_ROOT)
    test_df, test_paths, test_labels, te_missing = resolve_split_filepaths(manifest_df, "test", DATASET_ROOT)

    total_missing = tr_missing + va_missing + te_missing

    print("\n" + "=" * 60)
    print("DEEPSHIELD V5: TRAINING PATH AUDIT")
    print("=" * 60)
    print("Path source: face_path\n")
    print(f"Train:      {len(train_paths):,} / {EXPECTED_COUNTS['train']:,}")
    print(f"Validation: {len(val_paths):,} / {EXPECTED_COUNTS['validation']:,}")
    print(f"Test:       {len(test_paths):,} / {EXPECTED_COUNTS['test']:,}\n")
    print(f"Missing: {total_missing}")
    print("=" * 60)

    if total_missing > 0:
        print(f"\n[!] Critical Error: {total_missing:,} missing file(s) detected during face_path resolution!")
        print("STOPPING IMMEDIATELY. Training aborted.")
        sys.exit(1)

    print("[+] PATH RESOLUTION AUDIT: PASS (All 24,589 images verified on disk). Continuing to training...\n")

    # 7. Calculate class weights solely from training split
    class_weights = compute_training_class_weights(train_labels)

    # 8. Build tf.data datasets directly using resolved_path column
    print("\n[*] Building tf.data pipelines directly from resolved_path (batch_size={})...".format(BATCH_SIZE))
    train_ds = build_tf_dataset(train_df['resolved_path'].tolist(), train_labels, is_training=True, batch_size=BATCH_SIZE)
    val_ds = build_tf_dataset(val_df['resolved_path'].tolist(), val_labels, is_training=False, batch_size=BATCH_SIZE)

    # 9. Build architecture
    print("\n[*] Initializing DeepShield V5 architecture (EfficientNetB2)...")
    model = build_deepshield_v5(input_shape=(*IMAGE_SIZE, 3))
    model.summary(show_trainable=True)

    # 10. Stage 1: Frozen backbone training
    model = run_stage_1(model, train_ds, val_ds, class_weights)

    # 11. Stage 2: Fine-tuning upper ~30% layers
    model = run_stage_2(train_ds, val_ds, class_weights)

    # 12. Save final best V5 model
    print("\n" + "=" * 70)
    print("SAVING FINAL DEEPSHIELD V5 MODEL")
    print("=" * 70)
    # Check if Stage 2 checkpoint exists to copy as best final V5 model
    if STAGE2_CHECKPOINT.exists():
        best_model = tf.keras.models.load_model(str(STAGE2_CHECKPOINT))
        best_model.save(str(FINAL_MODEL_PATH))
        print(f"[+] Successfully saved final DeepShield V5 model to: {FINAL_MODEL_PATH}")
    else:
        model.save(str(FINAL_MODEL_PATH))
        print(f"[+] Saved model to: {FINAL_MODEL_PATH}")

    # 13. Backup/Persist final model and history to environment-specific destinations
    try:
        if ENVIRONMENT.lower() == "colab" and (Path("/content/drive") / "MyDrive").exists():
            MODEL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            drive_dest = MODEL_OUTPUT_ROOT / FINAL_MODEL_PATH.name
            import shutil
            if FINAL_MODEL_PATH.exists() and (FINAL_MODEL_PATH != drive_dest):
                shutil.copy2(FINAL_MODEL_PATH, drive_dest)
                print(f"[+] Backed up final V5 model to Google Drive: {drive_dest}")
            else:
                print(f"[+] Final V5 model saved directly to Google Drive: {FINAL_MODEL_PATH}")

            # Also backup training histories to Google Drive
            hist_dir = MODEL_OUTPUT_ROOT / "training_history"
            hist_dir.mkdir(parents=True, exist_ok=True)
            if HISTORY_STAGE1_CSV.exists() and (HISTORY_STAGE1_CSV != hist_dir / HISTORY_STAGE1_CSV.name):
                shutil.copy2(HISTORY_STAGE1_CSV, hist_dir / HISTORY_STAGE1_CSV.name)
            if HISTORY_STAGE2_CSV.exists() and (HISTORY_STAGE2_CSV != hist_dir / HISTORY_STAGE2_CSV.name):
                shutil.copy2(HISTORY_STAGE2_CSV, hist_dir / HISTORY_STAGE2_CSV.name)
            print(f"[+] Training histories persisted to Google Drive: {hist_dir}")
        elif ENVIRONMENT.lower() == "kaggle":
            eval_dir = MODEL_OUTPUT_ROOT / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            print(f"[+] Kaggle artifacts saved in {MODEL_OUTPUT_ROOT}:")
            print(f"    - Final Model:        {FINAL_MODEL_PATH}")
            print(f"    - Stage 1 Checkpoint: {STAGE1_CHECKPOINT}")
            print(f"    - Stage 2 Checkpoint: {STAGE2_CHECKPOINT}")
            print(f"    - Stage 1 History:    {HISTORY_STAGE1_CSV}")
            print(f"    - Stage 2 History:    {HISTORY_STAGE2_CSV}")
    except Exception as save_err:
        print(f"[!] Note: Environment model persistence notice: {save_err}")

    # 14. Evaluation on test split and persist evaluation results
    try:
        eval_dir = MODEL_OUTPUT_ROOT / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 70)
        print("RUNNING FINAL TEST SPLIT EVALUATION")
        print("=" * 70)
        test_ds = build_tf_dataset(test_df['resolved_path'].tolist(), test_labels, is_training=False, batch_size=BATCH_SIZE)
        eval_results = model.evaluate(test_ds, return_dict=True, verbose=1)

        import json
        eval_json_path = eval_dir / "test_evaluation_results.json"
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in eval_results.items()}, f, indent=2)
        print(f"[+] Evaluation results saved to: {eval_json_path}")
        for m_name, m_val in eval_results.items():
            print(f"    - {m_name}: {m_val:.4f}")
    except Exception as eval_err:
        print(f"[*] Test evaluation notice: {eval_err}")

    print("[+] Training histories saved to:")
    print(f"    - {HISTORY_STAGE1_CSV}")
    print(f"    - {HISTORY_STAGE2_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
