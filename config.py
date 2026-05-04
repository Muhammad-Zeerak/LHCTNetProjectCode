import os
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM SEED
# ═══════════════════════════════════════════════════════════════════════════════
SEED: int = 42

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.name == "nt":  # Windows
    NUM_WORKERS: int = 0
else:
    NUM_WORKERS: int = min(os.cpu_count() or 1, 8)
PIN_MEMORY: bool = torch.cuda.is_available()
PERSISTENT_WORKERS: bool = True if NUM_WORKERS > 0 else False
CUDNN_BENCHMARK: bool = True
USE_AMP: bool = torch.cuda.is_available()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA PATHS
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR: str = "data"

# APTOS 2019
APTOS_DIR: str = os.path.join(DATA_DIR, "aptos2019")
APTOS_TRAIN_CSV: str = os.path.join(APTOS_DIR, "train.csv")
APTOS_TEST_CSV: str = os.path.join(APTOS_DIR, "test.csv")
APTOS_TRAIN_IMAGES: str = os.path.join(APTOS_DIR, "train_images")
APTOS_TEST_IMAGES: str = os.path.join(APTOS_DIR, "test_images")

# IDRiD
IDRID_DIR: str = os.path.join(DATA_DIR, "idrid")
IDRID_TRAIN_LABELS: str = os.path.join(
    IDRID_DIR, "IDRiD_Disease Grading_Training Labels.csv"
)
IDRID_TEST_LABELS: str = os.path.join(
    IDRID_DIR, "IDRiD_Disease Grading_Testing Labels.csv"
)
IDRID_TRAIN_IMAGES: str = os.path.join(IDRID_DIR, "images", "training")
IDRID_TEST_IMAGES: str = os.path.join(IDRID_DIR, "images", "testing")
IDRID_LESION_MASKS_DIR: str = os.path.join(IDRID_DIR, "lesion_masks")

# EyePACS
EYEPACS_DIR: str = os.path.join(DATA_DIR, "eyepacs")
EYEPACS_LABELS_CSV: str = os.path.join(EYEPACS_DIR, "trainLabels.csv")
EYEPACS_IMAGES: str = os.path.join(EYEPACS_DIR, "images")

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT PATHS
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR: str = "results"
CHECKPOINT_DIR: str = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR: str = os.path.join(RESULTS_DIR, "logs")
PREDICTION_DIR: str = os.path.join(RESULTS_DIR, "predictions")
EVAL_DIR: str = os.path.join(RESULTS_DIR, "evaluation")
FIGURE_DIR: str = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR: str = os.path.join(RESULTS_DIR, "tables")

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
NUM_CLASSES: int = 5  # ICDR grades 0–4
CLASS_NAMES: list = [
    "Grade 0 (No DR)",
    "Grade 1 (Mild NPDR)",
    "Grade 2 (Moderate NPDR)",
    "Grade 3 (Severe NPDR)",
    "Grade 4 (Proliferative DR)",
]
# NHS DESP referral boundary: Grade ≥ 2 is referable
REFERABLE_THRESHOLD: int = 2

# Train/Val/Test split ratios for APTOS 2019
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# EyePACS cross-dataset subset size
EYEPACS_SUBSET_SIZE: int = 3000

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
IMG_SIZE: int = 224  # EfficientNet-B0 and Swin-Tiny default input resolution
INTERPOLATION: str = "bilinear"

# ImageNet normalisation
IMAGENET_MEAN: tuple = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple = (0.229, 0.224, 0.225)

# CLAHE parameters
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: tuple = (8, 8)

# Data augmentation
AUG_HFLIP_PROB: float = 0.5
AUG_VFLIP_PROB: float = 0.5
AUG_ROTATION_DEGREES: int = 15
AUG_BRIGHTNESS_FACTOR: float = 0.2
AUG_CONTRAST_FACTOR: float = 0.2

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
# Cross-attention fusion embedding dimension
FUSION_DIM: int = 256
# MLP hidden dimension in classifier head
CLASSIFIER_HIDDEN_DIM: int = 512
# Dropout in classifier
CLASSIFIER_DROPOUT: float = 0.3

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
MAX_EPOCHS: int = 25
BATCH_SIZE: int = 32  # Starting batch size; halved on OOM

LEARNING_RATE: float = 1e-4
BACKBONE_LR_RATIO: float = 0.1  # Backbone LR = LEARNING_RATE * 0.1
WEIGHT_DECAY: float = 1e-2
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.999

# Learning rate schedule
LINEAR_WARMUP_EPOCHS: int = 2
COSINE_MIN_LR: float = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE: int = 5
EARLY_STOPPING_METRIC: str = "qwk"  # Quadratic weighted kappa

# Label smoothing
LABEL_SMOOTHING: float = 0.1

# Gradient clipping
GRAD_CLIP_MAX_NORM: float = 1.0

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_PARAMS: int = 25_000_000
PRUNING_RATIO_SWIN_LATE: float = 0.20  # Swin stages 3–4
PRUNING_RATIO_CNN: float = 0.10  # EfficientNet-B0
PRUNING_RATIO_STEP: float = 0.05  # Increment if target not reached
QAT_EPOCHS: int = 10  # Quantisation-aware training epochs after pruning
COMPRESSION_ACCURACY_THRESHOLD: float = 0.90  # 90% of baseline QWK

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
BOOTSTRAP_N_ITERATIONS: int = 500  # For confidence intervals
BOOTSTRAP_CI_ALPHA: float = 0.05  # 95% confidence interval
CPU_INFERENCE_PASSES: int = 50  # For timing measurement
CROSS_DATASET_DROP_THRESHOLD: float = 0.10  # 10% relative drop warning

# ═══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
GRADCAM_TARGET_LAYER: str = "cnn_stream.features.8"  # EfficientNet-B0 final conv
NUM_EXPLAINABILITY_EXAMPLES: int = 4  # Per class: 2 correct, 2 incorrect

# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════
ABLATION_VARIANTS: list = [
    "efficientnet_only",   # CNN stream + GAP + classifier
    "swin_only",           # Transformer stream + GAP + classifier
    "lhctnet_concat",      # Both streams, concatenation fusion
    "lhctnet_full",        # Both streams, cross-attention fusion (proposed)
]

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
FIGURE_DPI: int = 300
FIGURE_FORMAT: list = ["png", "pdf"]
FONT_SIZE_MIN: int = 12
COLOUR_PALETTE: str = "Set2"  # Seaborn palette for consistent styling

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
LOG_LEVEL: str = "INFO"
LOG_FILE: str = os.path.join(LOG_DIR, "run_log.txt")