"""
Training Configuration for Tamil Handwritten Character Recognition
"""

# Dataset paths
TRAIN_DIR = r"c:\D\Projects\image to text\own model\archive\train"
TEST_DIR = r"c:\D\Projects\image to text\own model\archive\test"

# Model paths
PRETRAINED_MODEL_PATH = None  # Set to checkpoint path if loading pretrained model
CHECKPOINT_DIR = r"c:\D\Projects\image to text\own model\tamil_ocr-main-1\checkpoints"
LOG_DIR = r"c:\D\Projects\image to text\own model\tamil_ocr-main-1\logs"

# Training hyperparameters
BATCH_SIZE = 16  # Reduced for faster initial testing
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_PCT = 0.1  # Percentage of training for warmup

# Model hyperparameters
IMG_SIZE = (32, 128)  # (height, width)
PATCH_SIZE = (4, 8)
EMBED_DIM = 256  # Reduced from 384 for faster training
ENC_NUM_HEADS = 4  # Reduced from 6
ENC_MLP_RATIO = 4
ENC_DEPTH = 6  # Reduced from 12
DEC_NUM_HEADS = 8  # Reduced from 12
DEC_MLP_RATIO = 4
DEC_DEPTH = 1
PERM_NUM = 6
PERM_FORWARD = True
PERM_MIRRORED = True
DECODE_AR = True
REFINE_ITERS = 1
DROPOUT = 0.1
MAX_LABEL_LENGTH = 25  # Maximum sequence length (needs to be > 1 for PARSeq)

# Training settings
NUM_WORKERS = 4
PIN_MEMORY = True
GRADIENT_CLIP_VAL = 1.0
ACCUMULATE_GRAD_BATCHES = 1

# Mixed precision training
USE_FP16 = True  # Set to False if you encounter issues

# Early stopping
PATIENCE = 10  # Stop if no improvement for this many epochs

# Checkpoint settings
SAVE_TOP_K = 3  # Save top 3 best models
SAVE_EVERY_N_EPOCHS = 5  # Save checkpoint every N epochs

# Validation
VAL_CHECK_INTERVAL = 1.0  # Validate every epoch
LIMIT_VAL_BATCHES = 1.0  # Use full validation set

# Quick testing (set to True for fast development)
FAST_DEV_RUN = False  # Run 1 batch for quick testing
SUBSET_SIZE = None  # Use 1000 samples for initial training (set to None for full dataset)

# Freeze encoder
FREEZE_ENCODER = False  # Set to True to freeze encoder during training

# Data augmentation
USE_AUGMENTATION = True
