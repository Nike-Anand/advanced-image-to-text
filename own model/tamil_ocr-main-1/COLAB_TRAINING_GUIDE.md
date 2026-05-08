# Training on Google Colab - Setup Guide

## Files to Upload to Google Colab

### Essential Training Files (Upload these to Colab)

1. **Training Scripts**
   - `train_handwritten.py` - Main training script
   - `config_handwritten.py` - Configuration file
   - `char_mapping.py` - Character mapping (156 classes)
   - `tamil_char_dataset.py` - Dataset loader

2. **Model Files** (from `ocr_tamil/` directory)
   - `ocr_tamil/strhub/models/base.py` - Base model classes
   - `ocr_tamil/strhub/models/parseq/system.py` - PARSeq model
   - `ocr_tamil/strhub/models/parseq/modules.py` - Model modules
   - `ocr_tamil/strhub/data/utils.py` - Data utilities
   - `ocr_tamil/strhub/models/utils.py` - Model utilities

3. **Pretrained Checkpoint** (Optional but recommended)
   - `checkpoints/tamil-char-epoch=29-val_accuracy=97.45.ckpt` - Your trained model (to continue training)

4. **Dataset** (50GB)
   - Upload to Google Drive and mount in Colab
   - Or upload directly to Colab (slower)

---

## Step-by-Step Colab Setup

### 1. Create Colab Notebook

Create a new notebook with this structure:

```python
# Cell 1: Mount Google Drive (if dataset is on Drive)
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install dependencies
!pip install pytorch-lightning==2.0.0
!pip install timm
!pip install seaborn scikit-learn

# Cell 3: Setup directory structure
!mkdir -p tamil_ocr_training
%cd tamil_ocr_training

# Cell 4: Upload training files
# Use Colab's file upload or copy from Drive
```

---

## Minimal File Structure for Colab

```
tamil_ocr_training/
├── train_handwritten.py          # Main training script
├── config_handwritten.py          # Configuration
├── char_mapping.py                # Character mapping
├── tamil_char_dataset.py          # Dataset loader
├── ocr_tamil/
│   └── strhub/
│       ├── models/
│       │   ├── base.py           # Base model (MODIFIED)
│       │   ├── utils.py
│       │   └── parseq/
│       │       ├── system.py     # PARSeq model
│       │       └── modules.py
│       └── data/
│           └── utils.py          # Tokenizer, etc.
├── dataset/                       # Your 50GB dataset
│   ├── train/
│   │   ├── 0/
│   │   ├── 1/
│   │   └── ...
│   └── test/
│       ├── 0/
│       ├── 1/
│       └── ...
└── checkpoints/                   # (Optional) Pretrained model
    └── tamil-char-epoch=29-val_accuracy=97.45.ckpt
```

---

## Files You MUST Upload

### Core Files (Small, ~50KB total)
1. `train_handwritten.py`
2. `config_handwritten.py`
3. `char_mapping.py`
4. `tamil_char_dataset.py`

### Model Files (from ocr_tamil/, ~500KB total)
5. `ocr_tamil/strhub/models/base.py` ⚠️ **MODIFIED VERSION**
6. `ocr_tamil/strhub/models/parseq/system.py`
7. `ocr_tamil/strhub/models/parseq/modules.py`
8. `ocr_tamil/strhub/models/utils.py`
9. `ocr_tamil/strhub/data/utils.py`

### Optional but Recommended
10. `checkpoints/tamil-char-epoch=29-val_accuracy=97.45.ckpt` (~500MB) - Continue training from this

---

## Recommended Approach for 50GB Dataset

### Option 1: Google Drive (Recommended)
1. Upload dataset to Google Drive
2. Mount Drive in Colab
3. Update paths in `config_handwritten.py`:
   ```python
   TRAIN_DIR = "/content/drive/MyDrive/tamil_dataset/train"
   TEST_DIR = "/content/drive/MyDrive/tamil_dataset/test"
   ```

### Option 2: Direct Upload to Colab
- Slower but works
- Dataset will be lost when runtime disconnects

### Option 3: Download from URL
If dataset is hosted online:
```python
!wget https://your-dataset-url.com/dataset.zip
!unzip dataset.zip
```

---

## Complete Colab Notebook Template

```python
# ============================================================================
# CELL 1: Setup Environment
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# CELL 2: Install Dependencies
# ============================================================================
!pip install pytorch-lightning==2.0.0 timm seaborn scikit-learn -q

# ============================================================================
# CELL 3: Create Directory Structure
# ============================================================================
!mkdir -p tamil_ocr_training/ocr_tamil/strhub/models/parseq
!mkdir -p tamil_ocr_training/ocr_tamil/strhub/data
!mkdir -p tamil_ocr_training/checkpoints
!mkdir -p tamil_ocr_training/logs

%cd tamil_ocr_training

# ============================================================================
# CELL 4: Upload Training Files
# ============================================================================
# Upload these files using Colab's file upload:
# - train_handwritten.py
# - config_handwritten.py
# - char_mapping.py
# - tamil_char_dataset.py
# - ocr_tamil/strhub/models/base.py
# - ocr_tamil/strhub/models/parseq/system.py
# - ocr_tamil/strhub/models/parseq/modules.py
# - ocr_tamil/strhub/models/utils.py
# - ocr_tamil/strhub/data/utils.py

from google.colab import files
print("Upload training files...")
# Manually upload files or copy from Drive

# ============================================================================
# CELL 5: Copy Files from Drive (if already uploaded there)
# ============================================================================
# Copy training scripts
!cp /content/drive/MyDrive/tamil_ocr_files/*.py .
!cp -r /content/drive/MyDrive/tamil_ocr_files/ocr_tamil .

# Copy pretrained checkpoint (optional)
!cp /content/drive/MyDrive/tamil_ocr_files/checkpoints/*.ckpt checkpoints/

# ============================================================================
# CELL 6: Update Config for Colab
# ============================================================================
# Edit config_handwritten.py to point to your dataset location
!sed -i 's|c:\\\\D\\\\Projects\\\\image to text\\\\own model\\\\archive|/content/drive/MyDrive/tamil_dataset|g' config_handwritten.py

# Or manually edit:
# TRAIN_DIR = "/content/drive/MyDrive/tamil_dataset/train"
# TEST_DIR = "/content/drive/MyDrive/tamil_dataset/test"

# ============================================================================
# CELL 7: Verify GPU
# ============================================================================
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"CUDA Version: {torch.version.cuda}")

# ============================================================================
# CELL 8: Test Dataset Loading
# ============================================================================
!python -c "from tamil_char_dataset import create_dataloaders; train, val = create_dataloaders('/content/drive/MyDrive/tamil_dataset/train', '/content/drive/MyDrive/tamil_dataset/test', batch_size=16, subset_size=100); print(f'Train batches: {len(train)}, Val batches: {len(val)}')"

# ============================================================================
# CELL 9: Start Training (Small Test First)
# ============================================================================
# Test with small subset first
!python train_handwritten.py --epochs 5 --batch_size 16

# ============================================================================
# CELL 10: Full Training (50GB Dataset)
# ============================================================================
# After test succeeds, run full training
!python train_handwritten.py --epochs 30 --batch_size 32 --full_dataset

# ============================================================================
# CELL 11: Monitor Training (Optional - Run in parallel)
# ============================================================================
# Install tensorboard
%load_ext tensorboard
%tensorboard --logdir logs

# ============================================================================
# CELL 12: Save Checkpoints to Drive
# ============================================================================
# Copy best model back to Drive
!cp checkpoints/tamil-char-epoch*.ckpt /content/drive/MyDrive/tamil_ocr_checkpoints/

# ============================================================================
# CELL 13: Evaluate Model
# ============================================================================
!python evaluate_handwritten.py --checkpoint checkpoints/best_model.ckpt
```

---

## Important Configuration Changes for Colab

### In `config_handwritten.py`:

```python
# Update these paths for Colab
TRAIN_DIR = "/content/drive/MyDrive/tamil_dataset/train"
TEST_DIR = "/content/drive/MyDrive/tamil_dataset/test"
CHECKPOINT_DIR = "/content/tamil_ocr_training/checkpoints"
LOG_DIR = "/content/tamil_ocr_training/logs"

# Adjust for Colab GPU (usually T4 or V100)
BATCH_SIZE = 32  # Increase if you have V100
NUM_WORKERS = 2  # Colab has limited CPU cores
USE_FP16 = True  # Definitely use mixed precision

# For 50GB dataset
SUBSET_SIZE = None  # Use full dataset
NUM_EPOCHS = 30  # Or more
```

---

## Files to Create a ZIP Package

Create a ZIP file with these files to upload to Colab:

```bash
# On your local machine, create a package
tamil_ocr_colab.zip
├── train_handwritten.py
├── config_handwritten.py
├── char_mapping.py
├── tamil_char_dataset.py
└── ocr_tamil/
    └── strhub/
        ├── models/
        │   ├── base.py
        │   ├── utils.py
        │   └── parseq/
        │       ├── __init__.py
        │       ├── system.py
        │       └── modules.py
        └── data/
            ├── __init__.py
            └── utils.py
```

---

## Expected Training Time on Colab

### With 50GB Dataset (~500K images):

| GPU Type | Batch Size | Time per Epoch | Total (30 epochs) |
|----------|------------|----------------|-------------------|
| T4 (Free) | 16 | ~30-40 min | ~15-20 hours |
| T4 (Free) | 32 | ~20-30 min | ~10-15 hours |
| V100 (Pro) | 64 | ~10-15 min | ~5-7 hours |
| A100 (Pro+) | 128 | ~5-8 min | ~2.5-4 hours |

**Note**: Free Colab has 12-hour runtime limit. You'll need Colab Pro for 50GB dataset training.

---

## Tips for Colab Training

1. **Use Colab Pro**: Free tier will timeout before training completes
2. **Save Checkpoints Frequently**: Set `SAVE_EVERY_N_EPOCHS = 5`
3. **Resume Training**: Use `--pretrained` flag to continue from checkpoint
4. **Monitor Progress**: Use TensorBoard in separate cell
5. **Backup to Drive**: Auto-save checkpoints to Google Drive

---

## Quick Start Commands

```bash
# 1. Upload files to Colab
# 2. Mount Drive
# 3. Install dependencies
# 4. Run training:

python train_handwritten.py --epochs 30 --batch_size 32 --full_dataset
```

---

## Summary

**Minimum files needed**: 9 files (~1MB total)
**Recommended**: Include pretrained checkpoint (~500MB)
**Dataset**: Upload to Google Drive (50GB)
**Expected accuracy**: 95-98% with 50GB dataset
**Training time**: 5-20 hours depending on GPU

Good luck with your training! 🚀
