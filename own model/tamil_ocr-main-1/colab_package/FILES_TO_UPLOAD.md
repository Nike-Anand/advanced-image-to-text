# Files to Upload to Google Colab

## Essential Files (Must Upload)

### 1. Training Scripts (4 files)
- ✅ `train_handwritten.py` - Main training script
- ✅ `config_handwritten.py` - Configuration
- ✅ `char_mapping.py` - Character mapping (156 classes)
- ✅ `tamil_char_dataset.py` - Dataset loader

### 2. Model Files (from ocr_tamil/)
Copy these files maintaining directory structure:

```
ocr_tamil/
├── strhub/
│   ├── models/
│   │   ├── base.py           ⚠️ MODIFIED - Use the version from this project!
│   │   ├── utils.py
│   │   └── parseq/
│   │       ├── __init__.py
│   │       ├── system.py
│   │       └── modules.py
│   └── data/
│       ├── __init__.py
│       └── utils.py
```

**Important**: The `base.py` file has been modified with:
- `configure_optimizers()` method
- `on_validation_epoch_end()` instead of `validation_epoch_end()`

### 3. Optional but Recommended
- `checkpoints/tamil-char-epoch=29-val_accuracy=97.45.ckpt` - Pretrained model (~500MB)

---

## Quick Checklist

- [ ] `train_handwritten.py`
- [ ] `config_handwritten.py`
- [ ] `char_mapping.py`
- [ ] `tamil_char_dataset.py`
- [ ] `ocr_tamil/strhub/models/base.py` ⚠️ **MODIFIED**
- [ ] `ocr_tamil/strhub/models/parseq/system.py`
- [ ] `ocr_tamil/strhub/models/parseq/modules.py`
- [ ] `ocr_tamil/strhub/models/utils.py`
- [ ] `ocr_tamil/strhub/data/utils.py`
- [ ] `checkpoints/*.ckpt` (optional)

---

## Total Size
- Core files: ~1MB
- With checkpoint: ~500MB
- Dataset: Upload to Google Drive separately (50GB)

---

## See COLAB_TRAINING_GUIDE.md for complete setup instructions!
