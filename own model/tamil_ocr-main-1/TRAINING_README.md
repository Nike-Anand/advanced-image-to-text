# Tamil Handwritten Character Recognition Training

This project fine-tunes the PARSeq-based Tamil OCR model on handwritten Tamil character dataset.

## Dataset Structure

The dataset contains 156 Tamil character classes organized in directories:
- **Training**: 50,296 images in `archive/train/`
- **Testing**: 28,080 images in `archive/test/`

Each class is in a separate subdirectory (0-155) containing BMP images.

## Requirements

### GPU Requirements
**IMPORTANT**: This training requires a CUDA-enabled GPU. Training on CPU will be extremely slow.

Check GPU availability:
```bash
python check_gpu.py
```

### Python Dependencies
```bash
pip install torch torchvision pytorch-lightning
pip install pillow numpy matplotlib seaborn scikit-learn tqdm
```

## Quick Start

### 1. Check GPU Configuration
```bash
python check_gpu.py
```

### 2. Test Dataset Loading
```bash
python tamil_char_dataset.py
```

### 3. Quick Training Test (Fast Dev Run)
Test the pipeline with 1 batch:
```bash
python train_handwritten.py --fast_dev_run
```

### 4. Train on Small Subset (Recommended First)
Train on 1000 samples for 10 epochs:
```bash
python train_handwritten.py --epochs 10
```
(Edit `config_handwritten.py` and set `SUBSET_SIZE = 1000`)

### 5. Full Training
Remove subset limitation and train on full dataset:
```bash
python train_handwritten.py --epochs 50 --full_dataset
```

## Training Configuration

Edit `config_handwritten.py` to customize:

```python
# Key parameters
BATCH_SIZE = 32          # Reduce if GPU memory is low
NUM_EPOCHS = 50          # Number of training epochs
LEARNING_RATE = 1e-4     # Learning rate
USE_FP16 = True          # Mixed precision training (faster)
FREEZE_ENCODER = False   # Freeze encoder weights
USE_AUGMENTATION = True  # Data augmentation
```

## Command-Line Arguments

```bash
python train_handwritten.py [OPTIONS]

Options:
  --epochs INT              Number of epochs (overrides config)
  --batch_size INT          Batch size (overrides config)
  --pretrained PATH         Path to pretrained model checkpoint
  --fast_dev_run           Run 1 batch for quick testing
  --full_dataset           Use full dataset (ignore SUBSET_SIZE)
```

## Monitoring Training

### TensorBoard
View training progress in real-time:
```bash
tensorboard --logdir logs
```
Then open http://localhost:6006 in your browser.

### Checkpoints
Models are saved to `checkpoints/` directory:
- `tamil-char-{epoch}-{val_accuracy}.ckpt` - Best models
- `tamil-char-last.ckpt` - Latest checkpoint
- `tamil-char-epoch-{epoch}.ckpt` - Periodic checkpoints

## Evaluation

Evaluate trained model on test set:
```bash
python evaluate_handwritten.py --checkpoint checkpoints/best_model.ckpt
```

This generates:
- Overall accuracy metrics
- Per-class accuracy
- Confusion matrix (`evaluation_results/confusion_matrix.png`)
- Error analysis
- Predictions CSV (`evaluation_results/predictions.csv`)

## GPU Configuration

The training script automatically:
- ✓ Detects and configures GPU
- ✓ Enables cuDNN auto-tuner for optimal performance
- ✓ Uses mixed precision (FP16) training if available
- ✓ Displays GPU memory and utilization
- ✓ **Forces GPU usage** (will prompt if no GPU detected)

### GPU Memory Management

If you encounter out-of-memory errors:
1. Reduce `BATCH_SIZE` in `config_handwritten.py`
2. Disable FP16: Set `USE_FP16 = False`
3. Reduce model size (not recommended)

## Training Tips

### First Time Training
1. Start with `--fast_dev_run` to verify everything works
2. Train on subset (1000-5000 samples) for 10 epochs
3. Check results and adjust hyperparameters
4. Train on full dataset

### Hyperparameter Tuning
- **Learning Rate**: Start with 1e-4, reduce if loss oscillates
- **Batch Size**: Larger = faster but needs more GPU memory
- **Freeze Encoder**: Set `FREEZE_ENCODER = True` to train only decoder (faster)
- **Data Augmentation**: Disable if overfitting is not an issue

### Expected Training Time
On a modern GPU (e.g., RTX 3080):
- **Subset (1000 samples)**: ~5-10 minutes for 10 epochs
- **Full dataset (50K samples)**: ~2-4 hours for 50 epochs

## File Structure

```
tamil_ocr-main-1/
├── char_mapping.py              # Character ID to Tamil char mapping
├── tamil_char_dataset.py        # Dataset loader
├── config_handwritten.py        # Training configuration
├── train_handwritten.py         # Main training script
├── evaluate_handwritten.py      # Evaluation script
├── check_gpu.py                 # GPU availability checker
├── checkpoints/                 # Saved models
├── logs/                        # TensorBoard logs
└── evaluation_results/          # Evaluation outputs
```

## Troubleshooting

### No GPU Detected
```
WARNING: CUDA is not available!
```
**Solution**: Install CUDA-enabled PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Error
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in `config_handwritten.py`:
```python
BATCH_SIZE = 16  # or even 8
```

### Slow Training
- Ensure GPU is being used (check with `check_gpu.py`)
- Enable FP16: `USE_FP16 = True`
- Increase batch size if GPU memory allows
- Enable cuDNN benchmark (already enabled by default)

## Character Mapping

The dataset uses class IDs (0-155) mapped to Tamil characters:
- Independent vowels: அ, ஆ, இ, ஈ, உ, ஊ, எ, ஏ, ஐ, ஒ, ஓ, ஔ
- Consonants: க, ங, ச, ஞ, ட, ண, த, ந, ப, ம, ய, ர, ல, வ, ழ, ள, ற, ன
- Compound characters and more...

See `char_mapping.py` for complete mapping.

## Next Steps

After training:
1. Evaluate model performance with `evaluate_handwritten.py`
2. Analyze confusion matrix to identify problematic characters
3. Fine-tune hyperparameters based on results
4. Integrate trained model into your OCR pipeline

## Support

For issues or questions, check:
- GPU setup: Run `python check_gpu.py`
- Dataset loading: Run `python tamil_char_dataset.py`
- Quick test: Run `python train_handwritten.py --fast_dev_run`
