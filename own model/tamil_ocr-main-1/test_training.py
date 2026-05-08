"""
Simple test script to verify GPU training works
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

import torch
import pytorch_lightning as pl
from tamil_char_dataset import create_dataloaders
from char_mapping import get_charset_string
import config_handwritten as config
from ocr_tamil.strhub.models.parseq.system import PARSeq

print("="*80)
print("SIMPLE GPU TRAINING TEST")
print("="*80)

# Check GPU
print(f"\nGPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Create small dataset
print("\nCreating dataloaders...")
train_loader, val_loader = create_dataloaders(
    train_dir=config.TRAIN_DIR,
    test_dir=config.TEST_DIR,
    batch_size=4,
    num_workers=0,  # Set to 0 for testing
    img_size=config.IMG_SIZE,
    subset_size=100  # Small subset for testing
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Get charset
charset = get_charset_string()
print(f"\nCharset size: {len(charset)}")

# Create model
print("\nCreating model...")
model = PARSeq(
    charset_train=charset,
    charset_test=charset,
    max_label_length=1,
    batch_size=4,
    lr=1e-4,
    warmup_pct=0.1,
    weight_decay=1e-4,
    img_size=config.IMG_SIZE,
    patch_size=config.PATCH_SIZE,
    embed_dim=config.EMBED_DIM,
    enc_num_heads=config.ENC_NUM_HEADS,
    enc_mlp_ratio=config.ENC_MLP_RATIO,
    enc_depth=config.ENC_DEPTH,
    dec_num_heads=config.DEC_NUM_HEADS,
    dec_mlp_ratio=config.DEC_MLP_RATIO,
    dec_depth=config.DEC_DEPTH,
    perm_num=config.PERM_NUM,
    perm_forward=config.PERM_FORWARD,
    perm_mirrored=config.PERM_MIRRORED,
    decode_ar=config.DECODE_AR,
    refine_iters=config.REFINE_ITERS,
    dropout=config.DROPOUT
)

print("Model created successfully!")

# Test forward pass
print("\nTesting forward pass...")
batch = next(iter(train_loader))
images, labels = batch
print(f"Batch images shape: {images.shape}")
print(f"Batch labels: {labels[:5]}")

# Try to create trainer
print("\nCreating trainer...")
trainer = pl.Trainer(
    max_epochs=1,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    fast_dev_run=True,
    enable_checkpointing=False,
    logger=False
)

print("Trainer created!")

# Try to fit
print("\nAttempting to fit...")
try:
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("\n✓ Training test SUCCESSFUL!")
except Exception as e:
    print(f"\n✗ Training test FAILED:")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
