"""
Training Script for Tamil Handwritten Character Recognition
Fine-tunes the PARSeq model on Tamil handwritten character dataset
"""

import os
import sys
import argparse
from pathlib import Path

# Set UTF-8 encoding for Windows console (MUST be before other imports that print)
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tamil_char_dataset import create_dataloaders
from char_mapping import get_charset_string
import config_handwritten as config

# Import the PARSeq model
from ocr_tamil.strhub.models.parseq.system import PARSeq


def create_model(charset_train, charset_test=None, pretrained_path=None):
    """
    Create PARSeq model for Tamil character recognition.
    
    Args:
        charset_train: String of characters for training
        charset_test: String of characters for testing (defaults to charset_train)
        pretrained_path: Path to pretrained checkpoint (optional)
    
    Returns:
        PARSeq model instance
    """
    if charset_test is None:
        charset_test = charset_train
    
    model = PARSeq(
        charset_train=charset_train,
        charset_test=charset_test,
        max_label_length=config.MAX_LABEL_LENGTH,
        batch_size=config.BATCH_SIZE,
        lr=config.LEARNING_RATE,
        warmup_pct=config.WARMUP_PCT,
        weight_decay=config.WEIGHT_DECAY,
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
    
    # Add configure_optimizers method if not present (for PyTorch Lightning)
    if not hasattr(model, 'configure_optimizers'):
        def configure_optimizers(self):
            """Configure optimizer and learning rate scheduler."""
            import math
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import OneCycleLR
            
            # Get parameters to optimize
            params = [p for p in self.parameters() if p.requires_grad]
            
            # Create optimizer
            optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
            
            # Create learning rate scheduler
            # OneCycleLR for warmup and decay
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.warmup_pct,
                anneal_strategy='cos',
                cycle_momentum=False
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                }
            }
        
        # Bind the method to the model instance
        import types
        model.configure_optimizers = types.MethodType(configure_optimizers, model)
    
    # Load pretrained weights if specified
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Freeze encoder if specified
    if config.FREEZE_ENCODER:
        print("Freezing encoder weights")
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    return model


def setup_callbacks(checkpoint_dir):
    """Setup training callbacks."""
    callbacks = []
    
    # Model checkpoint callback - save best models
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='tamil-char-{epoch:02d}-{val_accuracy:.2f}',
        monitor='val_accuracy',
        mode='max',
        save_top_k=config.SAVE_TOP_K,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Periodic checkpoint callback
    periodic_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='tamil-char-epoch-{epoch:02d}',
        every_n_epochs=config.SAVE_EVERY_N_EPOCHS,
        save_top_k=-1  # Save all periodic checkpoints
    )
    callbacks.append(periodic_checkpoint)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=config.PATIENCE,
        mode='max',
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_gpu():
    """Setup and configure GPU for training."""
    if not torch.cuda.is_available():
        print("\n" + "!" * 80)
        print("WARNING: CUDA is not available! Training will be VERY slow on CPU.")
        print("Please install CUDA-enabled PyTorch for GPU acceleration.")
        print("!" * 80 + "\n")
        return False
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    
    print("\n" + "=" * 80)
    print("GPU CONFIGURATION")
    print("=" * 80)
    print(f"CUDA Available: Yes")
    print(f"GPU Count: {gpu_count}")
    print(f"GPU Name: {gpu_name}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Set memory growth (similar to TensorFlow's set_memory_growth)
    # PyTorch handles this automatically, but we can set some optimizations
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.backends.cudnn.enabled = True
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Get memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total GPU Memory: {total_memory:.2f} GB")
    print("=" * 80 + "\n")
    
    return True


def train(args):
    """Main training function."""
    print("=" * 80)
    print("Tamil Handwritten Character Recognition Training")
    print("=" * 80)
    
    # Setup GPU - Force GPU usage
    gpu_available = setup_gpu()
    if not gpu_available:
        response = input("\nNo GPU detected. Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled. Please setup CUDA and try again.")
            return None
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Get charset
    charset = get_charset_string()
    print(f"\nCharset size: {len(charset)}")
    print(f"Sample characters: {charset[:20]}...")
    
    # Create dataloaders
    print(f"\nLoading data from:")
    print(f"  Train: {config.TRAIN_DIR}")
    print(f"  Test: {config.TEST_DIR}")
    
    subset_size = config.SUBSET_SIZE if not args.full_dataset else None
    
    train_loader, val_loader = create_dataloaders(
        train_dir=config.TRAIN_DIR,
        test_dir=config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_size=config.IMG_SIZE,
        subset_size=subset_size
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Create model
    print(f"\nCreating model...")
    model = create_model(
        charset_train=charset,
        pretrained_path=args.pretrained if args.pretrained else config.PRETRAINED_MODEL_PATH
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup callbacks
    callbacks = setup_callbacks(config.CHECKPOINT_DIR)
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config.LOG_DIR,
        name='tamil_char_recognition'
    )
    
    # Create trainer - Force GPU usage
    if not torch.cuda.is_available() and not (args.fast_dev_run or config.FAST_DEV_RUN):
        raise RuntimeError(
            "GPU is required for training! No CUDA device found.\n"
            "Please install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/"
        )
    
    # Determine accelerator and precision
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    precision = '16-mixed' if config.USE_FP16 and torch.cuda.is_available() else 32
    
    if accelerator == 'cpu':
        print("\n" + "!" * 80)
        print("WARNING: Training on CPU - This will be extremely slow!")
        print("!" * 80 + "\n")
    
    trainer = pl.Trainer(
        max_epochs=args.epochs if args.epochs else config.NUM_EPOCHS,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        accumulate_grad_batches=config.ACCUMULATE_GRAD_BATCHES,
        precision=precision,
        val_check_interval=config.VAL_CHECK_INTERVAL,
        limit_val_batches=config.LIMIT_VAL_BATCHES,
        fast_dev_run=args.fast_dev_run or config.FAST_DEV_RUN,
        log_every_n_steps=10,
        deterministic=False,  # Set to True for reproducibility (slower)
        benchmark=True  # Enable cuDNN benchmarking for faster training
    )
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {trainer.max_epochs}")
    print(f"  Device: {accelerator.upper()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  Precision: {precision}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Gradient clipping: {config.GRADIENT_CLIP_VAL}")
    print("=" * 80)
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Test on best model
    print("\n" + "=" * 80)
    print("Training completed! Testing best model...")
    print("=" * 80)
    
    trainer.test(model, val_loader)
    
    print(f"\nBest model saved to: {trainer.checkpoint_callback.best_model_path}")
    print(f"Logs saved to: {config.LOG_DIR}")
    
    return trainer.checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser(description='Train Tamil Handwritten Character Recognition')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model checkpoint')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run 1 batch for quick testing')
    parser.add_argument('--full_dataset', action='store_true', help='Use full dataset (ignore SUBSET_SIZE)')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    # Train
    best_model_path = train(args)
    
    print("\n" + "=" * 80)
    print("Training pipeline completed successfully!")
    print("=" * 80)
    print(f"\nTo view training logs, run:")
    print(f"  tensorboard --logdir {config.LOG_DIR}")
    print(f"\nBest model: {best_model_path}")


if __name__ == "__main__":
    main()
