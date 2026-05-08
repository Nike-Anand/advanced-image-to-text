#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("Installing PyTorch with CUDA...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install PyTorch with CUDA")
        print("Trying CPU version...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
            return True
        except subprocess.CalledProcessError:
            return False

def check_gpu():
    """Check GPU availability after PyTorch is installed"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ ERROR: CUDA GPU not available")
            return False
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_data():
    """Check if training data exists"""
    required_paths = [
        Path("archive/train"),
        Path("archive/test"),
        Path("archive/test.csv")
    ]
    
    for path in required_paths:
        if not path.exists():
            print(f"❌ Missing: {path}")
            return False
    
    print("Training data found")
    return True

def install_requirements():
    """Install other required packages"""
    requirements = ["pillow", "pandas", "tqdm"]
    
    print("Installing other requirements...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {req}")
            return False
    
    print("Requirements installed")
    return True

def main():
    print("Tamil OCR Training Pipeline")
    print("=" * 40)
    
    # Install PyTorch first
    if not install_pytorch():
        print("❌ Failed to install PyTorch")
        sys.exit(1)
    
    # Check GPU after PyTorch installation
    if not check_gpu():
        print("❌ GPU check failed - stopping training")
        sys.exit(1)
    
    # Install other requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check data
    if not check_data():
        print("Please ensure training data is in 'archive/' directory")
        sys.exit(1)
    
    # Start training
    print("\nStarting training...")
    try:
        from train_custom import train_model
        model = train_model()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()