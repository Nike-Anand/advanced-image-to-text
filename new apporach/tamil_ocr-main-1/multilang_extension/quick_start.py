"""
Quick Start Guide for Multi-Language Tamil OCR Training
GPU-FORCED TRAINING SETUP
"""

import torch
import os

def check_gpu():
    """Check GPU availability and requirements"""
    print("🔍 Checking GPU Requirements...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        print("\n📋 Requirements:")
        print("1. NVIDIA GPU (GTX 1060+ or RTX series recommended)")
        print("2. CUDA 11.0+ installed")
        print("3. PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 4:
        print("⚠️  Warning: Less than 4GB GPU memory. Training may be slow.")
    
    return True

def setup_directories():
    """Create required directories"""
    print("\n📁 Setting up directories...")
    
    dirs = [
        "models",
        "datasets/multilang_train/hindi",
        "datasets/multilang_train/telugu", 
        "datasets/multilang_train/kannada",
        "datasets/multilang_val/hindi",
        "datasets/multilang_val/telugu",
        "datasets/multilang_val/kannada",
        "fonts"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ Directories created")

def main():
    """Main setup and training guide"""
    print("🚀 Multi-Language Tamil OCR - GPU Training Setup")
    print("=" * 60)
    
    # Step 1: Check GPU
    if not check_gpu():
        return
    
    # Step 2: Setup directories
    setup_directories()
    
    # Step 3: Training steps
    print("\n🎯 TRAINING STEPS:")
    print("=" * 30)
    
    print("\n1️⃣  Generate Training Data:")
    print("   python data_generator.py")
    
    print("\n2️⃣  Start GPU Training:")
    print("   python train_multilang.py")
    
    print("\n3️⃣  Test Multi-Language OCR:")
    print("   python usage_examples.py")
    
    print("\n⚡ GPU Training Configuration:")
    print("   - Batch Size: 16 (adjust based on GPU memory)")
    print("   - Precision: 16-bit (faster training)")
    print("   - Epochs: 300")
    print("   - Languages: Tamil + English + Hindi")
    
    print("\n💡 Memory Requirements:")
    print("   - 4GB+ GPU: Basic training")
    print("   - 8GB+ GPU: Recommended")
    print("   - 12GB+ GPU: All languages")
    
    print("\n🔧 Troubleshooting:")
    print("   - Out of memory: Reduce batch_size in train_multilang.py")
    print("   - Slow training: Check GPU utilization with nvidia-smi")
    print("   - CUDA errors: Reinstall PyTorch with CUDA support")

if __name__ == "__main__":
    main()