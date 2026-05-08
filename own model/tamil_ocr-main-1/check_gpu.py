"""
Quick script to check GPU availability and configuration
"""

import torch

print("=" * 80)
print("GPU AVAILABILITY CHECK")
print("=" * 80)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")
    
    # Test GPU with a simple operation
    print("\nTesting GPU with simple operation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU test successful!")
        
        # Memory info
        print(f"\nCurrent GPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("\n" + "!" * 80)
    print("WARNING: No GPU detected!")
    print("Training will be very slow on CPU.")
    print("\nTo install CUDA-enabled PyTorch, visit:")
    print("https://pytorch.org/get-started/locally/")
    print("!" * 80)

print("\n" + "=" * 80)
