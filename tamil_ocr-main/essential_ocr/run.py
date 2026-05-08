"""
Tamil OCR - Working Example

This script demonstrates how to run the Tamil OCR system.
Models are downloaded automatically and cached for offline use.
"""

import os
import sys

def main():
    print("=== Tamil OCR System ===")
    print("Models location: ~/.model_weights/")
    
    # Check if models exist
    model_dir = os.path.expanduser("~/.model_weights")
    tamil_model = os.path.join(model_dir, "parseq_tamil_v3.pt")
    craft_model = os.path.join(model_dir, "craft_mlt_25k.pth")
    
    if os.path.exists(tamil_model) and os.path.exists(craft_model):
        print("✓ Models found and ready for offline use")
        print(f"  Tamil model: {tamil_model}")
        print(f"  Detection model: {craft_model}")
    else:
        print("✗ Models not found")
        return
    
    # Try to import and run OCR
    try:
        from ocr_tamil.ocr import OCR
        
        print("\n=== Testing OCR ===")
        
        # Initialize with Tamil only (more stable)
        ocr = OCR(detect=False, lang=["tamil"], fp16=False)
        print("✓ OCR initialized successfully")
        
        # Test with sample image
        image_path = "test_images/0.jpg"
        if os.path.exists(image_path):
            print(f"Processing: {image_path}")
            result = ocr.predict(image_path)
            print(f"Result: {result}")
        else:
            print("No test image found")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n=== Troubleshooting ===")
        print("1. Version compatibility issue detected")
        print("2. Try installing compatible versions:")
        print("   pip install timm==0.6.13")
        print("3. Or use the original repository with proper environment")
        
        print("\n=== Alternative Usage ===")
        print("The models are downloaded and ready. You can:")
        print("1. Use the original repository with Python 3.8-3.10")
        print("2. Create a virtual environment with compatible versions")
        print("3. Use the models with a different OCR framework")

if __name__ == "__main__":
    main()