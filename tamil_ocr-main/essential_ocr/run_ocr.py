"""
Tamil OCR - How to Run

QUICK START:
1. cd "c:\D\Projects\image to text\tamil_ocr-main\essential_ocr"
2. python run_ocr.py

MODELS: Downloaded and cached in ~/.model_weights/ for offline use
"""

import os

def main():
    print("=== Tamil OCR System ===")
    print("Models location: ~/.model_weights/")
    
    # Check if models exist
    model_dir = os.path.expanduser("~/.model_weights")
    tamil_model = os.path.join(model_dir, "parseq_tamil_v3.pt")
    craft_model = os.path.join(model_dir, "craft_mlt_25k.pth")
    
    if os.path.exists(tamil_model) and os.path.exists(craft_model):
        print("SUCCESS: Models found and ready for offline use")
        print(f"  Tamil model: {os.path.basename(tamil_model)} ({os.path.getsize(tamil_model)//1024//1024}MB)")
        print(f"  Detection model: {os.path.basename(craft_model)} ({os.path.getsize(craft_model)//1024//1024}MB)")
    else:
        print("ERROR: Models not found")
        return
    
    # Try to run OCR
    try:
        from ocr_tamil.ocr import OCR
        
        print("\n=== Testing OCR ===")
        ocr = OCR(detect=False, lang=["tamil"], fp16=False)
        print("SUCCESS: OCR initialized")
        
        # Test with sample image
        image_path = "test_images/0.jpg"
        if os.path.exists(image_path):
            print(f"Processing: {image_path}")
            result = ocr.predict(image_path)
            print(f"Result: {result}")
        else:
            print("No test image found")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\n=== SOLUTION ===")
        print("Version compatibility issue. To fix:")
        print("1. Install compatible Python version (3.8-3.10)")
        print("2. Or use original repository environment")
        print("3. Models are ready for offline use once fixed")

if __name__ == "__main__":
    main()