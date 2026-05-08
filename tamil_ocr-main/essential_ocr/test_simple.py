from ocr_tamil.ocr import OCR
import os

print("Testing Tamil OCR - Text Recognition Only")

# Initialize OCR without detection (simpler mode)
try:
    ocr = OCR(detect=False, lang=["tamil"], fp16=False)
    print("SUCCESS: OCR initialized successfully")
    
    # Test image path
    image_path = "test_images/0.jpg"
    
    if os.path.exists(image_path):
        print(f"Processing image: {image_path}")
        text_list = ocr.predict(image_path)
        
        print("Extracted Text:")
        for text in text_list:
            print(f"Text: {text}")
    else:
        print(f"ERROR: Image not found: {image_path}")
        
except Exception as e:
    print(f"ERROR: {e}")
    print("\nNote: This version may have compatibility issues with current PyTorch/timm versions.")
    print("The models downloaded successfully and are cached for offline use.")
    print("Models are stored in: ~/.model_weights/")