#!/usr/bin/env python3
from ocr_tamil.ocr import OCR

def read_tamil_text(image_path):
    """Use the existing Tamil OCR library"""
    print(f"Processing: {image_path}")
    
    # Initialize OCR with text detection
    ocr = OCR(detect=True)
    
    # Process the image
    results = ocr.predict(image_path)
    
    # Print results
    print(f"\n📝 Detected Text:")
    full_text = " ".join(results[0]) if results and results[0] else "No text detected"
    print(full_text)
    
    return full_text

if __name__ == "__main__":
    image_path = r"C:\D\Projects\image to text\new apporach\tamil_ocr-main-1\image.png"
    read_tamil_text(image_path)