from ocr_tamil.ocr import OCR
import cv2
import numpy as np

def simple_multilingual_ocr(image_path):
    """Simplified multilingual OCR without complex dependencies"""
    
    print(f"Processing: {image_path}")
    
    # Initialize Tamil OCR
    tamil_ocr = OCR(detect=True, lang=["tamil"])
    
    # Simple script detection based on character patterns
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Image not found"}
    
    # Process with Tamil OCR (which handles both Tamil and English)
    result = tamil_ocr.predict(image_path)
    
    # Parse results
    extracted_texts = []
    if result and len(result) > 0:
        for item in result[0]:
            if isinstance(item, list) and len(item) >= 2:
                text = item[0]
                confidence = item[1] if len(item) > 1 else 0.0
                if text and text.strip():
                    extracted_texts.append({
                        "text": text,
                        "confidence": confidence,
                        "script": "TA" if any(ord(c) > 2944 and ord(c) < 3071 for c in text) else "EN"
                    })
    
    return {
        "image_path": image_path,
        "texts": extracted_texts,
        "full_text": " ".join([t["text"] for t in extracted_texts])
    }

def demo_multilingual_ocr():
    """Demo the multilingual OCR system"""
    print("=== Multilingual OCR Demo ===")
    
    # Test images
    test_images = [
        "test_images/0.jpg",
        "test_images/1.jpg", 
        "test_images/2.jpg"
    ]
    
    for image_path in test_images:
        print(f"\n--- Processing {image_path} ---")
        
        result = simple_multilingual_ocr(image_path)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
        
        print(f"Found {len(result['texts'])} text regions:")
        for i, text_info in enumerate(result['texts']):
            script = text_info['script']
            text = text_info['text']
            conf = text_info['confidence']
            print(f"  {i+1}. [{script}] {text} (conf: {conf:.2f})")
        
        print(f"Full text: {result['full_text']}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demo_multilingual_ocr()