from ocr_tamil.ocr import OCR
import cv2
import numpy as np

def segment_document_lines(image_path):
    """Document-style line segmentation (not scene text)"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocessing for document text
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Horizontal projection to find text lines
    horizontal_projection = np.sum(binary, axis=1)
    
    # Find line boundaries
    lines = []
    in_line = False
    start = 0
    
    for i, val in enumerate(horizontal_projection):
        if val > 0 and not in_line:
            start = i
            in_line = True
        elif val == 0 and in_line:
            if i - start > 5:  # Minimum line height
                lines.append((start, i))
            in_line = False
    
    # If no lines found, treat whole image as one line
    if not lines:
        lines = [(0, img.shape[0])]
    
    # Extract line images
    line_images = []
    for start, end in lines:
        line_img = img[start:end, :]
        line_images.append(line_img)
    
    return line_images

def detect_script_simple(line_image):
    """Simple script detection for line images"""
    # Basic heuristic: Tamil has more complex shapes
    contours, _ = cv2.findContours(255 - line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "UNKNOWN"
    
    # Calculate complexity metrics
    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
    
    if total_area > 0:
        complexity = total_perimeter / total_area
        # Tamil characters tend to be more complex
        return "TA" if complexity > 0.5 else "EN"
    
    return "UNKNOWN"

def document_multilingual_ocr(image_path):
    """Document-focused multilingual OCR (bypasses CRAFT)"""
    
    print(f"Processing document: {image_path}")
    
    # Initialize OCR engines
    tamil_ocr = OCR(detect=False, lang=["tamil"])  # No detection, direct recognition
    
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Image not found"}
    
    # Step 1: Segment into lines (document style)
    print("Step 1: Document line segmentation...")
    line_images = segment_document_lines(image_path)
    print(f"Found {len(line_images)} lines")
    
    results = {
        "image_path": image_path,
        "lines": [],
        "full_text": "",
        "stats": {"tamil": 0, "english": 0, "unknown": 0}
    }
    
    # Step 2 & 3: Process each line
    for i, line_img in enumerate(line_images):
        print(f"Processing line {i+1}/{len(line_images)}")
        
        # Save line temporarily
        line_path = f"temp_line_{i}.jpg"
        cv2.imwrite(line_path, line_img)
        
        # Detect script
        script = detect_script_simple(line_img)
        print(f"  Detected script: {script}")
        
        # Route to OCR (Tamil OCR handles both for now)
        try:
            if script in ["TA", "EN", "UNKNOWN"]:
                result = tamil_ocr.predict(line_path)
                text = result[0] if result and result[0] else ""
            else:
                text = "[UNSURE]"
        except Exception as e:
            text = f"[ERROR: {str(e)[:20]}]"
        
        print(f"  Extracted: {text[:30]}...")
        
        # Store results
        line_result = {
            "line_number": i + 1,
            "script": script,
            "text": text,
            "method": "document_line_ocr"
        }
        results["lines"].append(line_result)
        
        # Update stats
        results["stats"][script.lower()] = results["stats"].get(script.lower(), 0) + 1
        
        # Add to full text
        if text and not text.startswith("["):
            results["full_text"] += text + " "
    
    results["full_text"] = results["full_text"].strip()
    return results

def demo_document_ocr():
    """Demo document-focused multilingual OCR"""
    print("=== Document Multilingual OCR Demo ===")
    print("(Using line segmentation, not scene text detection)")
    
    test_images = [
        "test_images/0.jpg",
        "test_images/1.jpg", 
        "test_images/2.jpg"
    ]
    
    for image_path in test_images:
        print(f"\n{'='*50}")
        
        result = document_multilingual_ocr(image_path)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
        
        print(f"Document: {result['image_path']}")
        print(f"Lines processed: {len(result['lines'])}")
        print(f"Scripts detected: {result['stats']}")
        
        print("\nLine-by-line results:")
        for line in result["lines"]:
            script = line['script']
            text = line['text'][:50] + "..." if len(line['text']) > 50 else line['text']
            print(f"  Line {line['line_number']} [{script}]: {text}")
        
        print(f"\nFull extracted text:")
        print(f"'{result['full_text']}'")
    
    print(f"\n{'='*50}")
    print("Demo complete - Document OCR pipeline working!")

if __name__ == "__main__":
    demo_document_ocr()