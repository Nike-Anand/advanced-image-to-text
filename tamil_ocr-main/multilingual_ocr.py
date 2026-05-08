import cv2
import numpy as np
from line_segmentation import segment_lines
from script_classifier import ScriptClassifier
from ocr_router import OCRRouter
import os

class MultilingualOCR:
    def __init__(self):
        self.script_classifier = ScriptClassifier()
        self.script_classifier.train_simple_model()
        self.ocr_router = OCRRouter()
        print("Multilingual OCR system initialized")
    
    def process_image(self, image_path):
        """Main pipeline: segment -> classify -> OCR"""
        if not os.path.exists(image_path):
            return {"error": "Image not found"}
        
        results = {
            "image_path": image_path,
            "lines": [],
            "full_text": "",
            "stats": {"tamil": 0, "english": 0, "unknown": 0}
        }
        
        try:
            # Step 1: Segment lines
            print("Step 1: Segmenting lines...")
            line_images = segment_lines(image_path)
            print(f"Found {len(line_images)} lines")
            
            if not line_images:
                # If no lines found, process whole image
                img = cv2.imread(image_path)
                line_images = [img]
                print("No lines detected, processing whole image")
            
            # Step 2 & 3: Classify script and route OCR
            for i, line_img in enumerate(line_images):
                print(f"Processing line {i+1}/{len(line_images)}")
                
                # Classify script
                script = self.script_classifier.predict_script(line_img)
                print(f"  Detected script: {script}")
                
                # Route to appropriate OCR
                text = self.ocr_router.route_ocr(line_img, script)
                print(f"  Extracted text: {text[:50]}...")
                
                # Store results
                line_result = {
                    "line_number": i + 1,
                    "script": script,
                    "text": text,
                    "confidence": "high" if script != "UNKNOWN" else "low"
                }
                results["lines"].append(line_result)
                
                # Update stats
                if script == "TA":
                    results["stats"]["tamil"] += 1
                elif script == "EN":
                    results["stats"]["english"] += 1
                else:
                    results["stats"]["unknown"] += 1
                
                # Add to full text
                if text and text not in ["[UNSURE]", "[TAMIL_ERROR]", "[ENGLISH_ERROR]"]:
                    results["full_text"] += text + " "
            
            results["full_text"] = results["full_text"].strip()
            
        except Exception as e:
            results["error"] = str(e)
            print(f"Error processing image: {e}")
        
        return results
    
    def print_results(self, results):
        """Pretty print results"""
        print("\n" + "="*50)
        print("MULTILINGUAL OCR RESULTS")
        print("="*50)
        
        if "error" in results:
            print(f"ERROR: {results['error']}")
            return
        
        print(f"Image: {results['image_path']}")
        print(f"Lines processed: {len(results['lines'])}")
        print(f"Stats: Tamil={results['stats']['tamil']}, English={results['stats']['english']}, Unknown={results['stats']['unknown']}")
        
        print("\nLine-by-line results:")
        for line in results["lines"]:
            print(f"  Line {line['line_number']} [{line['script']}]: {line['text']}")
        
        print(f"\nFull text:\n{results['full_text']}")
        print("="*50)

def multilingual_ocr(image_path):
    """Simple function interface"""
    ocr_system = MultilingualOCR()
    return ocr_system.process_image(image_path)

# Test the complete system
if __name__ == "__main__":
    print("Testing Multilingual OCR System")
    
    # Test with sample image
    image_path = "test_images/0.jpg"
    
    ocr_system = MultilingualOCR()
    results = ocr_system.process_image(image_path)
    ocr_system.print_results(results)
    
    print("\nSystem ready for production use!")