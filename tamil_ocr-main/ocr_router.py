from ocr_tamil.ocr import OCR
import easyocr
import cv2
import numpy as np

class OCRRouter:
    def __init__(self):
        # Initialize OCR engines
        self.tamil_ocr = OCR(detect=False, lang=["tamil"])
        self.english_ocr = None  # Will initialize on first use
        print("OCR Router initialized")
    
    def get_english_ocr(self):
        """Lazy initialization of English OCR"""
        if self.english_ocr is None:
            try:
                self.english_ocr = easyocr.Reader(['en'])
                print("English OCR initialized")
            except:
                print("Warning: English OCR not available")
        return self.english_ocr
    
    def process_tamil(self, image):
        """Process Tamil text"""
        try:
            # Save image temporarily for Tamil OCR
            cv2.imwrite("temp_tamil.jpg", image)
            result = self.tamil_ocr.predict("temp_tamil.jpg")
            return result[0] if result and result[0] else ""
        except Exception as e:
            print(f"Tamil OCR error: {e}")
            return "[TAMIL_ERROR]"
    
    def process_english(self, image):
        """Process English text"""
        try:
            ocr = self.get_english_ocr()
            if ocr is None:
                return "[EN_OCR_UNAVAILABLE]"
            
            result = ocr.readtext(image)
            text = " ".join([item[1] for item in result if item[2] > 0.5])
            return text if text else ""
        except Exception as e:
            print(f"English OCR error: {e}")
            return "[ENGLISH_ERROR]"
    
    def route_ocr(self, image, script):
        """Route to appropriate OCR based on script"""
        if script == "TA":
            return self.process_tamil(image)
        elif script == "EN":
            return self.process_english(image)
        else:
            return "[UNSURE]"

# Test routing
if __name__ == "__main__":
    router = OCRRouter()
    
    # Test with sample image
    img = cv2.imread("test_images/0.jpg")
    if img is not None:
        # Test Tamil routing
        result = router.route_ocr(img, "TA")
        print(f"Tamil result: {result}")
        
        # Test English routing  
        result = router.route_ocr(img, "EN")
        print(f"English result: {result}")
        
        # Test unknown
        result = router.route_ocr(img, "UNKNOWN")
        print(f"Unknown result: {result}")
    else:
        print("Test image not found")