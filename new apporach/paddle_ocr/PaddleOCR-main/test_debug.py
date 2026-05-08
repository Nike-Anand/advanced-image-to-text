import sys
import traceback

try:
    from paddleocr import PaddleOCR
    print("PaddleOCR imported successfully!")
    
    ocr = PaddleOCR(use_textline_orientation=True, lang='ta')
    print("PaddleOCR initialized successfully!")
    
except Exception as e:
    print(f"Error occurred: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
