from ocr_tamil.ocr import OCR
import os

print("=== Official OCR Tamil Package Test ===")

# Test with recognition only
print("\n1. Text Recognition:")
ocr = OCR()
result = ocr.predict("tamil_ocr-main/test_images/1.jpg")
print(f"Result: {result}")

# Test with detection + recognition
print("\n2. Detection + Recognition:")
ocr = OCR(detect=True)
result = ocr.predict("tamil_ocr-main/test_images/0.jpg")
print(f"Result: {result}")

print("\n=== SUCCESS! Official package working ===")