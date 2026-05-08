from ocr_tamil.ocr import OCR

print("=== Testing Official OCR Tamil Package ===")

# Test 1: Text Recognition Only
print("\n1. Text Recognition Only:")
ocr = OCR()
result = ocr.predict("test_images/1.jpg")
print(f"Result: {result}")

# Test 2: Text Detection + Recognition  
print("\n2. Text Detection + Recognition:")
ocr = OCR(detect=True)
result = ocr.predict("test_images/0.jpg")
print(f"Result: {result}")

print("\n=== SUCCESS! Official package working ===")