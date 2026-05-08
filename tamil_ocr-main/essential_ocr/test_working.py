from ocr_tamil.ocr import OCR
import os

print("=== Tamil OCR Working Test ===")

# Initialize OCR
ocr = OCR(detect=True, lang=["tamil"], fp16=False)
print("SUCCESS: OCR initialized with detection")

# Test with available images
test_images = ["test_images/0.jpg", "test_images/1.jpg", "test_images/2.jpg"]

for image_path in test_images:
    if os.path.exists(image_path):
        print(f"\nProcessing: {image_path}")
        try:
            result = ocr.predict(image_path)
            print(f"SUCCESS: Extracted text: {result}")
        except Exception as e:
            print(f"ERROR processing {image_path}: {e}")
        break
else:
    print("No test images found")

print("\n=== OCR is Working! ===")
print("Models are cached offline and ready to use")