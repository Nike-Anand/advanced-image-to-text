from ocr_tamil.ocr import OCR

image_path = r"C:\D\Projects\image to text\image.png"

# Text recognition
ocr = OCR()
result = ocr.predict(image_path)
print("Text Recognition Result:", result)

# With detection
ocr = OCR(detect=True)
result = ocr.predict(image_path)
print("Detection Result:", result)
