import sys
import os
import cv2
import numpy as np
from time import time

# -------------------------------
# UTF-8 fix for Windows terminal
# -------------------------------
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8")

from ocr_tamil.ocr import OCR

# -------------------------------
# Image Preprocessing Function
# -------------------------------
def preprocess_image(image_path, save_debug=True):
    """
    OCR-safe preprocessing for handwritten Tamil:
    - grayscale
    - contrast enhancement
    - light denoising
    - NO binarization
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found")

    # 1️⃣ Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Contrast enhancement (safe)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3️⃣ Light denoising (preserves strokes)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    if save_debug:
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/preprocessed.png", enhanced)

    return enhanced

# -------------------------------
# OCR Pipeline
# -------------------------------
ocr = OCR(
    detect=True,
    details=2,
    text_threshold=0.35,   # stricter = less garbage
    fp16=False
)

image_path = r"image.png"

# Preprocess image
processed_img = preprocess_image(image_path)

start = time()

# OCR accepts numpy image directly
text_list = ocr.predict(processed_img)

end = time()

print("Single text detect recognize")
print(text_list)
print("time taken:", end - start)


# -------------------------------
# Proper Line Grouping + Save
# -------------------------------
with open(r"outputs\english_tamil.txt", "w", encoding="utf-8") as f:
    for item in text_list:
        current_line = None
        for info in item:
            text, conf, bbox_info = info
            box, line_id = bbox_info   # ✅ correct structure

            if line_id == current_line:
                f.write(text + " ")
            else:
                if current_line is not None:
                    f.write("\n")
                f.write(text + " ")
                current_line = line_id

        f.write("\n\n")
