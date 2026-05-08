import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

from ocr_tamil.ocr import OCR
from time import time

ocr = OCR(detect=True,details=2,text_threshold=0.3,fp16=False)
# For single image - text detect + text recognize
image_path = r"image.png" # insert your own path here

s = time()
text_list = ocr.predict(image_path)
e = time()

print("Single text detect recognize",text_list)

print("time taken",e-s)

with open(r"outputs\english_tamil.txt","w",encoding="utf-8") as f:
    for item in text_list:
        current_line = 1
        for info in item:
            text,conf,bbox = info
            line = bbox[1]
            if line == current_line:
                f.write(text + " ")
            else:
                f.write("\n"+text+ " ")
                current_line = line

        f.write("\n")