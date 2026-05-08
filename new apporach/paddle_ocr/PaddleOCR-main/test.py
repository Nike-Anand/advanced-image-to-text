from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True,lang='ta')
result = ocr.predict('test.jpeg')

# Extract just the text
for line in result[0]['rec_texts']:
    print(line)
