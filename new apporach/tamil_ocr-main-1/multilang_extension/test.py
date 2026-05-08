from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype("fonts/NotoSansTelugu-Regular.ttf", 48)
img = Image.new("RGB", (400, 100), "white")
draw = ImageDraw.Draw(img)
draw.text((10, 20), "నమస్కారం", font=font, fill="black")
img.show()
