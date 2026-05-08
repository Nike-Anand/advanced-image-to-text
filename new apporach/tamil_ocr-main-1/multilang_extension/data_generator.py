from PIL import Image, ImageDraw, ImageFont
import os
import random

class SyntheticTextGenerator:
    def __init__(self, font_paths, output_dir="datasets"):
        self.font_paths = font_paths
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, language, texts, img_size=(320, 96), font_size=48):
        save_path = os.path.join(self.output_dir, language)
        os.makedirs(save_path, exist_ok=True)

        font_path = self.font_paths[language]

        try:
            font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.LAYOUT_RAQM)
        except:
            font = ImageFont.truetype(font_path, font_size)

        for i, text in enumerate(texts):
            img = Image.new("RGB", img_size, "white")
            draw = ImageDraw.Draw(img)

            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            x = (img_size[0] - w) // 2
            y = (img_size[1] - h) // 2

            draw.text((x, y), text, fill="black", font=font)
            img.save(f"{save_path}/{i:05d}.png", dpi=(300, 300))

# ---------------- SAMPLE TEXTS ---------------- #

TEXTS = {
    "hindi": [
        "नमस्ते", "भारत", "स्वागत", "दिल्ली", "मुंबई",
        "विद्यालय", "छात्र", "शिक्षा", "ज्ञान", "पुस्तक"
    ],
    "telugu": [
        "నమస్కారం", "భారతదేశం", "విద్య", "తెలుగు", "పుస్తకం",
        "హైదరాబాద్", "గురు", "విద్యార్థి", "భాష", "పాఠశాల"
    ],
    "kannada": [
        "ನಮಸ್ಕಾರ", "ಭಾರತ", "ಶಿಕ್ಷಣ", "ಪುಸ್ತಕ",
        "ವಿದ್ಯಾರ್ಥಿ", "ಶಾಲೆ", "ಗುರು", "ಭಾಷೆ", "ಬೆಂಗಳೂರು"
    ]
}

# ---------------- FONT PATHS ---------------- #

FONTS = {
    "hindi": "fonts/NotoSansDevanagari-Regular.ttf",
    "telugu": "fonts/NotoSansTelugu-Regular.ttf",
    "kannada": "fonts/NotoSansKannada-Regular.ttf"
}

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    generator = SyntheticTextGenerator(FONTS)

    for lang, texts in TEXTS.items():
        generator.generate(lang, texts)
