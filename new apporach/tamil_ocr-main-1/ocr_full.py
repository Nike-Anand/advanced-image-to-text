#!/usr/bin/env python3
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
from pathlib import Path
from train_custom import TamilCNN
from character_mapping import TAMIL_CHARACTER_MAPPING

def load_model():
    """Load the trained model"""
    model_files = glob.glob("saved_models/tamil_ocr_best_*.pth")
    if not model_files:
        model_files = glob.glob("saved_models/tamil_ocr_final_*.pth")
    
    if not model_files:
        print("❌ No saved models found!")
        return None, None
    
    latest_model = max(model_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"Loading model: {Path(latest_model).name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TamilCNN(num_classes=157).to(device)
    
    checkpoint = torch.load(latest_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def detect_text_regions(image_path):
    """Simple text detection using contours"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    char_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < w < 200 and 15 < h < 200:  # Filter by reasonable character size
            char_boxes.append((x, y, w, h))
    
    # Sort boxes left to right, top to bottom
    char_boxes.sort(key=lambda box: (box[1] // 30, box[0]))  # Group by rows, then sort by x
    
    return char_boxes, img

def recognize_character(model, device, img, box):
    """Recognize single character from bounding box"""
    x, y, w, h = box
    char_img = img[y:y+h, x:x+w]
    
    # Convert to PIL and preprocess
    char_pil = Image.fromarray(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(char_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    char = TAMIL_CHARACTER_MAPPING.get(predicted_class, "?")
    return char, confidence

def ocr_paragraph(image_path):
    """Full OCR: detect + recognize text"""
    model, device = load_model()
    if model is None:
        return
    
    print(f"\n🔍 Processing: {Path(image_path).name}")
    
    # Detect text regions
    char_boxes, img = detect_text_regions(image_path)
    print(f"Found {len(char_boxes)} potential characters")
    
    if len(char_boxes) == 0:
        print("❌ No text detected")
        return
    
    # Recognize each character
    recognized_text = []
    current_line_y = char_boxes[0][1] if char_boxes else 0
    line_text = []
    
    for i, box in enumerate(char_boxes):
        x, y, w, h = box
        
        # Check if we're on a new line
        if abs(y - current_line_y) > 20:  # New line threshold
            if line_text:
                recognized_text.append("".join(line_text))
                line_text = []
            current_line_y = y
        
        char, confidence = recognize_character(model, device, img, box)
        
        if confidence > 0.3:  # Only include confident predictions
            line_text.append(char)
            print(f"  {char} ({confidence:.2f})")
        else:
            print(f"  ? ({confidence:.2f}) - low confidence")
    
    # Add the last line
    if line_text:
        recognized_text.append("".join(line_text))
    
    # Print results
    print(f"\n📝 Recognized Text:")
    for i, line in enumerate(recognized_text, 1):
        print(f"   Line {i}: {line}")
    
    print(f"\n📄 Full Text:")
    print(" ".join(recognized_text))
    
    return recognized_text

if __name__ == "__main__":
    print("Tamil OCR - Full Text Recognition")
    print("=" * 40)
    
    # Process the specific image
    image_path = r"C:\D\Projects\image to text\new apporach\tamil_ocr-main-1\image.png"
    ocr_paragraph(image_path)