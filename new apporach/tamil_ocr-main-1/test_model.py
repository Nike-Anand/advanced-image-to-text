#!/usr/bin/env python3
import torch
from torchvision import transforms
from PIL import Image
import glob
from pathlib import Path
from train_custom import TamilCNN
from character_mapping import TAMIL_CHARACTER_MAPPING

def load_latest_model():
    """Load the latest saved model"""
    model_files = glob.glob("saved_models/tamil_ocr_best_*.pth")
    if not model_files:
        model_files = glob.glob("saved_models/tamil_ocr_final_*.pth")
    
    if not model_files:
        print("❌ No saved models found!")
        return None, None
    
    # Get the latest model
    latest_model = max(model_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"Loading model: {latest_model}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TamilCNN(num_classes=157).to(device)
    
    checkpoint = torch.load(latest_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    accuracy = checkpoint.get('accuracy', 'Unknown')
    print(f"Model accuracy: {accuracy}%")
    
    return model, device

def predict_image(image_path):
    """Predict Tamil character from image"""
    model, device = load_latest_model()
    if model is None:
        return
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Original image size: {image.size}")
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get character mapping
        if predicted_class in TAMIL_CHARACTER_MAPPING:
            predicted_char = TAMIL_CHARACTER_MAPPING[predicted_class]
        else:
            predicted_char = f"Unknown (Class {predicted_class})"
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Class ID: {predicted_class}")
        print(f"   Character: {predicted_char}")
        print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Show top 3 predictions
        top3_probs, top3_classes = torch.topk(probabilities[0], 3)
        print(f"\n📊 Top 3 Predictions:")
        for i, (prob, cls) in enumerate(zip(top3_probs, top3_classes)):
            char = TAMIL_CHARACTER_MAPPING.get(cls.item(), f"Class {cls.item()}")
            print(f"   {i+1}. {char} - {prob.item()*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")

def test_sample_images():
    """Test on sample images from test_images folder"""
    test_dir = Path("test_images")
    if not test_dir.exists():
        print("❌ test_images folder not found")
        return
    
    # Get some test images
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No test images found")
        return
    
    print(f"Found {len(image_files)} test images")
    
    # Test first few images
    for img_path in image_files[:3]:
        print(f"\n{'='*50}")
        print(f"Testing: {img_path.name}")
        print('='*50)
        predict_image(str(img_path))

if __name__ == "__main__":
    print("Tamil OCR Model Tester")
    print("=" * 30)
    
    # # Test with sample images
    # test_sample_images()
    
    # Or test specific image
    predict_image("image.png")