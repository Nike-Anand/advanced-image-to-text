import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from character_mapping import TAMIL_CHARACTER_MAPPING, get_character

class TamilCNN(nn.Module):
    def __init__(self, num_classes=157):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def check_validation_data():
    """Check validation dataset"""
    print("Checking validation dataset...")
    
    test_csv = Path("archive/test.csv")
    test_dir = Path("archive/test")
    
    if not test_csv.exists():
        print("test.csv not found")
        return False
    
    if not test_dir.exists():
        print("test directory not found")
        return False
    
    df = pd.read_csv(test_csv)
    print(f"CSV contains {len(df)} rows")
    
    valid_count = 0
    missing_count = 0
    
    for _, row in df.iterrows():
        if row['Class Label'] < 157:
            img_path = test_dir / row['ID']
            if img_path.exists():
                valid_count += 1
            else:
                missing_count += 1
                if missing_count <= 5:  # Show first 5 missing files
                    print(f"Missing: {img_path}")
    
    print(f"Valid images: {valid_count}")
    print(f"Missing images: {missing_count}")
    
    # Check what files actually exist in test directory
    if test_dir.exists():
        actual_files = list(test_dir.glob('*.bmp'))
        print(f"Actual .bmp files in test dir: {len(actual_files)}")
        if len(actual_files) > 0:
            print(f"Sample files: {[f.name for f in actual_files[:5]]}")
    
    return valid_count > 0

def validate_model():
    """Validate trained model"""
    model_path = Path("best_tamil_model.pth")
    
    if not model_path.exists():
        print("❌ No trained model found. Run training first.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = TamilCNN(num_classes=157).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded. Best accuracy: {checkpoint['best_acc']:.2f}%")
    
    # Test transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Test on a few validation images
    test_dir = Path("archive/test")
    test_csv = Path("archive/test.csv")
    
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        tested = 0
        correct = 0
        
        for _, row in df.iterrows():
            if tested >= 10:  # Test first 10 valid images
                break
                
            if row['Class Label'] < 157:
                img_path = test_dir / row['ID']
                if img_path.exists():
                    try:
                        image = Image.open(img_path).convert('RGB')
                        image_tensor = transform(image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output = model(image_tensor)
                            predicted = torch.argmax(output, dim=1).item()
                        
                        actual = row['Class Label']
                        pred_char = get_character(predicted)
                        actual_char = get_character(actual)
                        
                        is_correct = predicted == actual
                        if is_correct:
                            correct += 1
                        
                        print(f"Image: {row['ID']} | Predicted: {pred_char} ({predicted}) | Actual: {actual_char} ({actual}) | {'✅' if is_correct else '❌'}")
                        tested += 1
                        
                    except Exception as e:
                        print(f"Error testing {img_path}: {e}")
        
        if tested > 0:
            accuracy = (correct / tested) * 100
            print(f"\n📊 Sample Test Results: {correct}/{tested} correct ({accuracy:.1f}%)")
        else:
            print("❌ No valid test images found")

if __name__ == "__main__":
    print("Tamil OCR Model Validation")
    print("=" * 40)
    
    # Check validation data
    if check_validation_data():
        print("\nTesting trained model...")
        validate_model()
    else:
        print("Cannot validate model - no valid test data")