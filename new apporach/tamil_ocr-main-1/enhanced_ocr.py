import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

class CustomTamilModel(nn.Module):
    def __init__(self, num_classes=157):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.features(x)

class EnhancedOCR:
    def __init__(self, custom_model_path='custom_tamil_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load custom model
        self.custom_model = CustomTamilModel()
        if torch.cuda.is_available():
            self.custom_model.load_state_dict(torch.load(custom_model_path))
        else:
            self.custom_model.load_state_dict(torch.load(custom_model_path, map_location='cpu'))
        self.custom_model.to(self.device).eval()
        
        # Transform for custom model
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Character mapping (simplified - you'd need the full mapping)
        self.id_to_char = {
            0: 'ஃ', 1: 'அ', 2: 'ஆ', 3: 'இ', 4: 'ஈ', 5: 'உ',
            # Add all 157 character mappings here
        }
    
    def predict_character(self, image):
        """Predict single character from image"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.custom_model(input_tensor)
            pred_id = torch.argmax(output, dim=1).item()
        
        return self.id_to_char.get(pred_id, '?')
    
    def enhance_existing_ocr(self, original_ocr_result, character_images):
        """Enhance existing OCR results with custom model predictions"""
        enhanced_results = []
        
        for i, char_img in enumerate(character_images):
            custom_pred = self.predict_character(char_img)
            
            # You can implement logic to combine original and custom predictions
            # For now, just use custom prediction
            enhanced_results.append(custom_pred)
        
        return ''.join(enhanced_results)

# Integration with existing OCR
def integrate_with_existing_ocr():
    """Example of how to integrate with the existing OCR system"""
    from ocr_tamil.ocr import OCR
    
    # Initialize both models
    original_ocr = OCR(detect=True)
    enhanced_ocr = EnhancedOCR()
    
    def enhanced_predict(image_path):
        # Get original OCR results
        original_results = original_ocr.predict(image_path)
        
        # For character-level enhancement, you'd need to:
        # 1. Extract individual character regions from the image
        # 2. Pass each character to the custom model
        # 3. Combine results
        
        # Simplified example - just return original for now
        return original_results
    
    return enhanced_predict

if __name__ == "__main__":
    # Test the custom model
    enhanced_ocr = EnhancedOCR()
    
    # Test with a sample image (if available)
    # result = enhanced_ocr.predict_character('path/to/character/image.bmp')
    # print(f"Predicted character: {result}")
    
    print("Enhanced OCR system ready!")