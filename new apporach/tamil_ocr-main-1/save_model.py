#!/usr/bin/env python3
"""
Quick script to save a new Tamil OCR model
"""
import torch
from model_saver import ModelSaver
from train_custom import TamilCNN

def save_current_model():
    """Save the current best model with new name"""
    # Load existing model if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Create model
    model = TamilCNN(num_classes=157)
    
    # Try to load existing weights
    try:
        checkpoint = torch.load('best_tamil_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        accuracy = checkpoint.get('best_acc', 0)
        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded existing model with {accuracy:.2f}% accuracy")
    except FileNotFoundError:
        print("No existing model found, saving new initialized model")
        accuracy = 0
        epoch = 0
    
    # Save with new name
    saver = ModelSaver()
    model_path = saver.save_model(
        model=model,
        model_name="tamil_ocr_v1",
        accuracy=accuracy,
        epoch=epoch,
        description="Tamil OCR model for character recognition",
        version="1.0"
    )
    
    print(f"Model saved successfully at: {model_path}")
    return model_path

def list_saved_models():
    """List all saved models"""
    saver = ModelSaver()
    saver.list_models()

if __name__ == "__main__":
    print("Tamil OCR Model Saver")
    print("=" * 30)
    
    # List existing models
    print("Current saved models:")
    list_saved_models()
    
    print("\nSaving new model...")
    save_current_model()
    
    print("\nUpdated model list:")
    list_saved_models()