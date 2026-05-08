#!/usr/bin/env python3
import torch
import json
import os
from datetime import datetime
from pathlib import Path
from character_mapping import TAMIL_CHARACTER_MAPPING

class ModelSaver:
    def __init__(self, models_dir="saved_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, model_name, accuracy=None, epoch=None, optimizer=None, **kwargs):
        """Save model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"{model_name}_{timestamp}.pth"
        
        # Prepare save data
        save_data = {
            'model_state_dict': model.state_dict(),
            'character_mapping': TAMIL_CHARACTER_MAPPING,
            'model_name': model_name,
            'timestamp': timestamp,
            'model_architecture': model.__class__.__name__,
        }
        
        # Add optional data
        if accuracy is not None:
            save_data['accuracy'] = accuracy
        if epoch is not None:
            save_data['epoch'] = epoch
        if optimizer is not None:
            save_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add any additional kwargs
        save_data.update(kwargs)
        
        # Save model
        torch.save(save_data, model_path)
        
        # Save metadata as JSON
        metadata = {k: v for k, v in save_data.items() 
                   if k not in ['model_state_dict', 'optimizer_state_dict']}
        
        json_path = model_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Metadata saved: {json_path}")
        
        return model_path
    
    def load_model(self, model_class, model_path):
        """Load model from path"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model instance
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint
    
    def list_models(self):
        """List all saved models"""
        models = list(self.models_dir.glob("*.pth"))
        if not models:
            print("No saved models found")
            return []
        
        print(f"Found {len(models)} saved models:")
        for model_path in sorted(models):
            json_path = model_path.with_suffix('.json')
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                acc = metadata.get('accuracy', 'N/A')
                epoch = metadata.get('epoch', 'N/A')
                print(f"  {model_path.name} - Acc: {acc}% - Epoch: {epoch}")
            else:
                print(f"  {model_path.name}")
        
        return models

def save_new_model(model, name="tamil_ocr_model", **kwargs):
    """Quick save function"""
    saver = ModelSaver()
    return saver.save_model(model, name, **kwargs)

if __name__ == "__main__":
    # Example usage
    saver = ModelSaver()
    saver.list_models()