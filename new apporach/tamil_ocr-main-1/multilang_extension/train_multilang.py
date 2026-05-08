import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import sys
sys.path.append('..')

from ocr_tamil.strhub.models.parseq.system import PARSeq
from ocr_tamil.strhub.data.utils import Tokenizer
from charset_utils import get_charset_for_languages, CHARSET_MULTILANG

class MultiLangTrainer:
    def __init__(self, languages=["tamil", "english", "hindi"], base_model_path=None):
        self.languages = languages
        self.charset = get_charset_for_languages(languages)
        self.tokenizer = Tokenizer(self.charset)
        self.base_model_path = base_model_path
        
    def load_base_model(self):
        """Load existing Tamil-English model as starting point"""
        if self.base_model_path:
            model = torch.load(self.base_model_path, weights_only=False)
        else:
            # Load from checkpoint
            from ocr_tamil.strhub.models.utils import load_from_checkpoint
            model = load_from_checkpoint("pretrained=parseq")
        
        return model
    
    def extend_model_charset(self, model):
        """Create new model with extended charset"""
        print(f"Creating new model with {len(self.charset)} characters")
        
        # Create new model with extended charset
        from ocr_tamil.strhub.models.utils import load_from_checkpoint
        new_model = load_from_checkpoint("pretrained=parseq")
        
        # Replace tokenizer
        new_model.tokenizer = self.tokenizer
        
        # Rebuild head layer with correct size
        if hasattr(new_model, 'head'):
            old_head = new_model.head
            new_vocab_size = len(self.tokenizer)
            new_head = torch.nn.Linear(old_head.in_features, new_vocab_size)
            
            # Initialize with Xavier uniform
            torch.nn.init.xavier_uniform_(new_head.weight)
            torch.nn.init.zeros_(new_head.bias)
            
            new_model.head = new_head
        
        print(f"New model created with vocab size: {len(self.tokenizer)}")
        return new_model
    
    def create_training_config(self):
        """Create training configuration"""
        config = {
            'model': {
                'charset': self.charset,
                'max_label_length': 25
            },
            'data': {
                'train_root': 'datasets/multilang_train',
                'val_root': 'datasets/multilang_val'
            },
            'trainer': {
                'max_epochs': 300,
                'precision': 16,
                'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu'
            }
        }
        return config
    
    def train_model(self, train_dataloader, val_dataloader, output_path="models/multilang_parseq.pt"):
        """Train the multi-language model with forced GPU usage"""
        # Force GPU usage
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for training. Please ensure GPU is available.")
        
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Create new model with extended charset
        model = self.extend_model_charset(None)
        model = model.to(device)
        
        # Setup trainer with forced GPU
        trainer = pl.Trainer(
            max_epochs=50,  # Reduced for faster training
            precision=16,
            accelerator='gpu',
            devices=1,
            enable_progress_bar=True,
            log_every_n_steps=5
        )
        
        # Train model
        print(f"Training model for languages: {self.languages}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        trainer.fit(model, train_dataloader, val_dataloader)
        
        # Save trained model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(model, output_path)
        print(f"Model saved to {output_path}")
        
        return model

def create_dataloader(data_dir, batch_size=8):
    """Create dataloader from dataset directory"""
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import glob
    from torchvision import transforms as T
    
    # Image transforms for PARSEQ
    transform = T.Compose([
        T.Resize([32, 128], T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])
    
    class TextDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.image_paths = glob.glob(f"{data_dir}/**/*.jpg", recursive=True)
            self.transform = transform
            
        def __len__(self):
            return len(self.image_paths)
            
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            
            # Get label from txt file
            label_path = img_path.replace('.jpg', '.txt')
            with open(label_path, 'r', encoding='utf-8') as f:
                label = f.read().strip()
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
    
    dataset = TextDataset(data_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def main():
    """Main training function with GPU enforcement"""
    print("=== Multi-Language PARSEQ Training ===")
    print("Step 1: Checking GPU availability...")
    
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA GPU not available!")
        print("Please ensure:")
        print("- NVIDIA GPU is installed")
        print("- CUDA drivers are installed")
        print("- PyTorch with CUDA support is installed")
        return
    
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\nStep 2: Initializing trainer...")
    languages = ["tamil", "english", "hindi"]
    trainer = MultiLangTrainer(languages, base_model_path="../ocr_tamil/model_weights/parseq_tamil_v3.pt")
    print(f"✅ Charset size: {len(trainer.charset)} characters")
    print(f"✅ Languages: {languages}")
    
    print("\nStep 3: Creating dataloaders...")
    try:
        train_dataloader = create_dataloader("datasets/multilang_train")
        val_dataloader = create_dataloader("datasets/multilang_val")
        print("✅ Dataloaders created")
    except Exception as e:
        print(f"❌ Error creating dataloaders: {e}")
        print("Please run: python data_generator.py first")
        return
    
    print("\nStep 4: Starting training...")
    model = trainer.train_model(train_dataloader, val_dataloader)
    print("✅ Training completed!")

if __name__ == "__main__":
    main()
    
# Quick start commands:
# 1. python data_generator.py
# 2. python train_multilang.py