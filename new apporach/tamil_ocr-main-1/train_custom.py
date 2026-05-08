import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
from character_mapping import TAMIL_CHARACTER_MAPPING
from model_saver import ModelSaver

# GPU Check - Stop if no GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA GPU not available. Training requires GPU. Stopping.")
    exit(1)

print(f"GPU Available: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class TamilCharDataset(Dataset):
    def __init__(self, data_dir, csv_file=None, transform=None, is_train=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        
        print(f"Loading dataset from {data_dir}...")
        
        self.samples = []
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir() and class_dir.name.isdigit():
                class_id = int(class_dir.name)
                if class_id < 157:  # Only valid Tamil classes
                    for img_file in class_dir.glob('*.bmp'):
                        if img_file.exists():
                            self.samples.append((str(img_file), class_id))
        
        print(f"Loaded {len(self.samples)} valid samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            # Skip to next valid sample
            return self.__getitem__((idx + 1) % len(self.samples))

class TamilCNN(nn.Module):
    def __init__(self, num_classes=157):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
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

def train_model():
    device = torch.device('cuda')
    print(f"Training on device: {device}")
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = TamilCharDataset('archive/train', transform=train_transform)
    val_dataset = TamilCharDataset('archive/test', transform=val_transform)
    
    # Check if we have enough data
    if len(train_dataset) < 100:
        print(f"❌ Not enough training data: {len(train_dataset)} samples")
        return None
    
    if len(val_dataset) < 10:
        print(f"❌ Not enough validation data: {len(val_dataset)} samples")
        print("Checking validation data structure...")
        test_csv = Path('archive/test.csv')
        if test_csv.exists():
            df = pd.read_csv(test_csv)
            print(f"CSV has {len(df)} rows")
            valid_files = 0
            for _, row in df.iterrows():
                if row['Class Label'] < 157:
                    img_path = Path('archive/test') / row['ID']
                    if img_path.exists():
                        valid_files += 1
            print(f"Found {valid_files} valid image files")
        return None
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = TamilCNN(num_classes=157).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training
    best_acc = 0
    epochs = 20
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for data, target in pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100*train_correct/train_total:.2f}%'})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            # Save with ModelSaver
            saver = ModelSaver()
            saver.save_model(
                model=model,
                model_name="tamil_ocr_best",
                accuracy=best_acc,
                epoch=epoch,
                optimizer=optimizer,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                train_acc=train_acc
            )
            print(f'New best model saved! Accuracy: {best_acc:.2f}%')
    
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    
    # Save final model
    saver = ModelSaver()
    final_path = saver.save_model(
        model=model,
        model_name="tamil_ocr_final",
        accuracy=val_acc,
        epoch=epochs,
        best_accuracy=best_acc,
        status="training_completed"
    )
    
    return model

if __name__ == "__main__":
    train_model()