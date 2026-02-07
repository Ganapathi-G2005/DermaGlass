import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import os
import json
import time
import sys
import numpy as np

# --- Configuration for RTX 3050 (4GB VRAM) ---
BATCH_SIZE = 16 
NUM_WORKERS = 2 
WARMUP_EPOCHS = 5      # Stage 1: Train Head Only
FINE_TUNE_EPOCHS = 20  # Stage 2: Train Full Model (Unfrozen)
TOTAL_EPOCHS = WARMUP_EPOCHS + FINE_TUNE_EPOCHS

DATA_DIR = "Dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CLASS_NAMES_FILE = "class_names.json"
MODEL_SAVE_PATH = "best_model.pth"

# Enable optimizations
torch.backends.cudnn.benchmark = True

def main():
    print(f"python version: {sys.version}")
    print(f"torch version: {torch.__version__}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # --- 1. Enhanced Data Augmentation ---
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Keep more of the object
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # Skin patches can be rotated any way
        transforms.RandomRotation(30),   # Increased rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- 2. Data Loaders & Imbalance Handling ---
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("Error: Dataset folders not found.")
        return

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    
    classes = train_dataset.classes
    num_classes = len(classes)
    
    with open(CLASS_NAMES_FILE, 'w') as f:
        json.dump(classes, f)
    
    # Weighted Sampler
    class_counts = [0] * num_classes
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    class_weights = [1.0 / c if c > 0 else 0 for c in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- 3. Model Setup ---
    print("Loading EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Initial Freeze
    for param in model.features.parameters():
        param.requires_grad = False
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4), # Increased Dropout
        nn.Linear(in_features, num_classes)
    )
    
    model = model.to(device)

    # --- 4. Training Setup ---
    # Label Smoothing to prevent overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer for Stage 1 (Head Only)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # Scheduler: Reduce LR if val_acc plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_acc = 0.0
    
    # --- Helper Function for One Epoch ---
    def run_epoch(epoch_idx, is_training=True):
        if is_training:
            model.train()
        else:
            model.eval()
            
        running_loss = 0.0
        running_corrects = 0
        
        loader = train_loader if is_training else val_loader
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if is_training:
                optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                if not is_training:
                    with torch.no_grad():
                         outputs = model(inputs)
                         loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            
            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects.double() / len(loader.dataset)
        
        return epoch_loss, epoch_acc

    # --- 5. Stage 1: Warmup (Frozen Features) ---
    print("\n" + "="*40)
    print(f"STAGE 1: Warming up Classifier Head ({WARMUP_EPOCHS} Epochs)")
    print("="*40)
    
    for epoch in range(WARMUP_EPOCHS):
        start = time.time()
        train_loss, train_acc = run_epoch(epoch)
        val_loss, val_acc = run_epoch(epoch, is_training=False)
        dur = time.time() - start
        
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | Time: {dur:.0f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Saved Best Model: {best_val_acc:.4f}")

    # --- 6. Stage 2: Fine Tuning (Unfrozen) ---
    print("\n" + "="*40)
    print(f"STAGE 2: Fine Tuning Full Model ({FINE_TUNE_EPOCHS} Epochs)")
    print("="*40)
    
    # Unfreeze specific blocks (Last 2 blocks of EfficientNet-B0)
    # Features[6] and Features[7] are the deeper layers
    for param in model.features.parameters():
        param.requires_grad = True # Unfreeze all for simplicity with low LR, or target specific layers
    
    # Re-create optimizer with much lower learning rate for stability
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-5}, # Very low LR for backbone
        {'params': model.classifier.parameters(), 'lr': 1e-4}  # Low LR for head
    ], lr=1e-4) # Base LR
    
    # Update scheduler to watch this new optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    for epoch in range(WARMUP_EPOCHS, TOTAL_EPOCHS):
        start = time.time()
        train_loss, train_acc = run_epoch(epoch)
        val_loss, val_acc = run_epoch(epoch, is_training=False)
        dur = time.time() - start
        
        # Step the scheduler
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | Time: {dur:.0f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Saved Best Model: {best_val_acc:.4f}")

    print(f"\nTraining Complete. Final Best Accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
