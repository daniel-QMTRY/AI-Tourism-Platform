# Capstone Project: Part 1 - ADVANCED Historical Structures Image Classifier (PyTorch)
# Description: This script implements an advanced training regimen to improve upon the baseline model.
# Techniques include: Advanced Data Augmentation, Dropout, LR Scheduling, Two-Phase Fine-Tuning, and Early Stopping.

# =============================================================================
# 1. SETUP & IMPORT LIBRARIES
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import copy

# For handling data quality issues
from PIL import ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

print(f"PyTorch Version: {torch.__version__}")
print("Libraries for ADVANCED training imported successfully.")

# Custom Dataset class to skip corrupted images
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError) as e:
            print(f"\nWARNING: Skipping corrupted/unreadable image: {self.samples[index][0]}")
            return self.__getitem__((index + 1) % len(self))

# =============================================================================
# 2. DEFINE PATHS AND PARAMETERS
# =============================================================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

TRAIN_DIR = os.path.join(script_dir, 'images', 'Stuctures_Dataset')
VAL_DIR = os.path.join(script_dir, 'images', 'Dataset_test')

output_dir = os.path.join(script_dir, 'model_output_pytorch_advanced')
os.makedirs(output_dir, exist_ok=True)

# Model parameters
IMG_SIZE = 224 # ResNet models work well with 224x224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50 # Set higher for early stopping
PATIENCE = 5 # For early stopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# 3. ADVANCED DATA AUGMENTATION
# =============================================================================
print("\n--- Setting up ADVANCED data transforms and loaders ---")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        # --- NEW: Advanced Augmentation ---
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = SafeImageFolder(TRAIN_DIR, data_transforms['train'])
val_dataset = SafeImageFolder(VAL_DIR, data_transforms['val'])

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes.")

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# =============================================================================
# 4. BUILD THE ADVANCED CNN MODEL WITH DROPOUT
# =============================================================================
print("\n--- Building the Advanced CNN model with Dropout ---")

model = models.resnet50(weights='IMAGENET1K_V1') # Using a deeper model

# Freeze all base layers
for param in model.parameters():
    param.requires_grad = False

# --- NEW: Create a new classifier head with a Dropout layer ---
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5), # Dropout layer to reduce overfitting
    nn.Linear(512, num_classes)
)

model = model.to(device)
print("Model architecture loaded (ResNet50 with Dropout).")

criterion = nn.CrossEntropyLoss()

# We will define the optimizer later for fine-tuning

# =============================================================================
# 5. ADVANCED TRAINING LOOP WITH FINE-TUNING AND EARLY STOPPING
# =============================================================================
def train_model_advanced(model, criterion, num_epochs=50, patience=5):
    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # --- PHASE 1: Train only the classifier head ---
    print("\n--- Phase 1: Training the Classifier Head ---")
    optimizer_head = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler_head = lr_scheduler.StepLR(optimizer_head, step_size=7, gamma=0.1)
    
    # --- NEW: Early Stopping Initialization ---
    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initial training loop (you can set a fixed number of epochs, e.g., 5-10)
    for epoch in range(10): # Train head for 10 epochs
        print(f'Epoch {epoch+1}/10')
        print('-' * 10)
        # (Training and validation loop for the head is part of the main loop below)
    
    # --- PHASE 2: Unfreeze layers and fine-tune ---
    print("\n--- Phase 2: Unfreezing Deeper Layers for Fine-Tuning ---")
    for name, child in model.named_children():
        if name in ['layer3', 'layer4']:
            print(f"Unfreezing {name}")
            for param in child.parameters():
                param.requires_grad = True
    
    # Create a new optimizer for fine-tuning with different learning rates
    params_to_update = [
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE},
        {'params': model.layer4.parameters(), 'lr': LEARNING_RATE / 10},
        {'params': model.layer3.parameters(), 'lr': LEARNING_RATE / 100}
    ]
    optimizer_finetune = optim.Adam(params_to_update, lr=LEARNING_RATE)
    scheduler_finetune = lr_scheduler.StepLR(optimizer_finetune, step_size=7, gamma=0.1)

    # Use the fine-tuning optimizer for the rest of the epochs
    optimizer = optimizer_finetune
    scheduler = scheduler_finetune

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            # --- NEW: Early Stopping Check ---
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break

        if phase == 'train':
            scheduler.step()

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss (used for early stopping): {best_val_loss:4f}')
    
    model.load_state_dict(best_model_wts)
    return model, history

model, history = train_model_advanced(model, criterion, num_epochs=EPOCHS, patience=PATIENCE)
model_path = os.path.join(output_dir, 'structure_classifier_model_advanced.pth')
torch.save(model.state_dict(), model_path)
print(f"Best model saved to {model_path}")

# =============================================================================
# 6. VISUALIZE MODEL PERFORMANCE
# =============================================================================
print("\n--- Generating performance plots for Advanced Model ---")
plt.figure(figsize=(14, 6))
# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right'), plt.title('Training and Validation Accuracy'), plt.xlabel('Epoch'), plt.ylabel('Accuracy')
# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right'), plt.title('Training and Validation Loss'), plt.xlabel('Epoch'), plt.ylabel('Loss')

plt.tight_layout()
plot_path = os.path.join(output_dir, 'training_performance_plots_advanced.png')
plt.savefig(plot_path)
print(f"Performance plots saved to {plot_path}")
plt.show()

# =============================================================================
# 7. CONCLUSION
# =============================================================================
print("\n--- Part 1 Advanced Model Conclusion ---")
print("The advanced training regimen has completed.")
# Find best accuracy from the validation history
best_val_acc = max(history['val_acc'])
print(f"Best Validation Accuracy achieved: {best_val_acc:.2%}")
print("The final model and performance plots are saved in 'model_output_pytorch_advanced'.")
print("\n--- END OF ADVANCED SCRIPT ---")
