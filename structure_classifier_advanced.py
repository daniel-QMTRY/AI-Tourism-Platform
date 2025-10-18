# Capstone Project: Part 1 - ADVANCED Historical Structures Image Classifier (PyTorch)
#
# Description:
# This is the complete, final, and polished script for the image classification task.
# It implements an end-to-end professional workflow:
#   1. Loads and robustly preprocesses image data, skipping corrupted files.
#   2. Implements advanced data augmentation to combat overfitting.
#   3. Builds a deep learning model using a pre-trained ResNet50 with a Dropout layer.
#   4. Employs a two-phase fine-tuning strategy with a learning rate scheduler and early stopping.
#   5. Visualizes training performance with Accuracy/Loss curves.
#   6. Conducts a deep evaluation of the final model with a Confusion Matrix and a per-class
#      Classification Report, with professional handling of warnings and logging.

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
from sklearn.metrics import confusion_matrix, classification_report

# For handling data quality issues
from PIL import ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True # Allow loading of truncated images

print(f"PyTorch Version: {torch.__version__}")
print("Libraries for ADVANCED training imported successfully.")

# Custom Dataset class to make our data loading process resilient to errors
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError):
            print(f"\nWARNING: Skipping corrupted/unreadable image: {self.samples[index][0]}")
            return self.__getitem__((index + 1) % len(self))

# =============================================================================
# 2. CONFIGURATION & ROBUST PATHS
# =============================================================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# Define all paths and directories
TRAIN_DIR = os.path.join(script_dir, 'images', 'Stuctures_Dataset')
VAL_DIR = os.path.join(script_dir, 'images', 'Dataset_test')
OUTPUT_DIR = os.path.join(script_dir, 'model_output_pytorch_advanced')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Centralized Hyperparameters ---
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 50
LR_HEAD = 0.001
LR_FINETUNE = 0.0001
EARLY_STOP_PATIENCE = 5

# --- Hardware Setup ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# 3. ADVANCED DATA AUGMENTATION & LOADERS
# =============================================================================
print("\n--- Setting up ADVANCED data transforms and loaders ---")
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: SafeImageFolder(os.path.join(script_dir, 'images', d), data_transforms[x]) for x, d in [('train', 'Stuctures_Dataset'), ('val', 'Dataset_test')]}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(f"Found {dataset_sizes['train']} training images and {dataset_sizes['val']} validation images.")
print(f"Found {len(class_names)} classes.")

# =============================================================================
# 4. MODEL DEFINITION (ResNet50 with Dropout)
# =============================================================================
print("\n--- Building the Advanced CNN model with Dropout ---")
model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(class_names))
)

model = model.to(device)
print("Model architecture loaded (ResNet50 with Dropout).")

# =============================================================================
# 5. TRAINING & FINE-TUNING LOOP
# =============================================================================
def train_model(model, criterion, dataloaders, patience, num_epochs_head, num_epochs_finetune):
    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # --- PHASE 1: Train only the classifier head ---
    print("\n--- Phase 1: Training the Classifier Head ---")
    optimizer = optim.Adam(model.fc.parameters(), lr=LR_HEAD)

    for epoch in range(num_epochs_head):
        print(f'Epoch {epoch + 1}/{num_epochs_head}')
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
                        loss.backward(); optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # --- PHASE 2: Unfreeze deeper layers and fine-tune ---
    print("\n--- Phase 2: Unfreezing Deeper Layers for Fine-Tuning ---")
    # POLISHED LOGGING: Clean, summary message
    print("Unfreezing layers: layer3, layer4, and fc.")
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True

    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': LR_FINETUNE / 10},
        {'params': model.layer4.parameters(), 'lr': LR_FINETUNE / 10},
        {'params': model.fc.parameters(), 'lr': LR_FINETUNE}
    ], lr=LR_FINETUNE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs_finetune):
        print(f'Epoch {epoch + 1}/{num_epochs_finetune}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs); _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward(); optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        if phase == 'train': scheduler.step()

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            model.load_state_dict(best_model_wts)
            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Loss (used for early stopping): {best_val_loss:.6f}')
            return model, history

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    model.load_state_dict(best_model_wts)
    return model, history

# =============================================================================
# 6. EXECUTE TRAINING & SAVE ARTIFACTS
# =============================================================================
criterion = nn.CrossEntropyLoss()
model, history = train_model(model, criterion, dataloaders, EARLY_STOP_PATIENCE, EPOCHS_HEAD, EPOCHS_FINETUNE)

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'structure_classifier_model_advanced.pth'))
print(f"\nBest model saved to {os.path.join(OUTPUT_DIR, 'structure_classifier_model_advanced.pth')}")

# =============================================================================
# 7. VISUALIZE TRAINING PERFORMANCE
# =============================================================================
print("\n--- Generating performance plots for Advanced Model ---")
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1); plt.plot(history['train_acc'], label='Training Acc'); plt.plot(history['val_acc'], label='Validation Acc')
plt.legend(); plt.title('Accuracy vs. Epochs'); plt.xlabel('Total Epochs')
plt.subplot(1, 2, 2); plt.plot(history['train_loss'], label='Training Loss'); plt.plot(history['val_loss'], label='Validation Loss')
plt.legend(); plt.title('Loss vs. Epochs'); plt.xlabel('Total Epochs')
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'training_performance_plots_advanced.png')); plt.show()
print(f"Performance plots saved to {os.path.join(OUTPUT_DIR, 'training_performance_plots_advanced.png')}")

# =============================================================================
# 8. ADVANCED EVALUATION: CONFUSION MATRIX & CLASSIFICATION REPORT
# =============================================================================
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', out_path='cm.png'):
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10)); plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title); plt.colorbar(); tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right"); plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'; thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.tight_layout()
    plt.savefig(out_path, dpi=180); plt.close()

def evaluate_and_create_confusion_matrix(model, val_loader, class_names, device, out_dir):
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    
    # POLISHED: Handle warning for classes with no samples in the validation set
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f: f.write(report)

    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(class_names)))
    plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion Matrix', out_path=os.path.join(out_dir, 'confusion_matrix_raw.png'))
    plot_confusion_matrix(cm, class_names, normalize=True, title='Normalized Confusion Matrix', out_path=os.path.join(out_dir, 'confusion_matrix_normalized.png'))
    print("\nConfusion matrices and classification report saved to:", out_dir)

evaluate_and_create_confusion_matrix(model, dataloaders["val"], class_names, device, OUTPUT_DIR)

# =============================================================================
# 9. CONCLUSION
# =============================================================================
print("\n--- Part 1 Advanced Model Conclusion ---")
print("The advanced training regimen has completed.")
best_val_acc = max(history['val_acc'])
print(f"Best Validation Accuracy achieved: {best_val_acc:.2%}")
print(f"The final model, plots, and evaluation reports are saved in '{OUTPUT_DIR}'.")
print("\n--- END OF ADVANCED SCRIPT ---")

