# Capstone Project: Part 1 - Historical Structures Image Classification (PyTorch)

# =============================================================================
# 1. SETUP & IMPORT LIBRARIES
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# --- ROBUSTNESS FIX ---
# This line tells the image loader to be tolerant of truncated/corrupted images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

print(f"PyTorch Version: {torch.__version__}")
print("Libraries imported successfully.")

# =============================================================================
# 2. DEFINE PATHS AND PARAMETERS (CORRECTED)
# =============================================================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# --- CORRECTED PATHS BASED ON YOUR FOLDER STRUCTURE ---
TRAIN_DIR = os.path.join(script_dir, 'images', 'Stuctures_Dataset')
VAL_DIR = os.path.join(script_dir, 'images', 'Dataset_test')

# Check if directories exist
if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    print("\nFATAL ERROR: Could not find the image directories.")
    print(f"Looked for Training Data in: {TRAIN_DIR}")
    print(f"Looked for Validation Data in: {VAL_DIR}")
    print("Please ensure your 'images' folder with 'Stuctures_Dataset' and 'Dataset_test' is in the project directory.")
    exit()

# Create a directory to save plots and the model
output_dir = os.path.join(script_dir, 'model_output_pytorch')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Model parameters
IMG_SIZE = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 25 # A good number for a pre-trained model

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# 3. DATA PREPROCESSING & AUGMENTATION
# =============================================================================
print("\n--- Setting up data transforms and loaders ---")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMG_SIZE + 20),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the datasets using ImageFolder from the correct directories
print(f"Loading training data from: {TRAIN_DIR}")
train_dataset = datasets.ImageFolder(TRAIN_DIR, data_transforms['train'])

print(f"Loading validation data from: {VAL_DIR}")
val_dataset = datasets.ImageFolder(VAL_DIR, data_transforms['val'])

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")
print(f"Found {num_classes} classes: {class_names}")

# Create DataLoaders
# NOTE: num_workers is set to 0. This is often necessary on Windows.
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}


# =============================================================================
# 4. BUILD THE CONVOLUTIONAL NEURAL NETWORK (CNN) MODEL
# =============================================================================
print("\n--- Building the CNN model ---")

model = models.resnet18(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device)

print("Model architecture loaded (ResNet18 with a new final layer).")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# =============================================================================
# 5. TRAIN THE MODEL
# =============================================================================
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_wts = model.state_dict()
    best_acc = 0.0

    print("\n--- Starting model training with real data ---")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
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

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

model, history = train_model(model, criterion, optimizer, num_epochs=EPOCHS)

model_path = os.path.join(output_dir, 'structure_classifier_model_pytorch.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# =============================================================================
# 6. VISUALIZE MODEL PERFORMANCE
# =============================================================================
print("\n--- Generating performance plots ---")

acc = history['train_acc']
val_acc = history['val_acc']
loss = history['train_loss']
val_loss = history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plot_path = os.path.join(output_dir, 'training_performance_plots_pytorch.png')
plt.savefig(plot_path)
print(f"Performance plots saved to {plot_path}")
plt.show()

# =============================================================================
# 7. CONCLUSION
# =============================================================================
print("\n--- Part 1 Conclusion ---")
print(f"The PyTorch model has been trained on the real image data.")
print(f"Best Validation Accuracy: {max(val_acc):.2%}")
print("The final model and performance plots are saved in 'model_output_pytorch'.")
print("\n--- END OF SCRIPT (PART 1) ---")

