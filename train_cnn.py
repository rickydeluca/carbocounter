import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm

from models.cnn import CNN
from datasets.food101_dataset import Food101Dataset, Food101Subset
from train_utils.images import imshow

# Set the default device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Useful paths
data_dir        = "data/food-101/images/"
best_model_path = "best_models/cnn.pth"

# Data augmentation
train_transform =   transforms.Compose([
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

test_transform =    transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])


# Load data without applying any transform
dataset         = Food101Dataset(data_dir)

# Split in train, val and test
train_size  = int(0.7 * len(dataset))
val_size    = int(0.15 * len(dataset))
test_size   = len(dataset) - train_size - val_size

train_indices, val_indices, test_indices = random_split(range(len(dataset)), [train_size, val_size, test_size])

train_dataset   = Food101Subset(Food101Dataset(data_dir), train_indices, transform=train_transform)
val_dataset     = Food101Subset(Food101Dataset(data_dir), val_indices, transform=test_transform)
test_dataset    = Food101Subset(Food101Dataset(data_dir), test_indices, transform=test_transform)

# Create Data Loaders
batch_size      = 256 if device.type == 'cuda' else 32
train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader      = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader     = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


# Define model
model = CNN(n_classes=101).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and validate (using early stopping)
num_epochs = 100
patience = 5            # Patience for early stopping
best_val_loss = np.inf  # Initialize best validation loss
stop_counter = 0        # Counter for early stopping

# ==========================
#   TRAININIG & VALIDATION
# ==========================

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print training informations
    print(f"Epoch {epoch+1}/{num_epochs}, "
        f"Train Loss: {running_loss/len(train_loader)}, "
        f"Val Loss: {val_loss/len(val_loader)}, "
        f"Val Acc: {100 * correct / total}%")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        stop_counter = 0                                    # Reset counter
        torch.save(model.state_dict(), best_model_path)     # Save best model
    else:
        stop_counter += 1
        if stop_counter >= patience:
            print("Early stopping triggered.")
            break


# ========
#   TEST 
# ========

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Acc: {100 * correct / total}%")