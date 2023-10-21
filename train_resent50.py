import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from train_utils.early_stopping import train_model

# Define transformations for data augmentation
data_transforms = {

    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 
data_dir = "path_to_Food101_dataset"


def create_data_loaders_from_directory(data_dir, train_split=0.8):
    # Create train and validation directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    for class_name in os.listdir(os.path.join(data_dir, 'images')):
        # Create corresponding directories in train and val folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # List all the images of this class
        images = os.listdir(os.path.join(data_dir, 'images', class_name))
        
        # Split the images into training and validation sets
        num_images = len(images)
        num_train = int(train_split * num_images)
        
        train_images = images[:num_train]
        val_images = images[num_train:]
        
        for image in train_images:
            source = os.path.join(data_dir, 'images', class_name, image)
            target = os.path.join(train_dir, class_name, image)
            shutil.move(source, target)
            
        for image in val_images:
            source = os.path.join(data_dir, 'images', class_name, image)
            target = os.path.join(val_dir, class_name, image)
            shutil.move(source, target)
    
    # Data augmentations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(32),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create ImageFolder and DataLoader
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=64, shuffle=False)
    }
    
    return dataloaders

data_dir = "data/food-101"
dataloaders = create_data_loaders_from_directory(data_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get ResNet 50 model and modify it wrt our dataset
model = models.resnet50(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 101)
model = model.to(device)

# Define loss function, optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train model with early stopping
model = train_model(model=model,
                    dataloaders=dataloaders,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=exp_lr_scheduler,
                    num_epochs=25,
                    patience=7,
                    outfile="best_models/resnet50.pth.tar")


