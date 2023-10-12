import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Modify these varibles to control the code execution
DEBUG   = False
TRAIN   = True
TEST    = True

# Set the default device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Useful paths
data_dir = "data/food-101/images/"
best_model_dir = "best_models/inception_v3.pth"

# Image preprocessing
train_transform =   transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

test_transform =    transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

# Load data without applying any transform
data = datasets.ImageFolder(root=data_dir)

# Split in train, val and test
train_size = int(0.7 * len(data))
val_test_size = len(data) - train_size
val_size = int(0.15 * len(data))
test_size = val_test_size - val_size

train_data, val_test_data = random_split(data, [train_size, val_test_size])
val_data, test_data = random_split(val_test_data, [val_size, test_size])

# Apply the train_transform only to train_data
train_data = Subset(train_data.dataset, train_data.indices)
train_data.dataset.transform = train_transform

# Apply the test_transform to val_data and test_data
val_data = Subset(val_data.dataset, val_data.indices)
val_data.dataset.transform = test_transform

test_data = Subset(test_data.dataset, test_data.indices)
test_data.dataset.transform = test_transform

# Create Data Loaders
batch_size = 16 if device.type == 'cuda' else 8
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

# Define model architecture
model = models.inception_v3(weights='IMAGENET1K_V1')
model.fc = nn.Sequential(
    nn.BatchNorm1d(2048, eps=0.001, momentum=0.01),
    nn.Dropout(0.2),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, len(data.classes)),
    nn.Softmax(dim=1)
)
model = model.to(device)    # Move the model to the selected device

if TRAIN:

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and validate (using early stopping)
    num_epochs = 1 if DEBUG else 50
    patience = 5            # Patience for early stopping
    best_val_loss = np.inf  # Initialize best validation loss
    stop_counter = 0        # Counter for early stopping

    for epoch in range(num_epochs):
        # Training loop:
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            if DEBUG:
                print(f"labels: {labels}")
                break
            optimizer.zero_grad()
            outputs, _ = model(inputs)          # Do not consider auxiliary output
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation loop:
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", unit="batch"):
                if DEBUG:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if DEBUG:
            break

        # Print training informations
        print(f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {running_loss/len(train_loader)}, "
            f"Val Loss: {val_loss/len(val_loader)}, "
            f"Val Acc: {100 * correct / total}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0                                # Reset counter
            torch.save(model.state_dict(), best_model_dir)  # Save best model
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print("Early stopping triggered.")
                break


# Load best model
if not DEBUG:   
    model.load_state_dict(torch.load(best_model_dir))


# Test:
if TEST:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            if DEBUG:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if not DEBUG:
        print(f"Test Acc: {100 * correct / total}%")


# ======================
#   SAMPLE PREDICTIONS
# ======================
def predict_image(image_path, model, class_names):
    """
    Predict the class of an image and display the image with the predicted class as title.

    This function loads an image from the specified path, applies the necessary transformations,
    and passes it through the provided model to predict its class. The image along with the 
    predicted class is then displayed using matplotlib.

    Args:
        image_path (str): Path to the image file to be predicted.
        model (torch.nn.Module): The PyTorch model to be used for prediction.
        class_names (list of str): List of class names, used to map the prediction output 
                                   to a human-readable class name.

    Returns:
        None: The function displays the image and does not return any value.
    """
    # Load image using PIL
    img_pil = Image.open(image_path)
    
    # Apply the same transformations as for the training images
    transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    
    # Convert to pytorch tensor
    img_tensor = transform(img_pil)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    
    _, predicted = torch.max(output, 1)
    
    # Display image with prediction using matplotlib
    plt.imshow(img_pil)
    plt.title(f"Prediction: {class_names[predicted.item()]}")
    plt.axis('off')  # Do not show axis
    plt.show()

predict_image(data_dir + 'baklava/1034361.jpg', model, data.classes)
predict_image(data_dir + 'bibimbap/1014434.jpg', model, data.classes)
predict_image(data_dir + 'donuts/104498.jpg', model, data.classes)

