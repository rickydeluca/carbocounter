import torch

from train_utils.engine import train_one_epoch, train_one_epoch_v2, evaluate
from train_utils.utils import collate_fn

from datasets.foodseg103_dataset import FoodSeg103Dataset, MyFoodSeg103Dataset
from models.maskrcnn import *

# Reproducibility
torch.manual_seed(42)

# Set default device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define dataset classes
num_classes = 104   # 103 food classes + background

# Useful paths
data_dir = 'data/MyFoodSeg103' # 'test/MiniFoodSeg103/Images'
model_dir = 'best_models'

# Load datasets and split them
dataset_size    = 7118
train_size      = dataset_size - 50
train_dataset   = MyFoodSeg103Dataset(data_dir, get_transform(train=True), start=0, end=train_size)
test_dataset    = MyFoodSeg103Dataset(data_dir, get_transform(train=False), start=0, end=dataset_size)


# Get the class names dictionary (class_id: class_name)
id2class = {}
with open(data_dir + "/category_id.txt", "r") as f:
    for line in f:
        key, value = line.split('\t')
        id2class[key] = value

# Define DataLoaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# Get the model
model = get_model_instance_segmentation(num_classes)

# Move model to default device
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# Learning Rate Scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# Initialize GradScaler for automatic mixed precision (AMP)
scaler = torch.cuda.amp.GradScaler()

# Train with early stopping
num_epochs          = 30  
best_metric         = -float('inf')
patience            = 3
no_improve_epochs   = 0

for epoch in range(num_epochs):
    
    # Train for one epoch, print every 10 iterations
    train_one_epoch(model,
                    optimizer,
                    train_dataloader,
                    device,
                    epoch,
                    print_freq=10,
                    scaler=scaler)

    # Update learning rate
    lr_scheduler.step()

    # Evaluate
    results = evaluate(model,
                       test_dataloader,
                       device=device)

    # Extract the desired metric for early stopping
    current_metric = results.coco_eval['bbox'].stats[2]     # mAP at IoU=0.75
    
    # Check for metric improvement
    if current_metric > best_metric:
        best_metric = current_metric
        no_improve_epochs = 0                               # Reset counter
        torch.save(model.state_dict(), model_dir + "/maskrcnn_best.pth")
        print(f"New best metric: {best_metric:.4f}, model saved.")
    else:
        no_improve_epochs += 1
        print(f"No improvement in metric. Patience: {no_improve_epochs}/{patience}.")
        
    # Check for early stopping
    if no_improve_epochs >= patience:
        print("Early stopping triggered.")
        break