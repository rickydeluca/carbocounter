import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from models.deeplabv3 import deeplabv3
from utils.segmentation_training import set_reproducibility, train_model, get_transforms, different_batch_size_collate_fn
from datasets.foodseg103_dataset import FoodSeg103SemanticDataset

def main():
    freeze_weights = True
    backbone = 'resnet50'

    set_reproducibility(seed=42)

    # Set default device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define dataset classes
    num_classes = 104   # 103 food classes + background

    # Define useful paths
    data_dir = 'data/FoodSeg103'                    # Dataset
    model_dir = 'best_models'                       # Where to store the best model found
    category_file   = data_dir + '/category_id.txt' # Where to find the index to category file

    # Define transformations for data augmentation    
    train_transforms = get_transforms(mode='train')
    test_transforms = get_transforms(mode='test')

    # Load datasets
    train_dataset = FoodSeg103SemanticDataset(data_dir, mode='train', transforms=train_transforms)
    val_dataset = FoodSeg103SemanticDataset(data_dir, mode='test', transforms=test_transforms)

    # DEBUG: Check loaded data
    img, mask = train_dataset[0]
    val_dataset[0]

    unique_classes = torch.unique(mask)
    print("\n\n*** Before Transform ***")
    print("Image shape: ", img.shape)
    print("Mask shape: ", mask.shape)
    print("Unique classes: ", unique_classes)

    exit(0)

    # Get the class names dictionary (class_id: class_name)
    id2class = {}
    with open(category_file, 'r') as f:
        for line in f:
            key, value = line.split('\t')
            id2class[key] = value

    # Define DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        # collate_fn=different_batch_size_collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        # collate_fn=different_batch_size_collate_fn
    )

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Get the model
    model = deeplabv3(
        backbone=backbone,
        num_classes=104,
        pretrained=True,
        freeze_weights=freeze_weights,
        progress=True
    )

    # Move model to default device
    model.to(device)

    # Define trainable parameters
    if freeze_weights: # Only parameters of final classficator are being optimized 
        params = [p for p in model.classifier.parameters() if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    # Define optimizer
    optimizer = torch.optim.SGD(
        params,
        lr=0.001,
        momentum=0.9
        # weight_decay=0.0005
    )

    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Train with early stopping
    train_model(
        model = model,
        optimizer = optimizer,
        dataloaders = dataloaders,
        num_classes = num_classes,
        criterion = criterion,
        scheduler = lr_scheduler,
        num_epochs = 25,
        patience = 7,
        device = device,
        outfile = model_dir + f"/deeplabv3_{backbone}.pth"
    )

if __name__ == '__main__':
    main()






