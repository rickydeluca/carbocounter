import os
import torch
import argparse
import albumentations as A
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler
from models.deeplabv3 import deeplabv3
from albumentations.pytorch import ToTensorV2

from utils.semantic_segmentation import train_model, visualize_model, visualize_model_predictions
from datasets.foodseg103_dataset import FoodSeg103SemanticDataset

cudnn.benchmark = True


def parse_args():
    """
    Read terminal inputs
    """

    parser = argparse.ArgumentParser(
        prog='Train DeepLabV3',
        description='Transfer learning on torchvision deeplabv3 to perform semantic segmenatation over food dishes.'
    )
    
    parser.add_argument('--backbone', type=str, default='resnet50', help='Which Resnet version to use as backbone(default: resnet50).')
    parser.add_argument('--data_dir', type=str, default='data/FoodSeg103', help='Dataset directory (default: data/FoodSeg103).')
    parser.add_argument('--classes_file', type=str, default='data/FoodSeg103/classes.txt', help='Path to the file containing the food class names. txt format; one class per row. (default: "data/FoodSeg103/classes.txt")')
    parser.add_argument('--model_dir', type=str, default='best_models', help='Directory in which save the best trained model (default: best_models).')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs (default: 25).')
    parser.add_argument('-p', '--pretrained', action='store_true', help='Load the model with the pretrained weights.')
    parser.add_argument('-f', '--freeze_weights', action='store_true', help='Train only the last layer parameters.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display the process step by step.')

    return parser.parse_args()


def main():

    args = parse_args()

    # Define data transformations

    data_transforms = {

        'train': A.Compose([
            A.SmallestMaxSize(max_size=256),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=224, width=224),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]),

        'test': A.Compose([
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]),
    }


    # Load datasets
    image_datasets = {x: FoodSeg103SemanticDataset('data/FoodSeg103', transforms=data_transforms, mode=x) for x in ['train', 'test']}

    # DEBUG: 
    # train_image, train_mask = image_datasets['train'][0]
    # test_image, test_mask = image_datasets['test'][0]
    # print("Image shape: ", train_image.shape)
    # print("Mask shape: ", train_mask.shape)
    # print("Unique mask labels: ", torch.unique(train_mask))
    # exit(0)
    
    # Init DataLoaders
    dataloaders = { x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=4,
                                                   shuffle=True if x=='train' else False,
                                                   num_workers=4)
                    for x in ['train', 'test']}
    
    # Get dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    # Get class names
    idx2class = {}
    with open(args.classes_file) as f:
        for idx, name in enumerate(f):
            idx2class[idx] = name.replace("\n", "")

    # print(idx2class)
    # exit()

    # Set default device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = deeplabv3(args.backbone,
                      num_classes=104,
                      pretrained=args.pretrained,
                      new_classifier=True,
                      freeze_weights=args.freeze_weights)
    
    # Move it to the selected device
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    if args.freeze_weights:  # (Optimize only for the last layer parameters
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # DEBUG
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # exit()

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    model = train_model(model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=exp_lr_scheduler,
                        dataloaders=dataloaders,
                        dataset_sizes=dataset_sizes,
                        device=device,
                        num_classes = 104,
                        num_epochs=args.epochs)

    visualize_model(model=model,
                    num_images=6,
                    dataloaders=dataloaders,
                    device=device)
    plt.ioff()
    plt.show()

    # Save best model params
    torch.save(model.state_dict(), args.model_dir + "/" + "deeplabv3_" + args.backbone + ".pth")

    # Visualize predictions
    visualize_model_predictions(model=model,
                                img_path=args.data_dir + "/Images/img_dir/test/00000048.jpg",
                                data_transforms=data_transforms,
                                device=device)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()





