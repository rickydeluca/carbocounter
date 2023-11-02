import os
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from datasets.food101 import Food101Dataset

from models.ResNet.utils import imshow, train_model, visualize_model, visualize_model_predictions
from models.ResNet.resnet import get_resnet

cudnn.benchmark = True


def parse_args():
    """
    Read terminal inputs
    """

    parser = argparse.ArgumentParser(
        prog='Train ResNet',
        description='Given two stereo images representing the same dish, segment the different foods and compute the quanity of carbohydrates within them.'
    )
    
    parser.add_argument('--version', type=str, default='resnet50', help='Which Resnet version to use (default: resnet50).')
    parser.add_argument('--data_dir', type=str, default='data/food-101', help='Dataset directory (default: data/food-101).')
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


    # Load datasets
    image_datasets = {x: Food101Dataset(os.path.join(args.data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    # Init DataLoaders
    dataloaders = { x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=4,
                                                   shuffle=True,
                                                   num_workers=4)
                    for x in ['train', 'val']}
    
    # Get dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Get class names
    class_names = image_datasets['train'].classes

    # Set default device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Visualize few images (if verbose):
    if args.verbose:
        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        imshow(out, title=[class_names[x] for x in classes])


    # Load the model
    model = get_resnet(args.version,
                       num_classes=104,
                       pretrained=args.pretrained,
                       freeze_weights=args.freeze_weights)
    
    # Move it to the selected device
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    if args.freeze_weights:  # (Optimize only for the last layer parameters
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model and get the best obe
    model = train_model(model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=exp_lr_scheduler,
                        dataloaders=dataloaders,
                        dataset_sizes=dataset_sizes,
                        device=device,
                        num_epochs=args.epochs)

    visualize_model(model=model,
                    num_images=6,
                    class_names=class_names,
                    dataloaders=dataloaders,
                    device=device)
    plt.ioff()
    plt.show()

    # Save the best model parameters
    torch.save(model.state_dict(), args.model_dir + "/" + args.version + ".pth")

    # Visualize model predictions
    visualize_model_predictions(model=model,
                                img_path=args.data_dir + "/train/cannoli/526.jpg",
                                data_transforms=data_transforms,
                                class_names=class_names,
                                device=device)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()