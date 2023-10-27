"""
source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""
import os
import time
import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from tempfile import TemporaryDirectory
from sklearn.metrics import jaccard_score


def compute_mIoU(preds, labels, num_classes):
    preds = preds.cpu().numpy().reshape(-1)
    labels = labels.cpu().numpy().reshape(-1)
    
    iou_list = []
    for cls in range(num_classes):
        iou = jaccard_score(labels, preds, labels=[cls], average='macro', zero_division=0)
        iou_list.append(iou)
    
    return np.mean(iou_list)


def train_model(model=None, criterion=None, optimizer=None, scheduler=None, dataloaders=None, dataset_sizes=None, device=None, num_classes=104, num_epochs=25):

    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_iou = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase], desc=f"Epoch {epoch+1}/{num_epochs} {phase}", unit="batch"):
                    inputs = inputs.to(device)
                    labels = labels.to(device).to(torch.long)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs_dict = model(inputs)
                        outputs = outputs_dict['out']  # Access the 'out' key to get the actual outputs
                        preds = torch.argmax(outputs, dim=1)

                        # DEBUG
                        # print(f"unique labels: {torch.unique(labels)}")
                        # print("***********************************")
                        # print(f"outputs \t shape {outputs.shape} \t dtype: {outputs.dtype}")
                        # print(f"labels \t shape {labels.shape} \t dtype: {labels.dtype}")
                        # exit()

                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)


                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_iou = compute_mIoU(preds, labels, num_classes)

                print(f'{phase} Loss: {epoch_loss:.4f} IoU: {epoch_iou:.4f}')

                # deep copy the model
                if phase == 'test' and epoch_iou > best_iou:
                    best_iou = epoch_iou
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val IoU: {best_iou:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

    return model


def imshow(inp, title=None, normalize=True):
    """Imshow for Tensor."""
    
    if len(inp.shape) == 2:  # if 2D tensor (like a mask)
        inp = inp.numpy()
        if normalize:
            inp = (inp - inp.min()) / (inp.max() - inp.min())  # Normalize to range [0,1]
        plt.imshow(inp, cmap='gray')  # using gray colormap for 2D
    else:
        inp = inp.numpy().transpose((1, 2, 0))
        if normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
    
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



def visualize_model(model=None, num_images=6, dataloaders=None, device=None):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))  # Adjusted figure size for clarity

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs_dict = model(inputs)
            outputs = outputs_dict['out']
            preds = torch.argmax(outputs, dim=1)

            for j in range(inputs.size()[0]):
                images_so_far += 1

                # Display input image
                ax = plt.subplot(num_images, 2, images_so_far*2-1)
                ax.axis('off')
                ax.set_title(f'Input Image')
                imshow(inputs.cpu().data[j])

                # Display predicted mask
                ax = plt.subplot(num_images, 2, images_so_far*2)
                ax.axis('off')
                ax.set_title(f'Predicted Mask')
                imshow(preds.cpu().data[j], normalize=False)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def visualize_model_predictions(model=None, img_path=None, data_transforms=None, device=None):
    was_training = model.training
    model.eval()

    # Load and preprocess image
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_transformed = data_transforms['test'](image=img)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs_dict = model(img_transformed)
        outputs = outputs_dict['out']
        preds = torch.argmax(outputs, dim=1)

        # Display original image
        ax = plt.subplot(1, 2, 1)
        ax.axis('off')
        ax.set_title('Original Image')
        plt.imshow(img)

        # Display predicted mask
        ax = plt.subplot(1, 2, 2)
        ax.axis('off')
        ax.set_title('Predicted Mask')
        # Assuming preds is a 2D tensor since it's a single image
        plt.imshow(preds.cpu().data[0], cmap='gray')  # Use gray colormap for the mask

    model.train(mode=was_training)

    plt.show()  # Added to display the plots


