import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth.tar'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def set_reproducibility(seed):
    """
    Set seed for random operation to ensure reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_transforms(mode='train'):
    """
    Get the data transforms both for input images and target annotations
    in a dictionary. 

    The function returns different transforms if we are in 'train' or 'test' mode.    
    """

    transforms = {}

    if mode == 'train':
        transforms['image'] = T.Compose([
            T.ToPILImage(),
            # T.Resize(224),
            T.RandomResizedCrop((224,224)),
            T.RandomHorizontalFlip(p = 0.5),
            # T.ColorJitter(brightness = .5, hue = .3),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        transforms['mask'] = T.Compose([
            T.ToPILImage(),
            T.Resize((224,224)),
            T.ToTensor()
        ])

        return transforms
    
    if mode == 'test':
        transforms['image'] = T.Compose([
            T.ToPILImage(),
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        transforms['mask'] = T.Compose([
            T.ToPILImage(),
            T.Resize((224,224)),
            T.ToTensor()
        ])

        return transforms
    
    raise ValueError("No valid 'mode' attribute. It must be 'train' or 'test'.")


def different_batch_size_collate_fn(batch):
    return tuple(zip(*batch))


def train_model(model=None, dataloaders=None, num_classes=104,
                device=None, criterion=None, optimizer=None,
                scheduler=None, num_epochs=25, patience=7,
                outfile='best_model_checkpoint.pth'):
    """
    Train a PyTorch model using the provided data and settings, with early stopping.


    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        dataloaders (dict): A dictionary containing the training and validation data loaders.
        device (torch.device): The device (CPU or GPU) where the model will be trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        num_epochs (int, optional): Number of epochs for training. Defaults to 25.
        patience (int, optional): Number of epochs with no improvement after which training stops. Defaults to 7.
        outfile (str, optional): The path where the best model checkpoint will be saved. Defaults to 'best_model_checkpoint.pth.tar'.

    Returns:
        torch.nn.Module: The trained model.
    """

    # Initialize early stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=outfile)

    # Helper function to compute IoU
    def compute_iou(pred, target, num_classes):
        iou_list = []
        pred = pred.view(-1)
        target = target.view(-1)

        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds[target_inds]).sum().float()
            union = pred_inds.sum().float() + target_inds.sum().float() - intersection
            if union == 0:
                iou_list.append(float('nan'))
            else:
                iou_list.append((intersection / union).item())
        return iou_list

    # Loop through each epoch
    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Storage for the IoU values
        iou_values = {phase: [] for phase in ['train', 'val']}

        # Iterate through both training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                print("Training:", end=" ")
                model.train()
            else:
                print("Evaluation:", end=" ")
                model.eval()

            running_loss = 0.0

            # Iterate through the data batches
            for inputs, labels in tqdm(dataloaders[phase], desc=f"Epoch {epoch+1}/{num_epochs} {phase}", unit="batch"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs_dict = model(inputs)
                    outputs = outputs_dict['out']
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                preds = torch.argmax(outputs, dim=1)
                running_loss += loss.item() * inputs.size(0)

                # Compute and store IoU for this batch
                iou_batch = compute_iou(preds, labels, num_classes)
                iou_values[phase].append(iou_batch)

            # Update learning rate using the scheduler only in the training phase
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_iou = np.nanmean(np.array(iou_values[phase]))
            print('{} Loss: {:.4f} mIoU: {:.4f}'.format(phase, epoch_loss, epoch_iou))

            # Reset IoU values for the next epoch
            iou_values[phase] = []

            # Check for early stopping in the validation phase
            if phase == 'val':
                early_stopping(epoch_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    model.load_state_dict(torch.load(outfile))
                    return model
    
    # Load best model found
    model.load_state_dict(torch.load(outfile))
    return model


def tensor_image_show(inp, title=None, denorm=True):
    """Display tensor image."""
    inp = inp.numpy().transpose((1, 2, 0))

    if denorm:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=6, class_names=None, dataloaders=None, device=None):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                tensor_image_show(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def visualize_model_predictions(model, img_path, class_names=None, device=None, transform=None):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        tensor_image_show(img.cpu().data[0])

        model.train(mode=was_training)
