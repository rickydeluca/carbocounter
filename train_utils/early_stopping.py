import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth.tar'):
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

def train_model(model=None, dataloaders=None, device=None, criterion=None, 
                optimizer=None, scheduler=None, num_epochs=25, 
                patience=7, outfile='best_model_checkpoint.pth.tar'):
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

    # Loop through each epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Iterate through both training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate through the data batches
            for inputs, labels in dataloaders[phase]:
                # Move inputs and labels to the specified device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero out the gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization only in the training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update loss and accuracy metrics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Update learning rate using the scheduler only in the training phase
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Check for early stopping in the validation phase
            if phase == 'val':
                early_stopping(epoch_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    return model

    return model

