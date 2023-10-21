
import torch
import torchvision.transforms as transforms
import math
import numpy as np
import matplotlib.pyplot as plt

def rescale_tensor_image(image_tensor, max_pixels):
    """
    Rescale a torch tensor image such that its total number of pixels is less than N.
    
    Parameters:
        image_tensor (torch.Tensor): The input image as a torch tensor of shape (C, H, W).
        max_pixels (int): Maximum number of pixels for the rescaled image.
        
    Returns:
        torch.Tensor: The rescaled image as a torch tensor.
    """
    # Ensure the input is a torch tensor
    if not torch.is_tensor(image_tensor):
        raise TypeError("Input must be a torch tensor")
    
    # Get original width and height
    _, orig_height, orig_width = image_tensor.shape
    
    # Calculate the aspect ratio
    aspect_ratio = orig_width / orig_height
    
    # Calculate new width and height ensuring total pixels are under max_pixels
    new_width = int(math.sqrt(max_pixels * aspect_ratio))
    new_height = int(new_width / aspect_ratio)
    
    # Ensure the new width and height do not exceed original dimensions
    new_width = min(new_width, orig_width)
    new_height = min(new_height, orig_height)
    
    # Define the resizing transformation
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((new_height, new_width)),
        transforms.ToTensor()
    ])
    
    # Apply the transformation and return the rescaled image tensor
    rescaled_image_tensor = resize_transform(image_tensor)
    
    return rescaled_image_tensor


def imshow(img, title=None):
    # Convert from tensor to numpy, and de-normalize
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause a bit so that plots are updated