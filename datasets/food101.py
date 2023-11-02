import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def Food101Dataset(root, transform=None):
    return ImageFolder(root=root, transform=transform)


class Food101Subset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample, label = self.dataset[self.indices[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
    

def load_food101(data_path, image_size=(224, 224), num_samples_per_category=None):
    """
    Load the Food-101 dataset from the specified path.

    Parameters:
    - data_path (str): The path to the extracted Food-101 dataset.
    - image_size (tuple): The target size to which images will be resized.
    - num_samples_per_category (int, optional): Number of images to load per category. 
      If None, all images are loaded.

    Returns:
    - X (np.array): Numpy array containing the image data.
    - y (np.array): Numpy array containing the corresponding labels.
    - categories (list): List of category (label) names.
    """
    
    # Get list of categories (labels)
    categories = os.listdir(data_path)

    # Initialize lists to store images and labels
    X = []
    y = []

    # Create a progress bar
    pbar = tqdm(total=num_samples_per_category * len(categories) if num_samples_per_category else None, 
                desc='Loading images', 
                dynamic_ncols=True)

    # Iterate through each category
    for label, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        
        # Get a list of image files, and optionally subsample them
        image_files = os.listdir(category_path)
        if num_samples_per_category:
            image_files = image_files[:num_samples_per_category]
        
        # Iterate through each image in the category
        for image_file in image_files:
            image_path = os.path.join(category_path, image_file)
            
            # Load and preprocess the image using OpenCV
            image = cv.imread(image_path)
            image = cv.resize(image, image_size)  # Resize the image
            
            # Append the image and label
            X.append(image)
            y.append(label)
            
            # Update the progress bar
            pbar.update(1)

    # Close the progress bar
    pbar.close()

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y, categories