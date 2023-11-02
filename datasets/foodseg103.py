import os
import cv2 as cv
from torch.utils.data import Dataset

class FoodSeg103SemanticDataset(Dataset):
    """
    A PyTorch Dataset for 103-class food segmentation.

    This dataset provides an interface to a food segmentation dataset with 103 classes.
    It assumes that segmentation masks are encoded with pixel values ranging from 0 to 103,
    where each unique value represents a different class and 0 is reserved for the background.

    Note:
        Ensure that the mask images are properly encoded with values 
        from 0 to 103, where each value represents a different class
        (including the background as class 0).

    Attributes:
        root_dir (str): The root directory where the dataset is stored.
        transforms (callable, optional): A function/transform that takes in an image and its annotation and returns a transformed version.
        mode (str): Mode in which the dataset is used. Can be 'train' or 'test'.
        img_dir (str): Directory for the images.
        ann_dir (str): Directory for the annotations (masks).
        image_files (list of str): List of image filenames.
        ann_files (list of str): List of annotation filenames.

    Args:
        root_dir (str): The root directory where the dataset is stored.
        transforms (callable, optional): Optional transform to be applied on a sample.
        mode (str, optional): Mode in which the dataset is used. Defaults to 'train'.

    Returns:
        tuple: A tuple containing the image and its corresponding mask.
    """
    def __init__(self, root_dir, transforms=None, mode='train'):
        self.root_dir    = root_dir
        self.transforms  = transforms
        self.mode        = mode
        self.img_dir     = os.path.join(root_dir, 'Images', 'img_dir', mode)
        self.ann_dir     = os.path.join(root_dir, 'Images', 'ann_dir', mode)
        self.image_files = sorted(os.listdir(self.img_dir))
        self.ann_files   = sorted(os.listdir(self.ann_dir))

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding mask by index.

        Args:
            idx (int): Index of the image and mask pair to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding mask.
        """
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        ann_path = os.path.join(self.ann_dir, self.ann_files[idx])

        # Read image and mask using OpenCV
        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mask = cv.imread(ann_path, cv.IMREAD_GRAYSCALE)

        if self.transforms:
            transformed = self.transforms[self.mode](image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

        return transformed_image, transformed_mask

    def __len__(self):
        return len(self.image_files)


 



