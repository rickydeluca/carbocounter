import os
import torch
import warnings
import glob

# import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from PIL import Image

from utils.training import tensor_image_show


# class FoodSeg103Dataset_Instance(Dataset):
#     """
#     A PyTorch Dataset for handling 103-class food segmentation.

#     This dataset assumes that the segmentation masks are encoded with pixel values
#     ranging from 0 to 103, where each unique value represents a different class
#     and 0 is reserved for the background.

#     Attributes:
#         root (str): The root directory where the dataset is stored.
#         transforms (callable): A function/transform that takes in an image and target
#             and returns a transformed version.
#         mode (str): The mode in which the dataset is used. Can be 'train' or 'test'.
#         imgs (list of str): A list of image file names.
#         masks (list of str): A list of mask file names.

#     Note:
#         Ensure that the mask images are properly encoded with values from 0 to 103,
#         where each value represents a different class (including the background as class 0).

#     """
#     def __init__(self, root, transforms, mode='train', start=0, end=None):
#         self.root       = root
#         self.transforms = transforms
#         self.mode       = mode

#         # Load all image files and sorting them to ensure that they are aligned
#         if end is not None:
#             self.imgs   = list(sorted(os.listdir(os.path.join(root, f"img_dir/{mode}"))))[start:end]
#             self.masks  = list(sorted(os.listdir(os.path.join(root, f"ann_dir/{mode}"))))[start:end]
#         else:
#             self.imgs   = list(sorted(os.listdir(os.path.join(root, f"img_dir/{mode}"))))
#             self.masks  = list(sorted(os.listdir(os.path.join(root, f"ann_dir/{mode}"))))


#     def __getitem__(self, idx):
#         # Load images and masks
#         img_path    = os.path.join(self.root, f"img_dir/{self.mode}", self.imgs[idx])
#         mask_path   = os.path.join(self.root, f"ann_dir/{self.mode}", self.masks[idx])
#         img         = read_image(img_path)
#         mask        = read_image(mask_path, mode=ImageReadMode.GRAY).squeeze(0)

#         # Instances are encoded as different colors (0 - 103)
#         obj_ids = torch.unique(mask)

#         # First id is the background, so remove it
#         obj_ids     = obj_ids[1:]
#         num_objs    = len(obj_ids)

#         # Split the color-encoded mask into a set of binary masks
#         masks = torch.stack([(mask == obj_id).to(dtype=torch.uint8) for obj_id in obj_ids])

#         # Get bounding box coordinates for each mask
#         boxes = masks_to_boxes(masks)

#         # Check if boxes are valid
#         valid_boxes = []
#         valid_labels = []
#         valid_masks = []
#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = box
#             # Check if box dimensions are valid (non-negative and width/height > 0)
#             if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
#                 valid_boxes.append(box)
#                 valid_labels.append(obj_ids[i])
#                 valid_masks.append(masks[i])
#             else:
#                 warnings.warn(f"Invalid box {box} for object id {obj_ids[i]}, skipping.")

#         # Update variables with valid data
#         boxes = torch.stack(valid_boxes) if valid_boxes else torch.zeros((0, 4))
#         masks = torch.stack(valid_masks) if valid_masks else torch.zeros((0, *masks.size()[1:]))
#         labels = torch.tensor(valid_labels, dtype=torch.int64) if valid_labels else torch.zeros((0,), dtype=torch.int64)
#         num_objs = len(valid_boxes)

#         # Labels corresponf to the object IDs (1 - 103)
#         labels = obj_ids.to(torch.int64)

#         image_id = idx
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

#         # Suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

#         # Wrap sample and targets into torchvision tv_tensors:
#         img = tv_tensors.Image(img)

#         target = {}
#         target["boxes"]     = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
#         target["masks"]     = tv_tensors.Mask(masks)
#         target["labels"]    = labels
#         target["image_id"]  = image_id
#         target["area"]      = area
#         target["iscrowd"]   = iscrowd

#         if self.transforms is not None:
#             img, target = self.transforms(img, target)

#         return img, target

#     def __len__(self):
#         return len(self.imgs)
    

# class MyFoodSeg103Dataset_Instance(Dataset):
#     """
#     A PyTorch Dataset for handling 103-class food segmentation.

#     This dataset assumes that the segmentation masks are encoded with pixel values
#     ranging from 0 to 103, where each unique value represents a different class
#     and 0 is reserved for the background.

#     Attributes:
#         root (str): The root directory where the dataset is stored.
#         transforms (callable): A function/transform that takes in an image and target
#             and returns a transformed version.
#         imgs (list of str): A list of image file names.
#         masks (list of str): A list of mask file names.

#     Note:
#         Ensure that the mask images are properly encoded with values from 0 to 103,
#         where each value represents a different class (including the background as class 0).

#     """
#     def __init__(self, root, transforms, start=0, end=7118):
#         self.root       = root
#         self.transforms = transforms

#         # Load all image files and sorting them to ensure that they are aligned
#         self.imgs   = list(sorted(os.listdir(os.path.join(root, f"img_dir"))))[start:end]
#         self.masks  = list(sorted(os.listdir(os.path.join(root, f"ann_dir"))))[start:end]


#     def __getitem__(self, idx):
#         # Load images and masks
#         img_path    = os.path.join(self.root, f"img_dir", self.imgs[idx])
#         mask_path   = os.path.join(self.root, f"ann_dir", self.masks[idx])
#         img         = read_image(img_path)
#         mask        = read_image(mask_path, mode=ImageReadMode.GRAY).squeeze(0)

#         # Instances are encoded as different colors (0 - 103)
#         obj_ids = torch.unique(mask)

#         # First id is the background, so remove it
#         obj_ids     = obj_ids[1:]
#         num_objs    = len(obj_ids)

#         # Split the color-encoded mask into a set of binary masks
#         masks = torch.stack([(mask == obj_id).to(dtype=torch.uint8) for obj_id in obj_ids])

#         # Get bounding box coordinates for each mask
#         boxes = masks_to_boxes(masks)

#         # Check if boxes are valid
#         valid_boxes = []
#         valid_labels = []
#         valid_masks = []
#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = box
#             # Check if box dimensions are valid (non-negative and width/height > 0)
#             if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
#                 valid_boxes.append(box)
#                 valid_labels.append(obj_ids[i])
#                 valid_masks.append(masks[i])
#             else:
#                 warnings.warn(f"Invalid box {box} for object id {obj_ids[i]}, skipping.")

#         # Update variables with valid data
#         boxes = torch.stack(valid_boxes) if valid_boxes else torch.zeros((0, 4))
#         masks = torch.stack(valid_masks) if valid_masks else torch.zeros((0, *masks.size()[1:]))
#         labels = torch.tensor(valid_labels, dtype=torch.int64) if valid_labels else torch.zeros((0,), dtype=torch.int64)
#         num_objs = len(valid_boxes)

#         # Labels corresponf to the object IDs (1 - 103)
#         labels = obj_ids.to(torch.int64)

#         image_id = idx
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

#         # Suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

#         # Wrap sample and targets into torchvision tv_tensors:
#         img = tv_tensors.Image(img)

#         target = {}
#         target["boxes"]     = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
#         target["masks"]     = tv_tensors.Mask(masks)
#         target["labels"]    = labels
#         target["image_id"]  = image_id
#         target["area"]      = area
#         target["iscrowd"]   = iscrowd

#         if self.transforms is not None:
#             img, target = self.transforms(img, target)

#         return img, target

#     def __len__(self):
#         return len(self.imgs)
    

# class FoodSeg103Dataset_Semantic(Dataset):
#     """
#     FoodSeg 103 dataset for semantic segmnetation.

#     Reference: https://expoundai.wordpress.com/2019/08/30/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch/
#     """

#     def __init__(self, root, image_folder, mask_folder, transform=None, seed=None,
#                  split_size=None, subset=None, image_color_mode='rgb',
#                  mask_color_mode='grayscale'):
        
#         self.color_dict = {'rgb': 1, 'grayscale': 0}
#         assert(image_color_mode in ['rgb', 'grayscale'])
#         assert(mask_color_mode in ['rgb', 'grayscale'])

#         self.image_color_flag = self.color_dict[image_color_mode]
#         self.mask_color_flag = self.color_dict[mask_color_mode]
#         self.root = root
#         self.transform = transform

#         if split_size is None:
#             self.image_names = sorted(glob.glob(os.path.join(self.root, image_folder, '*')))
#             self.mask_names = sorted(glob.glob(os.path.join(self.root, mask_folder, '*')))

#         else:
#             assert(subset in ['train', 'test'])
#             self.split_size = split_size
#             self.image_list = np.array(sorted(glob.glob(os.path.join(self.root, image_folder, '*'))))
#             self.mask_list = np.array(sorted(glob.glob(os.path.join(self.root, mask_folder, '*'))))
            
#             if seed:
#                 np.random.seed(seed)

#             indices = np.arange(len(self.image_list))
#             np.random.shuffle(indices)

#             self.image_list = self.image_list[indices]
#             self.mask_list = self.mask_list[indices]

#             if subset == 'train':
#                 self.image_names = self.image_list[:int(np.ceil(len(self.image_list)*(1-self.split_size)))]
#                 self.mask_names = self.mask_list[:int(np.ceil(len(self.mask_list)*(1-self.split_size)))]
#             else:
#                 self.image_names = self.image_list[int(np.ceil(len(self.image_list)*(1-self.split_size))):]
#                 self.mask_names = self.mask_list[int(np.ceil(len(self.mask_list)*(1-self.split_size))):]

#     def __len__(self):
#         return len(self.image_names)
    
 
#     def __getitem__(self, idx):
#         img_name = self.image_names[idx]

#         if self.image_color_flag:
#             image = cv.imread(img_name, self.image_color_flag).transpose(2, 0, 1)
#         else:
#             image = cv.imread(img_name, self.image_color_flag)

#         msk_name = self.mask_names[idx]

#         if self.mask_color_flag:
#             mask = cv.imread(msk_name, self.mask_color_flag).transpose(2, 0, 1)
#         else:
#             mask = cv.imread(msk_name, self.mask_color_flag)
        
#         sample = {'image': image, 'mask': mask}
 
#         if self.transform:
#             sample = self.transform(sample)
 
#         return sample
    

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
        # image = cv.imread(img_path, cv.IMREAD_COLOR)
        # mask = cv.imread(ann_path, cv.IMREAD_GRAYSCALE)
        image = torch.from_numpy(np.array(Image.open(img_path)))
        mask = torch.from_numpy(np.array(Image.open(ann_path))).unsqueeze(2)

        # Reshape images
        image = torch.einsum('xyz -> zxy', image)
        mask = torch.einsum('xyz -> zxy', mask)

        # Now, torch.unique should return the expected 4 unique class labels
        # unique_classes = torch.unique(mask)
        # print("\n\n*** Before Transform ***")
        # print("Image shape: ", image.shape)
        # print("Mask shape: ", mask.shape)
        # print("Unique classes: ", unique_classes)

        # DEBUG: Display images before transformation
        # plt.figure(figsize=(10, 5))

        # plt.subplot(2, 2, 1)
        # tensor_image_show(image, 'Original Image', denorm=False)

        # plt.subplot(2, 2, 2)
        # tensor_image_show(mask, 'Original Mask', denorm=False)

        if self.transforms:
            image = self.transforms['image'](image)
            mask = self.transforms['mask'](mask)

        # DEBUG: Display images after transformation
        # print("\n\n*** After Transform ***")
        # print("Image shape: ", image.shape)
        # print("Mask shape: ", mask.shape)
        # print("Unique classes: ", unique_classes)
        # plt.subplot(2, 2, 3)
        # tensor_image_show(image, 'Transformed Image', denorm=True)

        # plt.subplot(2, 2, 4)
        # tensor_image_show(mask, 'Transformed Mask', denorm=False)

        # plt.tight_layout()
        # plt.show()

        return image, mask

    def __len__(self):
        return len(self.image_files)


 



