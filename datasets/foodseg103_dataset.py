import os
import torch
import warnings

from torchvision.io import read_image, ImageReadMode
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class FoodSeg103Dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for handling 103-class food segmentation.

    This dataset assumes that the segmentation masks are encoded with pixel values
    ranging from 0 to 103, where each unique value represents a different class
    and 0 is reserved for the background.

    Attributes:
        root (str): The root directory where the dataset is stored.
        transforms (callable): A function/transform that takes in an image and target
            and returns a transformed version.
        mode (str): The mode in which the dataset is used. Can be 'train' or 'test'.
        imgs (list of str): A list of image file names.
        masks (list of str): A list of mask file names.

    Note:
        Ensure that the mask images are properly encoded with values from 0 to 103,
        where each value represents a different class (including the background as class 0).

    """
    def __init__(self, root, transforms, mode='train'):
        self.root       = root
        self.transforms = transforms
        self.mode       = mode

        # Load all image files and sorting them to ensure that they are aligned
        self.imgs   = list(sorted(os.listdir(os.path.join(root, f"img_dir/{mode}"))))
        self.masks  = list(sorted(os.listdir(os.path.join(root, f"ann_dir/{mode}"))))


    def __getitem__(self, idx):
        # Load images and masks
        img_path    = os.path.join(self.root, f"img_dir/{self.mode}", self.imgs[idx])
        mask_path   = os.path.join(self.root, f"ann_dir/{self.mode}", self.masks[idx])
        img         = read_image(img_path)
        mask        = read_image(mask_path, mode=ImageReadMode.GRAY).squeeze(0)

        # Instances are encoded as different colors (0 - 103)
        obj_ids = torch.unique(mask)

        # First id is the background, so remove it
        obj_ids     = obj_ids[1:]
        num_objs    = len(obj_ids)

        # Split the color-encoded mask into a set of binary masks
        masks = torch.stack([(mask == obj_id).to(dtype=torch.uint8) for obj_id in obj_ids])

        # Get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # Check if boxes are valid
        valid_boxes = []
        valid_labels = []
        valid_masks = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Check if box dimensions are valid (non-negative and width/height > 0)
            if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                valid_boxes.append(box)
                valid_labels.append(obj_ids[i])
                valid_masks.append(masks[i])
            else:
                warnings.warn(f"Invalid box {box} for object id {obj_ids[i]}, skipping.")

        # Update variables with valid data
        boxes = torch.stack(valid_boxes) if valid_boxes else torch.zeros((0, 4))
        masks = torch.stack(valid_masks) if valid_masks else torch.zeros((0, *masks.size()[1:]))
        labels = torch.tensor(valid_labels, dtype=torch.int64) if valid_labels else torch.zeros((0,), dtype=torch.int64)
        num_objs = len(valid_boxes)

        # Labels corresponf to the object IDs (1 - 103)
        labels = obj_ids.to(torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"]     = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"]     = tv_tensors.Mask(masks)
        target["labels"]    = labels
        target["image_id"]  = image_id
        target["area"]      = area
        target["iscrowd"]   = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __getitem__box_friendly(self, idx):
        # Load images and masks
        img_path    = os.path.join(self.root, f"img_dir/{self.mode}", self.imgs[idx])
        mask_path   = os.path.join(self.root, f"ann_dir/{self.mode}", self.masks[idx])
        img         = read_image(img_path).numpy()  # Convert to NumPy array for Albumentations
        mask        = read_image(mask_path).squeeze(0).numpy()

        # Instances are encoded as different colors (0 - 103)
        obj_ids = torch.unique(mask)

        # First id is the background, so remove it
        obj_ids     = obj_ids[1:]
        num_objs    = len(obj_ids)

        # Split the color-encoded mask into a set of binary masks
        masks = torch.stack([(mask == obj_id).to(dtype=torch.uint8) for obj_id in obj_ids])

        # Get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks).numpy()  # Convert to NumPy array for Albumentations

        # Labels correspond to the object IDs (1 - 103)
        labels = obj_ids.numpy()  # Convert to NumPy array for Albumentations

        # Apply transformations
        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=boxes, masks=masks, class_labels=labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            masks = transformed["masks"]
            labels = transformed["class_labels"]

        # Convert back to PyTorch tensors
        img = torch.from_numpy(img).float()
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Calculate area for bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"]     = boxes
        target["masks"]     = masks
        target["labels"]    = labels
        target["image_id"]  = torch.tensor([idx])
        target["area"]      = area
        target["iscrowd"]   = iscrowd

        return img, target
    

    def __len__(self):
        return len(self.imgs)