import torch
import random
import numpy as np
import torchvision.transforms as T

from PIL import Image
from torchvision.models.segmentation import *
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def deeplabv3(backbone, num_classes=104, pretrained=True, freeze_weights=True, progress=False):
    
    weights = 'DEFAULT' if pretrained else None
    
    if backbone == "resnet50":
        model = deeplabv3_resnet50(weights=weights, progress=progress)
    
    elif backbone == "resnet101":
        model = deeplabv3_resnet101(weights=weights, progress=progress)
        
    
    elif backbone == "mobilenet_v3_large":
        model = deeplabv3_mobilenet_v3_large(weights=weights, progress=progress)
        
    else:
        raise ValueError("No valid 'backbone', please choose from: 'resent50', 'resnet101' and 'mobilenet_v3_large'")
    
    
    if freeze_weights:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
    
    # Substitute the classifier to handle our number number of classes
    model.classifier = DeepLabHead(2048, num_classes)
    
    # DEBUG: Check params with grad
    # params_with_grad = [p for p in model.parameters() if p.requires_grad]
    # print("params_with_grad:", len(params_with_grad))

    return model



class CustomTransforms:
    def __init__(self, mean, std, is_train=True):
        self.mean = mean
        self.std = std
        self.is_train = is_train

    def __call__(self, img, ann):
        if self.is_train:
            img, ann = self._train_transform(img, ann)
        else:
            img, ann = self._test_transform(img, ann)

        return img, ann

    def _train_transform(self, img, ann):
        joint_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(0.75, 1.33), interpolation=Image.BILINEAR)
        ])
        
        img, ann = self._joint_transform(joint_transforms, img, ann)

        img_transforms = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

        img = img_transforms(img)
        ann = T.ToTensor()(ann).squeeze().long()

        return img, ann

    def _test_transform(self, img, ann):
        joint_transforms = T.Resize((256, 256))
        
        img = joint_transforms(img)
        ann = joint_transforms(ann)
        
        img = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])(img)
        ann = T.ToTensor()(ann).squeeze().long()

        return img, ann

    @staticmethod
    def _joint_transform(transform, img, ann):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        img = transform(img)

        random.seed(seed)  # Ensure same transform for the annotation
        ann = transform(ann)

        return img, ann

    
