import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet101


def get_resent18(num_classes: int, pretrained: bool = True, freeze_weights:bool = False):

    # Define weights
    weights = 'IMAGENET1K_V1' if pretrained else None

    # Load model
    model =  resnet18(weights=weights)

    # Freeze weights if needed
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def get_resent50(num_classes: int, pretrained: bool = True, freeze_weights:bool = False):

    # Define weights
    weights = 'IMAGENET1K_V1' if pretrained else None

    # Load model
    model =  resnet50(weights=weights)

    # Freeze weights if needed
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def get_resent101(num_classes: int, pretrained: bool = True, freeze_weights:bool = False):

    # Define weights
    weights = 'IMAGENET1K_V1' if pretrained else None

    # Load model
    model =  resnet101(weights=weights)

    # Freeze weights if needed
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def get_resnet(version: str, num_classes: int, pretrained: bool = True, freeze_weights:bool = False):

    if version == 'resent18':
        return get_resent18(num_classes, pretrained=pretrained, freeze_weights=freeze_weights)
    
    if version == 'resent50':
        return get_resent50(num_classes, pretrained=pretrained, freeze_weights=freeze_weights)
    
    if version == 'resnet101':
        return get_resent101(num_classes, pretrained=pretrained, freeze_weights=freeze_weights)

    raise ValueError("No valid resnet version. Choose from 'resnet18', 'resent50' or 'resent101'")