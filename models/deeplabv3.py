import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from torchvision.models.segmentation import *
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def deeplabv3(backbone, num_classes=104, new_classifier=False, pretrained=True, freeze_weights=False, progress=False):
    
    weights = 'DEFAULT' if pretrained else None
    
    # Load backbone
    if backbone == "resnet50":
        model = deeplabv3_resnet50(weights=weights, progress=progress)
    
    elif backbone == "resnet101":
        model = deeplabv3_resnet101(weights=weights, progress=progress)
        
    
    elif backbone == "mobilenet_v3_large":
        model = deeplabv3_mobilenet_v3_large(weights=weights, progress=progress)
        
    else:
        raise ValueError("No valid 'backbone', please choose from: 'resnet50', 'resnet101' and 'mobilenet_v3_large'")
    
    
    # Freeze weights if requested
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False
    
    # Modify classifier
    if new_classifier:
        model.classifier = DeepLabHead(2048, num_classes)
        # model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    return model


if __name__ == "__main__":
    model = deeplabv3("resnet50", pretrained=True, progress=True)
    model.eval()

    # Sample execution (requires torchvision)
    input_image = Image.open("test/shepard_01.png")
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    print("Semantic map shape:", output_predictions.shape)
    print("Unique semantic labels: ", torch.unique(output_predictions))

    # Create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # Plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    plt.imshow(r)
    plt.show()
