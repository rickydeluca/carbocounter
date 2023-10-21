import torch
import cv2 as cv
import matplotlib.pyplot as plt
import torchvision.transforms as T

from utils.segmentation import *
from models.deeplabv3 import deeplabv3
from skimage.segmentation import slic, clear_border, expand_labels
from skimage.color import label2rgb
from PIL import Image


class FoodSegmenter:
    """
    A class to segment food items in an image.

    Attributes:
        img (ndarray): The input image.
        plate_mask (ndarray): The mask indicating the plate region.
        segmentation_map (ndarray): The final segmented map.
        slic (bool): If True, use the SLIC algorithm for segmentation.
    """

    def __init__(self, model="slic", params_path="best_models/deeplabv3_resnet50.pth"):
        self.img = None
        self.plate_mask = None
        self.segmentation_map = None
        self.model = model               # If True, use SLIC algorithm
        self.params_paths = params_path if model == 'deeplabv3' else None

    def __call__(self, img, plate_coords, display=False):
        return self.segment_food(img, plate_coords, display=display)
    
    def preprocess_opencv_image(self, cv2_image):
        # Convert BGR to RGB
        rgb_image = cv.cvtColor(cv2_image, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        preprocess = T.Compose([
            T.Resize((256, 256)),  
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(pil_image).unsqueeze(0)  # add a batch dimension
    
    def segment_food(self, img, plate_mask, display=False):
        """
        Segments the food items in the given image.

        Args:
            img (ndarray): The input image.
            plate_mask (ndarray): The mask indicating the plate region.
            display (bool, optional): If True, display the segmentation results. Defaults to False.

        Returns:
            ndarray: The segmented map.
        """
        
        self.img = img
        self.plate_mask = plate_mask

        if self.model == "slic":

            # Convert to RGB for SLIC
            image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # Apply SLIC
            image_segments = slic(image_rgb,
                                  n_segments=30,
                                  compactness=10,
                                  max_num_iter=10,
                                  convert2lab=True,
                                  min_size_factor=0.3,
                                  max_size_factor=3)
            
            # Clear borders
            image_segments_filtered = clear_border(image_segments, mask=self.plate_mask)

            # Region growing
            image_segments_merged = expand_labels(image_segments_filtered, distance=10)
            
            if display:                 # Display output
                
                plt.subplot(2,2,1)      # Original
                plt.imshow(image_rgb)

                plt.subplot(2,2,2)      # Segmented
                plt.imshow(label2rgb(image_segments,
                                     image_rgb,
                                     kind = 'avg'))

                plt.subplot(2,2,3)      # Filtered
                plt.imshow(label2rgb(image_segments_filtered,
                                     image_rgb,
                                     kind = 'avg'))
                
                plt.subplot(2,2,4)      # Region growing
                plt.imshow(label2rgb(image_segments_merged,
                                     image_rgb,
                                     kind = 'avg'))
                
                plt.show()

            # Store the final segmentation map
            self.segmentation_map = image_segments_merged
            
            return self.segmentation_map
        
        if self.model == "deeplabv3":
            
            # Set default device
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
            input_tensor = self.preprocess_opencv_image(self.img).to(device)

            # Load model...
            model = deeplabv3(backbone="resnet50",
                              num_classes=104,
                              pretrained=True,
                              freeze_weights=True,
                              progress=True).to(device)
            
            # ...and weights
            model.load_state_dict(torch.load(self.params_paths))

            # Inference
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)

            print("output: ", output["out"].shape)

            # Post-process the Output
            predictions = torch.argmax(output["out"], dim=1)  # get the most likely prediction for each pixel
            self.segmentation_map = predictions.squeeze().cpu().numpy()

            print("predicions", predictions)
            print("segmentation_map", np.unique(self.segmentation_map))

            return self.segmentation_map


