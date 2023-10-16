import cv2 as cv
import matplotlib.pyplot as plt

from utils.segmentation import *

from skimage.segmentation import slic, clear_border, expand_labels
from skimage.color import label2rgb

class FoodSegmenter:
    """
    A class to segment food items in an image.

    Attributes:
        img (ndarray): The input image.
        plate_mask (ndarray): The mask indicating the plate region.
        segmentation_map (ndarray): The final segmented map.
        slic (bool): If True, use the SLIC algorithm for segmentation.
    """

    def __init__(self, model="slic"):
        self.img = None
        self.plate_mask = None
        self.segmentation_map = None
        self.model = model               # If True, use SLIC algorithm

    def __call__(self, img, plate_coords, display=False):
        return self.segment_food(img, plate_coords, display=display)
    
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

