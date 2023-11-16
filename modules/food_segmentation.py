import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from skimage.graph import cut_threshold, rag_mean_color
from skimage.morphology import binary_erosion
from skimage.color import label2rgb
from skimage.segmentation import slic, clear_border, expand_labels


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
        self.model = model


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

            # image_cielab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            # cv.imshow("CIELAB image", image_cielab)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # mean_shift_image = cv.pyrMeanShiftFiltering(img, sp=30, sr=30, maxLevel=4)
            # cv.imshow("Mean shifted mage", mean_shift_image)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # exit()

            # Apply SLIC
            segments = slic(image_rgb,
                            n_segments=20,
                            compactness=10,
                            sigma=2,
                            mask=self.plate_mask,
                            max_num_iter=10,
                            convert2lab=True,
                            min_size_factor=0.3,
                            max_size_factor=3)
            
            # Clear border regions
            # segments = expand_labels(segments, distance=10)
            mask = binary_erosion(self.plate_mask)
            cleared_segments = clear_border(segments, mask=mask)

            # Region growing
            try:
                g = rag_mean_color(img, cleared_segments)
                growed_segments = cut_threshold(cleared_segments, g, 29)
            except:
                growed_segments = cleared_segments

            growed_segments = clear_border(growed_segments)

            if display:                 # Display output

                plt.subplot(2,2,1)      # Original
                plt.imshow(image_rgb)

                plt.subplot(2,2,2)      # Segmented
                plt.imshow(label2rgb(segments,
                                     image_rgb,
                                     kind = 'avg'))

                plt.subplot(2,2,3)      # Cleared
                plt.imshow(label2rgb(cleared_segments,
                                     image_rgb,
                                     kind = 'avg'))

                plt.subplot(2,2,4)      # Region growing
                plt.imshow(label2rgb(growed_segments,
                                     image_rgb,
                                     kind = 'avg'))

                plt.show()

            # Store the final segmentation map
            self.segmentation_map = growed_segments

            return self.segmentation_map

        raise ValueError("Segmentation model not valid. Please use: 'slic'.")