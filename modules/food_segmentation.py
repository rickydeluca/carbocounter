import cv2
import numpy as np
from plate_detection import PlateDetector

from utils.read_write import load_image
from utils.segmentation import *


class FoodSegmenter:
    """
    A class to segment food in an image.
    """

    def __init__(self):
        self.input_image = None
        self.mean_shifted_image = None
        self.segmented_image = None

    def __call__(self, image, plate_coords, show_process=False):
        return self.segment_food(image, plate_coords, show_process=show_process)
    
    def crop_image(self, image, coordinates):
        """
        Crop the "image" wrt the given plate "coordinates".
        
        Returns:
            cropped_image (np.array):           The cropped image.

            relative_coordinates (np.array):    The new coordinates rescaled to 
                                                match the cropped image.
        """
        # Get the border coordinates.
        x_min = coordinates[:, 0].min()
        x_max = coordinates[:, 0].max()
        y_min = coordinates[:, 1].min()
        y_max = coordinates[:, 1].max()

        # Crop the images wrt the coordinates.
        cropped_image = image[y_min:y_max+1, x_min:x_max+1]
        
        # Adjust the ellipse coordinates to be relative to the cropped image.
        relative_coordinates = coordinates.copy()
        relative_coordinates[:, 0] -= x_min
        relative_coordinates[:, 1] -= y_min

        # Save the results.
        self.cropped_image = cropped_image
        self.relative_plate_coords = relative_coordinates 
        
        return cropped_image, relative_coordinates

    def segment_food(self, image, plate_coords, show_process=False):

        cv2.imshow("Original Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Crop image wrt the plate coordinates.
        cropped_image, relative_plate_coords = self.crop_image(image, plate_coords)

        if show_process:
            cv2.imshow("Cropped Image", cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Apply pyramidal mean shift fitlering.
        sth = 10
        cth = 5
        mean_shifted_image = pyramidal_mean_shift_filtering(cropped_image, sth=sth, cth=cth, gaussian_levels=4, max_iterations=10, cast_back=False)
        
        if show_process:
            cv2.imshow("Mean Shifted Image", mean_shifted_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Apply region growing.
        rth = 4
        regions = region_growing(mean_shifted_image, rth)

        if show_process:
            cv2.imshow("Segmented Image", regions)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Merge regions.
        merged_regions = merge_small_regions(mean_shifted_image, regions, ath=3000, cth=5000)

        if show_process:
            cv2.imshow("Merged Image", merged_regions)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Filter regions.
        filtered_image, filtered_regions = filter_regions(merged_regions, relative_plate_coords)

        if show_process:
            cv2.imshow("Filtered Image", filtered_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return filtered_regions


if __name__ == "__main__":

    # Load sample image.
    path = "test/test_dish_3.png"
    image = load_image(path, max_size=120000)
    
    # Init modules.
    food_segmenter = FoodSegmenter()
    plate_detector = PlateDetector()

    # Segment food.
    image, plate_coords = plate_detector.detect_plate(image, scale=1.0)
    segmentation_map = food_segmenter(image, plate_coords, show_process=True)

    # Unique labels in the segmentation map.
    print("segmentation_shape:", segmentation_map.shape)


