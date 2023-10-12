import sys

import cv2
import imutils
import numpy as np

from utils.read_write import load_image


class PlateDetector:
    """
    A class to detect plates in an image.
    
    Attributes:
        mask (numpy.ndarray): A mask used for image processing.
        result_draw (numpy.ndarray): Result image with the drawn contour.
        result_mask (numpy.ndarray): Result image with masked background.
    """

    def __init__(self):
        """Initializes with default attributes."""
        self.mask = None    
        self.result_draw = None     # Result image with the drawed contour
        self.result_mask = None     # Result image with masked background
        self.plate_coords = None    # Coordinates of the found plate

    def __call__(self, image, scale=0.5):
        self.scale = scale
        return self.detect_plate(image)

    def detect_plate_and_mask(self, image, scale=0.5):
        """
        Detects the plate in the image and returns the image with masked background.
        
        Args:
            image (numpy.ndarray): The input image.
            scale (int): Reduction percentage of the image.
        
        Returns:
            numpy.ndarray: The processed image with masked background.
        """

        # Resize the image
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        image = cv2.resize(image, (width, height))

        # Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive threshold with gaussian size 51x51
        thresh_gray = cv2.adaptiveThreshold(
            gray, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            thresholdType=cv2.THRESH_BINARY, blockSize=51, C=0
        )

        # Find connected components (clusters)
        nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_gray, connectivity=8)

        # Find second largest cluster (the cluster is the background):
        max_size = np.max(stats[1:, cv2.CC_STAT_AREA])
        max_size_idx = np.where(stats[:, cv2.CC_STAT_AREA] == max_size)[0][0]

        self.mask = np.zeros_like(thresh_gray)

        # Draw the cluster on mask
        self.mask[labels == max_size_idx] = 255

        # Use "open" morphological operation for removing some artifacts
        self.mask = cv2.morphologyEx(
            self.mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        )

        # Fill the plate with white pixels
        cv2.floodFill(self.mask, None, tuple(centroids[max_size_idx].astype(int)), newVal=255, loDiff=1, upDiff=1)

        # Find contours, and get the contour with maximum area
        cnts = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        self.plate_coords = c

        # Draw contours with maximum size on new mask
        mask2 = np.zeros_like(self.mask)
        cv2.drawContours(mask2, [c], -1, 255, -1)
        
        # Apply the mask to the original image
        self.result_mask = image.copy()
        self.result_mask[(mask2 == 0)] = 0

        # Get boolean mask
        plate_mask = np.ones(mask2.shape, dtype=bool)
        plate_mask[(mask2 == 0)] = False
        self.plate_mask = plate_mask
        
        return self.result_mask, self.plate_mask
    
    def detect_plate(self, image, scale=0.5):
        """
        Detects the plate in the image and returns the image with the plate contour.
        
        Args:
            image (numpy.ndarray): The input image.
        
        Returns:
            numpy.ndarray: The processed image with the plate contour.
        """

        # Resize the image
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        image = cv2.resize(image, (width, height))

        # Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive threshold with gaussian size 51x51
        thresh_gray = cv2.adaptiveThreshold(
            gray, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            thresholdType=cv2.THRESH_BINARY, blockSize=51, C=0
        )

        # Find connected components (clusters)
        nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_gray, connectivity=8)

        # Find second largest cluster (the cluster is the background):
        max_size = np.max(stats[1:, cv2.CC_STAT_AREA])
        max_size_idx = np.where(stats[:, cv2.CC_STAT_AREA] == max_size)[0][0]

        # Find contours, and get the contour with maximum area
        cnts = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        self.plate_coords = c

        # Draw contours with maximum size on the original image
        self.result_draw = image.copy()
        cv2.drawContours(self.result_draw, [c], -1, (0, 255, 0), 2)

        return self.result_draw, self.plate_coords
    
    def save_result(self, path, type="contour"):
        """
        Saves the processed image to the specified path based on the type.
        
        Args:
            path (str): The path to save the image.
            type (str, optional): The type of result to save. Either "contour" or "mask". (default: "contour").
        
        Raises:
            ValueError: If no result to save or an invalid type is provided.
        """

        if type == "contour":
            if self.result_draw is not None:
                cv2.imwrite(path, self.result_draw)
            else:
                raise ValueError("No result to save. Ensure you've run the detect_plate method first.")
        
        elif type == "mask":
            if self.result_mask is not None:
                cv2.imwrite(path, self.result_draw)
            else:
                raise ValueError("No result to save. Ensure you've run the detect_plate_and_mask method first.")
        
        else:
            raise ValueError("No valid type.")
        


if __name__ == "__main__":

    # Load sample image
    path = "test/test_dish_3.png"
    image = load_image(path, 120000)
    
    # Init plate detector
    plate_detector = PlateDetector()

    # Detect plate
    image_plate, plate_coords = plate_detector.detect_plate(image, scale=0.5)
    image_plate_masked, plate_coords = plate_detector.detect_plate_and_mask(image, scale=0.5)

    # Show result
    cv2.imshow("Contoured image", image_plate)
    cv2.imshow("Masked image", image_plate_masked)
    k = cv2.waitKey(0)

    # Save output image
    plate_detector.save_result("test/detected_plate.jpg", type="contour")