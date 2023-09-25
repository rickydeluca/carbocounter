import numpy as np
import cv2
import imutils

from utils import load_image

class PlateDetector:
    """
    A class to detect plates in an image.
    
    Attributes:
        mask (numpy.ndarray): A mask used for image processing.
        result_contour (numpy.ndarray): Result image with the drawn contour.
        result_mask (numpy.ndarray): Result image with masked background.
    """

    def __init__(self):
        """Initializes with default attributes."""
        self.mask = None    
        self.result_contour = None  # Result image with the drawed contour
        self.result_mask = None     # Result image with masked background

    def detect_plate_and_mask(self, image):
        """
        Detects the plate in the image and returns the image with masked background.
        
        Args:
            image (numpy.ndarray): The input image.
        
        Returns:
            numpy.ndarray: The processed image with masked background.
        """

        # Resize the image
        scale_percent = 50.5
        width = int(image.shape[1] * scale_percent)
        height = int(image.shape[0] * scale_percent)
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

        # Draw contours with maximum size on new mask
        mask2 = np.zeros_like(self.mask)
        cv2.drawContours(mask2, [c], -1, 255, -1)
        
        # Apply the mask to the original image
        self.result_mask = image.copy()
        self.result_mask[(mask2 == 0)] = 0

        return self.result_mask
    
    def detect_plate(self, image):
        """
        Detects the plate in the image and returns the image with the plate contour.
        
        Args:
            image (numpy.ndarray): The input image.
        
        Returns:
            numpy.ndarray: The processed image with the plate contour.
        """

        # Resize the image
        scale = 0.5
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

        # Draw contours with maximum size on the original image
        self.result_contour = image.copy()
        cv2.drawContours(self.result_contour, [c], -1, (0, 255, 0), 2)

        return self.result_contour
    
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
            if self.result_contour is not None:
                cv2.imwrite(path, self.result)
            else:
                raise ValueError("No result to save. Ensure you've run the detect_plate method first.")
        
        elif type == "mask":
            if self.result_mask is not None:
                cv2.imwrite(path, self.result)
            else:
                raise ValueError("No result to save. Ensure you've run the detect_plate_and_mask method first.")
        
        else:
            raise ValueError("No valid type.")
        


if __name__ == "__main__":

    # Load sample image
    path = "data/test_dish.jpg"
    image = load_image(path)
    
    # Init plate detector
    plate_detector = PlateDetector()

    # Detect plate
    result_contour = plate_detector.detect_plate(image)
    result_mask = plate_detector.detect_plate_and_mask(image)

    # Show result
    cv2.imshow("Contoured image", result_contour)
    cv2.imshow("Masked image", result_mask)
    k = cv2.waitKey(0)