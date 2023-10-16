import cv2 as cv
import imutils
import numpy as np


class PlateDetector:
    """
    A class to detect plates in an image.

    Attributes:
        img (array): The input image.
        plate_coords (array): Coordinates of the found plate.
        plate_mask (array): Boolean mask representing the plate.
        out_img_mask (array): Image with the plate contour drawn.
        out_img_contour (array): Image with masked background.

    Methods:
        __call__(img): Returns the detected plate for the given image.
        detect_plate(img): Detects the plate in the given image and returns its coordinates and mask.
        save_result(path, type="contour"): Saves the processed image to the specified path based on the type.
    """

    def __init__(self, ):
        self.img = None             # Input image
        self.plate_coords = None    # Coordinates of the found plate
        self.plate_mask = None      # Boolean mask representing the plate
        self.out_img_mask = None    # Image with the plate contour drawed
        self.out_img_contour = None # Image with masked background

    def __call__(self, img):
        return self.detect_plate(img)

    def detect_plate(self, img):
        """
        Detects the plate in the given image and returns its coordinates and mask.

        Args
            img (array): The input image.

        Returns:
            tuple: A tuple containing the plate coordinates and mask.
        """

        self.img = img

        # Convert to Grayscale
        gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        # Apply adaptive threshold with gaussian size 51x51
        thresh_gray = cv.adaptiveThreshold(
            gray_img,
            255,
            adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            thresholdType=cv.THRESH_BINARY,
            blockSize=51,
            C=0
        )

        # Find connected components (clusters)
        nlabel, labels, stats, centroids = cv.connectedComponentsWithStats(thresh_gray, connectivity=8)

        # Find second largest cluster (the cluster is the background):
        max_size = np.max(stats[1:, cv.CC_STAT_AREA])
        max_size_idx = np.where(stats[:, cv.CC_STAT_AREA] == max_size)[0][0]

        mask = np.zeros_like(thresh_gray)

        # Draw the cluster on mask
        mask[labels == max_size_idx] = 255

        # Use "open" morphological operation for removing some artifacts
        mask = cv.morphologyEx(
            mask,
            cv.MORPH_OPEN,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        )

        # Fill the plate with white pixels
        cv.floodFill(
            mask,
            None,
            tuple(centroids[max_size_idx].astype(int)),
            newVal=255,
            loDiff=1,
            upDiff=1
        )

        # Find contours, and get the contour with maximum area
        cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)

        # Store plate coordinates
        self.plate_coords = c

        # Draw contours with maximum size on the original image
        self.out_img_contour = self.img.copy()
        cv.drawContours(self.out_img_contour, [c], -1, (0, 255, 0), 2)

        # Draw contours with maximum size on new mask
        mask2 = np.zeros_like(mask)
        cv.drawContours(mask2, [c], -1, 255, -1)
        
        # Apply the mask to the original image
        self.out_img_mask = self.img.copy()
        self.out_img_mask[(mask2 == 0)] = 0

        # Get boolean mask
        self.plate_mask = np.ones(mask2.shape, dtype=bool)
        self.plate_mask[(mask2 == 0)] = False
        
        return self.plate_coords, self.plate_mask
    
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
            if self.out_img_contour is not None:
                cv.imwrite(path, self.out_img_contour)
            else:
                raise ValueError("No result to save. Ensure you've run the detect_plate method first.")
        
        elif type == "mask":
            if self.out_img_mask is not None:
                cv.imwrite(path, self.out_img_mask)
            else:
                raise ValueError("No result to save. Ensure you've run the detect_plate_and_mask method first.")
        
        else:
            raise ValueError("No valid type.")
        