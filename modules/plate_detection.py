import math
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

from skimage import transform
from skimage import data, io
from skimage.color import rgb2gray, gray2rgb
from skimage.util import img_as_float, img_as_ubyte
from skimage.feature import canny
from skimage.filters import sobel
from skimage.transform import hough_ellipse, hough_circle
from skimage.draw import ellipse_perimeter


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

    def __init__(self):
        self.img = None             # Input image
        self.plate_coords = None    # Coordinates of the found plate
        self.plate_mask = None      # Boolean mask representing the plate
        self.out_img_mask = None    # Image with the plate contour drawed
        self.out_img_contour = None # Image with masked background


    def __call__(self, image, display=False):
        self.display = display
        self.image = image
        return self.detect_plate_circular()
    

    def detect_plate_circular(self):
        """
        Detects the plate in the given image and returns its coordinates and mask.

        Args
            img (array): The input image.

        Returns:
            tuple: A tuple containing the plate coordinates and mask.
        """

        # Compute average gradient magnitude using Sobel
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        average_magnitude = np.mean(grad_magnitude)

        # Detect circles
        low_thresh = 0.4 * average_magnitude
        high_thresh = 2.0 * average_magnitude 
        smoothed_image = cv.GaussianBlur(gray_image, (0, 0), 2)
        edges = cv.Canny(smoothed_image, low_thresh, high_thresh)
        smoothed_edges = cv.GaussianBlur(edges, (0,0), 2)

        if self.display:
            cv.imshow("Edges", edges)
            cv.waitKey()
            cv.destroyAllWindows()
        
        detected_circles = cv.HoughCircles(
            smoothed_image,  
            method=cv.HOUGH_GRADIENT_ALT,
            dp=1.5,
            minDist=20,
            param1=high_thresh,
            param2=0.9,
            minRadius=30, maxRadius=0) 
        
        if detected_circles is not None: 
            
            plate_mask = np.ones(gray_image.shape, dtype=bool)
            contoured_image = self.image.copy()
            masked_image = self.image.copy()

            # Convert the circle parameters a, b and r to integers
            detected_circles = np.uint16(np.around(detected_circles)) 

            # Find the circle with the largest radius
            best_circle = max(detected_circles[0, :], key=lambda x: x[2])
            a, b, r = best_circle[0], best_circle[1], best_circle[2]

            # Get plate coords
            _mask = np.zeros_like(gray_image)
            cv.circle(_mask, (a, b), r, (255,255,255), 2)
            contours, _ = cv.findContours(_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            plate_coords = max(contours, key=cv.contourArea)
            
            # Get contoured image
            cv.drawContours(contoured_image, [plate_coords], -1, (0, 255, 0), 2)
            
            # Get masked image
            _mask = np.zeros_like(_mask)
            cv.drawContours(_mask, [plate_coords], -1, 255, -1)

            if self.display:
                cv.imshow("Mask", _mask)
                cv.waitKey()
                cv.destroyAllWindows()

            masked_image[(_mask == 0)] = 0

            # Get boolean mask
            plate_mask[(_mask == 0)] = False

        else:
            print("No detected circle. Trying with conncected components.")
            return self.detect_plate_conn_comp()


        self.plate_mask = plate_mask
        self.plate_coords = plate_coords
        self.out_img_contour = contoured_image
        self.out_img_mask = masked_image

        if self.display:
            cv.imshow("Plate contour", self.out_img_contour)
            cv.waitKey(0)
        
            cv.imshow("Masked image", self.out_img_mask)
            cv.waitKey(0)

            cv.destroyAllWindows()

        return self.plate_coords, self.plate_mask
    

    def detect_plate_conn_comp(self):
        """
        Detects the plate in the given image and returns its coordinates and mask.

        Args
            img (array): The input image.

        Returns:
            tuple: A tuple containing the plate coordinates and mask.
        """
        
        # Convert to Grayscale
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)


        # Apply adaptive threshold with gaussian size 51x51
        edges = cv.adaptiveThreshold(
            gray_image,
            255,
            adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            thresholdType=cv.THRESH_BINARY,
            blockSize=101,
            C=0
        )

        if self.display:
            cv.imshow("Edges", edges)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
        # Find connected components (clusters)
        nlabel, labels, stats, centroids = cv.connectedComponentsWithStats(edges, connectivity=8)

        # Find second largest cluster (the first is the background):
        max_size = np.max(stats[1:, cv.CC_STAT_AREA])
        max_size_idx = np.where(stats[:, cv.CC_STAT_AREA] == max_size)[0][0]

        mask = np.zeros_like(edges)

        # Draw the cluster on mask
        mask[labels == max_size_idx] = 255

        # Close morphological operation
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

        # Fill the plate with white pixels
        cv.floodFill(
            mask,
            None,
            tuple(centroids[max_size_idx].astype(int)),
            newVal=255,
            loDiff=1,
            upDiff=1
        )

        # Find contours
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Get the contour with maximum area
        self.plate_coords = max(contours, key=cv.contourArea)

        # Draw contours with maximum size on the original image
        self.out_img_contour = self.image.copy()
        cv.drawContours(self.out_img_contour, [self.plate_coords], -1, (0, 255, 0), 2)

        # Draw contours with maximum size on new mask
        mask2 = np.zeros_like(mask)
        cv.drawContours(mask2, [self.plate_coords], -1, 255, -1)

        if self.display:
            cv.imshow("Mask", mask2)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
        # Apply the mask to the original image
        self.out_img_mask = self.image.copy()
        self.out_img_mask[(mask2 == 0)] = 0

        # Get boolean mask
        self.plate_mask = np.ones(mask2.shape, dtype=bool)
        self.plate_mask[(mask2 == 0)] = False

        if self.display:
            cv.imshow("Plate contour", self.out_img_contour)
            cv.waitKey(0)
        
            cv.imshow("Masked image", self.out_img_mask)
            cv.waitKey(0)

            cv.destroyAllWindows()

        return self.plate_coords, self.plate_mask