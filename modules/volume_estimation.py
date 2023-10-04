import cv2
import numpy as np
from scipy.optimize import least_squares

from utils.calibration import salient_points_matching, salient_orb_matching, error_function
from utils.read_write import load_image
from modules.plate_detection import PlateDetector

class VolumeEstimator:

    def __init__(self, img1, img2, border1, border2, reference_card_dims=None, segmentation_map=None):
        self.img1 = img1
        self.img2 = img2
        self.border1 = border1 # Plate coords in the first image
        self.border2 = border2 # Plate coords in the second image
        self.reference_card_dims = reference_card_dims
        self.segmentation_map = segmentation_map

    def extrinsic_calibration(self):
        # Match salient points.
        img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY) 
        img_matches, points1, points2 = salient_orb_matching(img1_gray, img2_gray)

        # Display or save results.
        cv2.imshow('Matches', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Get candidate pose with RANSAC.
        H_init, inliers = cv2.findHomography(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        print("Homography Matrix:")
        print(H_init)
        print("Inliers mask:")
        print(inliers)

        # Flatten the initial H matrix to a vector.
        H_init_vec = H_init.flatten()

        # Run optimization using Levenberg-Marquardt algorithm.
        result = least_squares(
            error_function, H_init_vec, args=(points1, points2, border1, border2),
            method='lm'
        )

        # Retrieve the optimized H vector.
        H_optimized = result.x.reshape(3, 3)

        print("Optmizied Homography Matrix:")
        print(H_optimized)


if __name__ == "__main__":
    
    img1 = load_image('test/img1.jpg', max_size=120000)
    img2 = load_image('test/img1.jpg', max_size=120000)

    plate_detector = PlateDetector()

    _, border1 = plate_detector(img1, scale=1.0)
    _, border2 = plate_detector(img2, scale=1.0)

    volume_estimator = VolumeEstimator(img1, img2, border1, border2)
    volume_estimator.extrinsic_calibration()
