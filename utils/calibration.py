import random
import cv2 as cv
import numpy as np


def salient_points_matching_orb(image1, image2, n_keypoints=500, knn_matches_ratio=0.75):
    """
    Matches salient points between two images using ORB descriptor and k-NN matching.

    Args:
        image1 (numpy.ndarray): First input image.
        image2 (numpy.ndarray): Second input image.
        n_keypoints (int, optional): The maximum number of keypoints to generate. Defaults to 500.
        knn_matches_ratio (float, optional): Ratio used to identify good matches. Defaults to 0.75.

    Returns:
        numpy.ndarray: Image containing matched keypoints.
        list: List of point coordinates from image1.
        list: List of point coordinates from image2.
    """
    # Initialize ORB detector
    orb = cv.ORB_create(n_keypoints)
    
    # Find the keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
    
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6, 
                        key_size=12,    
                        multi_probe_level=1)
    search_params = dict(checks=50)
    
    # Apply FLANN based matcher with k=2
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Need to draw only good matches, so create a mask
    matches_mask = [[0,0] for i in range(len(matches))]
    good_matches = []

    # Ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < knn_matches_ratio * n.distance:
            matches_mask[i] = [1,0]
            good_matches.append((m, n))

    
    draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=(255,0,0),
                       matchesMask=matches_mask,
                       flags=0)
    
    img_matches = cv.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None, **draw_params)

    # Extract the coordinates of matched points
    points1 = np.float32([keypoints1[m.queryIdx].pt for (m, n) in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for (m, n) in good_matches]).reshape(-1, 1, 2)

    return img_matches, points1, points2, good_matches


def salient_points_matching_surf(image1, image2, min_hessian=400, knn_matches_ratio=0.75):
    """
    Matches salient points between two images using SURF descriptor and k-NN matching.

    Args:
        image1 (numpy.ndarray): First input image.
        image2 (numpy.ndarray): Second input image.
        min_hessian (float, optional): Hessian Threshold for SURF. Defaults to 400.
        knn_matches_ratio (float, optional): Ratio used to identify good matches. Defaults to 0.75.

    Returns:
        numpy.ndarray: Image containing matched keypoints.
        list: List of point coordinates from image1.
        list: List of point coordinates from image2.
    """
    # Initialize SURF detector
    surf = cv.xfeatures2d.SURF_create(hessianThreshold=min_hessian)
    
    # Find the keypoints and descriptors with SURF
    keypoints1, descriptors1 = surf.detectAndCompute(image1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(image2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # Apply FLANN based matcher with k=2
    flann = cvFlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Need to draw only good matches, so create a mask
    matches_mask = [[0,0] for _ in range(len(matches))]
    good_matches = []

    # Ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < knn_matches_ratio * n.distance:
            matches_mask[i] = [1,0]
            good_matches.append((m, n))
    
    draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=(255,0,0),
                       matchesMask=matches_mask,
                       flags=0)
    
    img_matches = cv.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None, **draw_params)
    

    # Extract the coordinates of matched points
    points1 = np.float32([keypoints1[m.queryIdx].pt for (m, n) in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for (m, n) in good_matches]).reshape(-1, 1, 2)

    return img_matches, points1, points2


def random_sample(matches, sample_size=5):
    return random.sample(matches, sample_size)

def generate_models(points1, points2):
    # Note: points1 and points2 should be corresponding points from the sampled matches
    E, mask = cv.findEssentialMat(points1, points2, method=cv.RANSAC, prob=0.999, threshold=1.0)
    return E, mask

def evaluate_models(generated_models, matches):
    # Implement model evaluation
    pass

def refine_model(best_model, inlier_matches):
    # Implement Levenberg-Marquardt optimization
    pass

def create_point_cloud(refined_model, inlier_matches):
    # Implement point cloud creation
    pass

def extract_relative_pose():
    pass

def error_function(H_vec, points1, points2, border1, border2):
    """
    Compute the total error due to mismatches in feature points and plate borders
    after applying a homography transformation.

    The error is computed as the sum of squared differences between the
    projected points using the homography and the actual points in the
    second image. The total error is the sum of the feature matching error
    and the plate border matching error.

    Args:
    H_vec (numpy.ndarray):      A 1D array of shape (9,) representing the flattened
                                3x3 homography matrix.
    points1 (numpy.ndarray):    An array of shape (N, 2) representing feature points
                                in the first image.
    points2 (numpy.ndarray):    An array of shape (N, 2) representing corresponding
                                feature points in the second image.
    border1 (numpy.ndarray):    An array of shape (M, 2) representing points along
                                the plate border in the first image.
    border2 (numpy.ndarray):    An array of shape (M, 2) representing points along
                                the plate border in the second image.

    Returns:
    float:  The total error, computed as the sum of squared differences between
            the projected points using the estimated homography matrix and the
            actual points in the second image, considering both feature points and
            plate border points.
    """
    
    H = H_vec.reshape(3, 3)
    
    # Ensure arrays are float32 type and have correct shape
    border1 = np.asarray(border1).reshape(-1, 2).astype(np.float32)
    border2 = np.asarray(border2).reshape(-1, 2).astype(np.float32)
    points1 = np.asarray(points1).reshape(-1, 2).astype(np.float32)
    points2 = np.asarray(points2).reshape(-1, 2).astype(np.float32)
    
    # Compute residuals for feature points
    projected_points = cv.perspectiveTransform(points1.reshape(-1, 1, 2), H)
    residuals_points = (projected_points - points2.reshape(-1, 1, 2)).flatten()
    
    # Compute residuals for border points
    projected_border = cv.perspectiveTransform(border1.reshape(-1, 1, 2), H)
    residuals_border = (projected_border - border2.reshape(-1, 1, 2)).flatten()
    
    # Concatenate residuals
    residuals = np.concatenate((residuals_points, residuals_border))
    
    return residuals

