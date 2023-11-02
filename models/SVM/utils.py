import os
import cv2 as cv
import numpy as np

from skimage.feature import local_binary_pattern
from sklearn.cluster import AgglomerativeClustering


def color_agglomerative_clustering(data, max_num_clusters=1024):
    """
    Wrapper for the AgglomerativeClustering function as defined by scikit-learn library.
    """
    hierarchical_cluster = AgglomerativeClustering(n_clusters=max_num_clusters, affinity='euclidean', linkage='ward')
    labels = hierarchical_cluster.fit_predict(data)

    return labels


def get_dominant_colors(colors, num_clusters=1024):
    """
    Get the dominant colors ceneter after applying an agglomerative hierarchical clustering
    over a set of color values.
    """
    # Hierachical clustering.
    labels = color_agglomerative_clustering(colors, max_num_clusters=num_clusters)

    # Compute dominant colors.
    dominant_colors = np.array([colors[labels == i].mean(axis=0) for i in range(num_clusters)])

    return dominant_colors


def dominant_color_histogram(image, dominant_colors):
    """
    Compute the histogram of an image based on the given dominant colors.
    """
    # Flatten the image and find the closest dominant color for each pixel.
    pixels = image.reshape(-1, 3)
    closest_dominant_colors = np.argmin(np.linalg.norm(pixels[:, np.newaxis] - dominant_colors, axis=2), axis=1)
    
    # Compute the histogram.
    histogram, _ = np.histogram(closest_dominant_colors, bins=np.arange(len(dominant_colors) + 1))
    
    return histogram


def get_color_histogram(image, num_clusters=1024, sample_size=None):
    """
    Compute the color histogram of an image based on dominant colors.

    This function first extracts the BGR values from the input image, and then
    determines the dominant colors by sampling and clustering the colors. Finally,
    it computes and returns the color histogram of the image based on the dominant colors.

    Args:
        image (np.array):
            A 3D NumPy array representing the input image, where
            the dimensions represent [height, width, color_channels].
        num_clusters (int, optional):
            The number of clusters to form when finding dominant colors
            (default: 1024).
        sample_size (int, optional):
            The number of BGR values to sample when determining
            dominant colors. If None, all values are used (default: None).

    Returns:
        np.array:
            A color histogram of the input image based on the dominant colors.
    """
    
    # Filter out near-black pixels based on a threshold.
    threshold = 20
    mask = np.all(image > threshold, axis=2)
    filtered_image = image[mask]

    # Extract BGR values from the filtered image.
    bgr_values = filtered_image.reshape(-1, 3)
    sample_size = bgr_values.shape[0] if sample_size is None else sample_size # Use all colors if not specified
    rgb_sample = bgr_values[np.random.choice(bgr_values.shape[0], sample_size, replace=False)]

    # Get dominant colors.
    dominant_colors = get_dominant_colors(rgb_sample, num_clusters=num_clusters)

    # Compute the color histogram for the image.
    color_histogram = dominant_color_histogram(image, dominant_colors)

    return color_histogram


def get_color_histograms(images, num_clusters=1024, sample_size=None):
    """
    Same as `get_color_histogram()` but for multiple images at once.
    """
    # Extract BGR values.
    bgr_values = np.vstack([img.reshape(-1, 3) for img in images])
    sample_size = bgr_values.shape[0] if sample_size is None else sample_size # Use all colors if not specified
    rgb_sample = bgr_values[np.random.choice(bgr_values.shape[0], sample_size, replace=False)]

    # Get dominant colors.
    dominant_colors = get_dominant_colors(rgb_sample, num_clusters=num_clusters)

    # Compute the color histograms for each image.
    color_histograms = [dominant_color_histogram(img, dominant_colors) for img in images]

    return color_histograms


def multi_radius_lbp(image, radii, points_per_radius):
    """
    Compute the concatenated LBP histograms for multiple radii.
    """
    histograms = []
    
    for radius, n_points in zip(radii, points_per_radius):
        lbp_img = local_binary_pattern(cv.cvtColor(image, cv.COLOR_BGR2GRAY), n_points, radius, method="uniform")
        
        # Given the "uniform" method and n_points, the maximum number of unique patterns is: n_points*(n_points-1)+3
        max_bins = n_points*(n_points-1) + 3
        lbp_hist, _ = np.histogram(lbp_img.ravel(), bins=max_bins, range=(0, max_bins))
        
        histograms.append(lbp_hist)
    
    return np.concatenate(histograms)


def get_lbp_histogram(image, radii=[1,2,3], points_per_radius=[8,16,24], vector_size=256):
    """
    Compute the Local Binary Pattern (LBP) histogram of an image with multiple radii.

    Args:
        image (ndarray):
            Input image data.
        radii (list of int, optional):
            List of radii to be used for LBP computation (default: [1,2,3]).
        points_per_radius (list of int, optional):
            Number of points to be considered per radius in LBP computation. 
            (default: [8,16,24]).
        vector_size (int, optional):
            Size of the LBP histogram vector (defaults to 256).

    Returns:
        ndarray:
            Histogram of LBP values.
    """
    # Compute Multi radious LBP for the given image.
    lbp_image = multi_radius_lbp(image, radii, points_per_radius)

    # Compute LBP histogram.
    lbp_histogram = np.histogram(lbp_image.ravel(), bins=vector_size, range=(0, vector_size))[0]

    return lbp_histogram


def get_lbp_histograms(images, radii=[1,2,3], points_per_radius=[8,16,24], vector_size=256):
    """
    Same as `get_lbp_histogram()` but for multiple images at once.
    """
    # Compute Multi radious LBP for each image.
    lbp_images = [multi_radius_lbp(img, radii, points_per_radius) for img in images]

    # Compute LBP histograms.
    lbp_histograms = [np.histogram(lbp_img.ravel(), bins=vector_size, range=(0, vector_size))[0] for lbp_img in lbp_images]

    return lbp_histograms


def extract_features(image, num_clusters=1024, sample_size=None, radii=[1,2,3], points_per_radius=[8,16,24], lbp_vector_size=256):
    """
    Extract color and texture features from an image.

    Args:
        image (ndarray):
            Input image data.
        num_clusters (int, optional):
            Number of clusters to be used in color histogram computation (default: 1024).
        sample_size (int, optional):
            Size of the sample to be used in color histogram computation (default: None)
        radii (list of int, optional):
            List of radii to be used for LBP computation (default: [1,2,3]).
        points_per_radius (list of int, optional):
            Number of points to be considered per radius in LBP computation. 
            (default: [8,16,24]).
        lbp_vector_size (int, optional):
            Size of the LBP histogram vector (default: 256).

    Returns:
        ndarray:
            Concatenated feature vector containing color and texture features.
    """
    
    color_features = get_color_histogram(image, num_clusters=num_clusters, sample_size=sample_size)
    texture_features = get_lbp_histogram(image, radii=radii, points_per_radius=points_per_radius, vector_size=lbp_vector_size)
    return np.concatenate((color_features, texture_features))