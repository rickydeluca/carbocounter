import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from skimage.feature import local_binary_pattern
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


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


def color_histogram(image, dominant_colors):
    """
    Compute the histogram of an image based on the given dominant colors.
    """
    # Flatten the image and find the closest dominant color for each pixel.
    pixels = image.reshape(-1, 3)
    closest_dominant_colors = np.argmin(np.linalg.norm(pixels[:, np.newaxis] - dominant_colors, axis=2), axis=1)
    
    # Compute the histogram.
    histogram, _ = np.histogram(closest_dominant_colors, bins=np.arange(len(dominant_colors) + 1))
    
    return histogram


def get_color_histograms(images, num_clusters=1024, sample_size=None):
    # Extract BGR values.
    bgr_values = np.vstack([img.reshape(-1, 3) for img in images])
    sample_size = bgr_values.shape[0] if sample_size is None else sample_size # Use all colors if not specified
    rgb_sample = bgr_values[np.random.choice(bgr_values.shape[0], sample_size, replace=False)]

    # Get dominant colors.
    dominant_colors = get_dominant_colors(rgb_sample, num_clusters=num_clusters)

    # Compute the color histograms for each image.
    color_histograms = [color_histogram(img, dominant_colors) for img in images]

    return color_histograms


def multi_radius_lbp(image, radii, points_per_radius):
    """
    Compute the concatenated LBP histograms for multiple radii.
    """
    histograms = []
    
    for radius, n_points in zip(radii, points_per_radius):
        lbp_img = local_binary_pattern(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), n_points, radius, method="uniform")
        
        # Given the "uniform" method and n_points, the maximum number of unique patterns is: n_points*(n_points-1)+3
        max_bins = n_points*(n_points-1) + 3
        lbp_hist, _ = np.histogram(lbp_img.ravel(), bins=max_bins, range=(0, max_bins))
        
        histograms.append(lbp_hist)
    
    return np.concatenate(histograms)


def get_lbp_histograms(images, radii=[1,2,3], points_per_radius=[8,16,24], vector_size=256):
    # Compute Multi radious LBP for each image.
    lbp_images = [multi_radius_lbp(img, radii, points_per_radius) for img in images]

    # Compute LBP histograms.
    lbp_histograms = [np.histogram(lbp_img.ravel(), bins=vector_size, range=(0, vector_size))[0] for lbp_img in lbp_images]

    return lbp_histograms


def main():
    # Load all images.
    img_dir = "data/FoodSeg103/Images/img_dir/train"
    image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")][:3800]
    images = [cv2.imread(img_file) for img_file in image_files]

    # Resize them.
    processed_images = [cv2.resize(img, (32, 32)) for img in images]

    # Get color histograms.
    num_clusters = 1024
    sample_size = 10000
    color_histograms = get_color_histograms(processed_images, num_clusters=num_clusters, sample_size=sample_size)

    # Get texture histograms.
    radii = [1, 2, 3]
    points_per_radius = [8, 16, 24]
    lbp_histograms = get_lbp_histograms(processed_images, radii=radii, points_per_radius=points_per_radius)

    # Combine features.
    combined_features = [np.concatenate((color_histograms[i], lbp_histograms[i])) for i in range(len(images))]

    print(f"Feature vector shape: {combined_features[0].shape}")

if __name__ == "__main__":
    main()
