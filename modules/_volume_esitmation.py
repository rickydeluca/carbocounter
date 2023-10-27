import math
import cv2 as cv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from functools import reduce
from scipy.spatial import Delaunay
from sklearn.datasets import make_regression
from sklearn.linear_model import RANSACRegressor
from utils.reconstruction3d import infer_depth_map

class VolumeEstimator:

    def __init__(self, idx2class=None, inference_dataset='nyu'):
        self.idx2class = idx2class
        self.inference_dataset = inference_dataset


    def __call__(self, img, segmentation_map, display=False):
        self.img = img
        self.segmentation_map = segmentation_map
        self.display = display
        return self.estimate_volume()


    def segment_point_cloud(self):
        # Get dimensions
        height, width = self.segmentation_map.shape

        # Label each (x,y) pixel coordinate
        labels = np.zeros((height, width), dtype=int)
        for i in range(height):
            for j in range(width):
                labels[i, j] = self.segmentation_map[i, j]

        # Map labels to Point Cloud
        pcd_labels = np.zeros((len(self.pcd.points),), dtype=int)
        for i in range(height):
            for j in range(width):
                point_index = i * width + j  # Index of the point in the point cloud
                pcd_labels[point_index] = labels[i, j]

        # Get the segmented point clouds
        segmented_pcds = {}

        # Iterate through labels and points to create segmented point clouds
        for unique_label in np.unique(pcd_labels):
            indices = np.where(pcd_labels == unique_label)[0]
            segmented_pcd = self.pcd.select_by_index(indices)

            segmented_pcds[unique_label] = segmented_pcd

        return segmented_pcds


    def get_segment_volumes(self):
        volumes = {}
        hull_meshes = []
        conversion_factor = 1_000_000_000   # cubic meters to milliliters

        # DEBUG
        print("segment labels:", self.segmented_pcds.keys(), end="\n\n")

        for label, segmented_pcd in self.segmented_pcds.items():
            # Ignore the background
            if label == 0:
                volumes['background'] = 0
                continue

            # Retrieve the food class name
            predicted_class = self.idx2class[label]
            hull, _ = segmented_pcd.compute_convex_hull()
            volumes[predicted_class] = hull.get_volume() * conversion_factor
            hull_meshes.append(hull)
        
        return volumes, hull_meshes


    def smooth_mask(self, mask, kernel_size=3):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Closing
        closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        # Opening
        opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)
        return opened


    def segment_depth_map(self, depth_map):
        # Create background mask
        background_mask = self.segmentation_map == 0

        # Get the max depth value for setting the background
        max_depth = np.max(depth_map)

        # Create a mask for the current food class
        food_mask = self.segmentation_map != 0
            
        # Smooth the mask's contours
        food_mask_smoothed = self.smooth_mask(np.uint8(food_mask))

        # Compute the distance transform on the smoothed food mask
        distance_transform = cv.distanceTransform(food_mask_smoothed, cv.DIST_L2, 3)

        # Normalize the distance transform to range [0, 1]
        max_distance = np.max(distance_transform)
        normalized_distance = distance_transform / max_distance

        # Interpolate between the original depth values and the max depth
        interpolated_depth = depth_map * normalized_distance + max_depth * (1 - normalized_distance)

        # Apply the interpolated depth to the current food region
        depth_map[food_mask] = interpolated_depth[food_mask]

        # Set the background to the max depth
        depth_map[background_mask] = max_depth

        return depth_map
    
    
    def rescale_depth(self, depth_value, min_old, max_old, min_new, max_new):
        """
        Rescale depth_value from [min_old, max_old] to [min_new, max_new]
        """
        return min_new + ((depth_value - min_old) / (max_old - min_old)) * (max_new - min_new)


    def estimate_volume(self):

        # Get the depth map
        scaling_factor = 100    # meters to centimeters
        depth_map = infer_depth_map(self.img, inference_dataset=self.inference_dataset) * scaling_factor

        # Rescale the depth
        min_original_depth = np.min(depth_map)
        max_original_depth = np.max(depth_map)
        depth_map = np.vectorize(self.rescale_depth)(depth_map, min_original_depth, max_original_depth, 37, 40).astype(np.float32)

        # # DEBUG
        # print("min depth: ", np.min(depth_map))
        # print("max depth:", np.max(depth_map))
        # exit()
        
        # Remove plate and background and keep only the food
        depth_map = self.segment_depth_map(depth_map)

        if self.display:
            plt.imshow(depth_map, cmap='plasma')
            plt.show()

       # Convert numpy arrays to standard Open3D images
        depth_open3d_image = o3d.geometry.Image(depth_map)
        color_open3d_image = o3d.geometry.Image(cv.cvtColor(self.img, cv.COLOR_BGR2RGB))

        # Create RGB-D image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_open3d_image, 
            depth_open3d_image, 
            convert_rgb_to_intensity=False
        )

        # Create point cloud
        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        )

        # Flip it
        self.pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        if self.display:
            o3d.visualization.draw_geometries([self.pcd]) # Original

        # Segment the point cloud wrt the segmentation map
        self.segmented_pcds = self.segment_point_cloud()
        
        if self.display:
            for label in self.segmented_pcds:
                o3d.visualization.draw_geometries([self.segmented_pcds[label]])

        # Estimate the volumes of the segments
        self.segment_volumes, self.hull_meshes = self.get_segment_volumes()
        print("Segment volumes:", self.segment_volumes)

        return self.segment_volumes

    