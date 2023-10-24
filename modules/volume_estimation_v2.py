import math
import cv2 as cv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from functools import reduce
from scipy.spatial import Delaunay
from utils.reconstruction3d import infer_depth_map

class VolumeEstimator:

    def __init__(self, inference_dataset='nyu'):
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
        conversion_factor = 1_000_000   # cubic meters to milliliters

        for label, segmented_pcd in self.segmented_pcds.items():
            # Ignore the background
            if label == 0:
                volumes[label] = 0
                continue

            hull, _ = segmented_pcd.compute_convex_hull()
            volumes[label] = hull.get_volume() * conversion_factor
            hull_meshes.append(hull)
        
        return volumes, hull_meshes


    def segment_depth_map(self, depth_map):

        # Create background mask
        background_mask = self.segmentation_map == 0
        food_mask = self.segmentation_map != 0

        # We use the max depth of the food to create a new plane.
        # We need to do this because the depth map it isn't very accurate because
        # it was estimate using a neural network.
        max_food_depth = np.max(depth_map[food_mask])
        depth_map[background_mask] = max_food_depth

        return depth_map
    
    
    def normalize_depth_map(self, depth_map):
        normalized_depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        return normalized_depth_map


    def estimate_volume(self):

        # Get the depth map
        scaling_factor = 100
        depth_map = infer_depth_map(self.img, inference_dataset=self.inference_dataset) * scaling_factor

        # Remove plate and background and keep only the food
        # print("Labels in segmentation map: ", np.unique(self.segmentation_map))
        # print("Unique depths before segmentation: ", np.unique(depth_map))
        depth_map = self.segment_depth_map(depth_map)
        # print("Unique depths after segmentatino: ", np.unique(depth_map))

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
        self.segment_volumes = self.get_segment_volumes()
        print("Segment volumes:", self.segment_volumes)

        return self.segment_volumes

    