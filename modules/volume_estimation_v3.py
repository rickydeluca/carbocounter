import math
import cv2 as cv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from functools import reduce
from scipy.spatial import Delaunay
from sklearn.datasets import make_regression
from sklearn.linear_model import RANSACRegressor
from utils.reconstruction3d import infer_depth_map, rescale_depth, smooth_mask, distance_from_diameter

class VolumeEstimator:

    def __init__(self, idx2class=None, focal_length= None, plate_diameter=None, inference_dataset='nyu'):
        self.idx2class = idx2class
        self.focal_length = focal_length
        self.plate_diameter = plate_diameter
        self.inference_dataset = inference_dataset


    def __call__(self, img, segmentation_map, plate_coords, display=False):
        self.img = img
        self.segmentation_map = segmentation_map
        self.plate_coords = plate_coords
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


    def get_segment_volumes_poisson(self):
        volumes = {}
        mesh_list = []
        conversion_factor = 1_000_000_000   # cubic meters to milliliters

        for label, segmented_pcd in self.segmented_pcds.items():
            # Ignore the background
            if label == 0:
                volumes['background'] = 0
                continue

            # Retrieve the food class name
            predicted_class = self.idx2class[label]

            # Use Poisson Surface Reconstruction for mesh generation
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(segmented_pcd, depth=8)

            # Simplify mesh if necessary
            # mesh = mesh.simplify_quadric_decimation(target_number_of_triangles)

            try:
                volumes[predicted_class] = mesh.get_volume() * conversion_factor
            except:
                print("Invalid volume")
                continue

            mesh_list.append(mesh)
            
        return volumes, mesh_list


    def get_segment_volumes(self):
        volumes = {}
        hull_meshes = []
        conversion_factor = 1_000_000_000   # cubic meters to milliliters

        # DEBUG
        # print("segment labels:", self.segmented_pcds.keys(), end="\n\n")

        for label, segmented_pcd in self.segmented_pcds.items():
            # Ignore the background
            if label == 0:
                volumes['background'] = 0
                continue

            # Retrieve the food class name
            predicted_class = self.idx2class[label]
            hull, _ = segmented_pcd.compute_convex_hull()

            try:
                volumes[predicted_class] = hull.get_volume() * conversion_factor
            except:
                print("Invalid volume")
                continue

            hull_meshes.append(hull)
        
        return volumes, hull_meshes


    def segment_depth_map(self, depth_map):
        # Create background mask
        background_mask = self.segmentation_map == 0

        # Get the max depth value for setting the background
        max_depth = np.max(depth_map)
        
        # Create a mask for the current food class
        food_mask = self.segmentation_map != 0
            
        # Smooth the mask's contours
        food_mask_smoothed = smooth_mask(np.uint8(food_mask))

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
    

    def segment_depth_map_by_food(self, depth_map):
        # Create background mask
        background_mask = self.segmentation_map == 0

        # Get the max depth value for setting the background
        max_depth = np.max(depth_map)

        # Get all unique food classes
        food_classes = np.unique(self.segmentation_map)
        food_classes = food_classes[food_classes != 0]  # Remove background class

        for food_class in food_classes:
            # Create a mask for the current food class
            food_mask = self.segmentation_map == food_class

            # Smooth the mask's contours
            food_mask_smoothed = smooth_mask(np.uint8(food_mask))

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
    

    def estimate_volume(self):

        # Get the depth map
        depth_map = infer_depth_map(self.img, inference_dataset=self.inference_dataset)

        # Rescale the depth using the plate diameter as a reference
        (center_x, center_y), radius = cv.minEnclosingCircle(self.plate_coords)
        diameter_in_pixels = 2 * radius
        distance = distance_from_diameter(focal_length=self.focal_length, actual_diameter=self.plate_diameter, diameter_in_pixels=diameter_in_pixels)

        print("distance:", distance)

        min_original_depth = np.min(depth_map)
        max_original_depth = np.max(depth_map)
        depth_map = np.vectorize(rescale_depth)(depth_map, min_original_depth, max_original_depth, distance-3, distance).astype(np.float32)

        # DEBUG
        print("Plate diameter:", diameter_in_pixels)
        print("min depth: ", np.min(depth_map))
        print("max depth:", np.max(depth_map))

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

    