import json
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from plate_detection import PlateDetector
from utils import load_image


class FoodSegmenter:
    """
    A class to segment food in an image.
    """

    def __init__(self):
        """Initializes with default attributes."""
        self.image = None
        self.cropped_image = None   
        self.ellipse_coordinates = None

    def __call__(self, image, ellipse_coordinates):
        self.image = image
        self.ellipse_coordinates = ellipse_coordinates

        return self.segment_food()
    
    def crop_image(self):
        # Get the border coordinates.
        x_min = self.ellipse_coordinates[:, 0].min()
        x_max = self.ellipse_coordinates[:, 0].max()
        y_min = self.ellipse_coordinates[:, 1].min()
        y_max = self.ellipse_coordinates[:, 1].max()

        # Crop the images wrt the coordinates.
        self.cropped_image = self.image[y_min:y_max+1, x_min:x_max+1]
        
        # Adjust the ellipse coordinates to be relative to the cropped image.
        relative_ellipse_coordinates = self.ellipse_coordinates.copy()
        relative_ellipse_coordinates[:, 0] -= x_min
        relative_ellipse_coordinates[:, 1] -= y_min
        
        return self.cropped_image, relative_ellipse_coordinates
    

    def merge_small_regions(self, img, regions_coords, area_threshold=500):
        height, width = img.shape[:2]
        
        # 1. Calculate Region Areas and Colors
        region_areas = {}
        region_colors = {}
        for region, pixels in regions_coords.items():
            region_areas[region] = len(pixels)
            avg_color = np.mean([img[y, x] for x, y in pixels], axis=0)
            region_colors[region] = avg_color

        # 2. Identify Small Regions
        small_regions = [region for region, area in region_areas.items() if area < area_threshold]

        # Utility function to get neighboring regions
        def get_neighbors(r, regions_coords):
            neighbors = set()
            for x, y in regions_coords[r]:
                if y > 0 and segmentation[y-1, x] in regions_coords: neighbors.add(segmentation[y-1, x])
                if y < height-1 and segmentation[y+1, x] in regions_coords: neighbors.add(segmentation[y+1, x])
                if x > 0 and segmentation[y, x-1] in regions_coords: neighbors.add(segmentation[y, x-1])
                if x < width-1 and segmentation[y, x+1] in regions_coords: neighbors.add(segmentation[y, x+1])
            neighbors.discard(r)  # Remove the region itself
            return list(neighbors)

        # 3. Merge Small Regions
        segmentation = np.zeros(img.shape[:2], dtype=np.uint8)
        for region, pixels in regions_coords.items():
            for x, y in pixels:
                segmentation[y, x] = region

        for small_region in small_regions:
            if small_region not in regions_coords:  # Check if the small region is still present
                continue
            
            neighbors = get_neighbors(small_region, regions_coords)
            if not neighbors:
                continue

            # Sort neighbors based on color difference
            color_differences = [np.linalg.norm(region_colors[small_region] - region_colors[neighbor])
                                for neighbor in neighbors]
            sorted_neighbors = [x for _, x in sorted(zip(color_differences, neighbors))]

            # Find the closest available neighboring region in terms of color
            for neighbor in sorted_neighbors:
                if neighbor in regions_coords:
                    closest_neighbor = neighbor
                    break

            # Merge small region with closest neighbor
            regions_coords[closest_neighbor].extend(regions_coords[small_region])
            del regions_coords[small_region]

            for x, y in regions_coords[closest_neighbor]:
                segmentation[y, x] = closest_neighbor

        # 4. Recompute Contours
        contour_image = img.copy()
        for region, pixels in regions_coords.items():
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for x, y in pixels:
                mask[y, x] = 255
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

        return contour_image, regions_coords
    

    def filter_and_draw_contours(self, image, ellipse_coords, regions_dict):
        # Create the plate mask
        plate_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        plate_mask[ellipse_coords[:, 1], ellipse_coords[:, 0]] = 255
        plate_mask = cv2.fillPoly(plate_mask, [ellipse_coords], 255)

        # Filtered regions dictionary
        filtered_regions = {}

        for region, pixels in regions_dict.items():
            # Convert pixel list to a mask
            region_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for x, y in pixels:
                region_mask[y, x] = 255

            # Criteria 1: Check if more than 10% of the segment's area is outside the plate
            outside_plate = np.sum((region_mask == 255) & (plate_mask == 0))
            if outside_plate / np.sum(region_mask == 255) > 0.1:
                continue

            # Criteria 2: Check if segment shares borders with background for more than 10% of its contour's length
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour_length = cv2.arcLength(contours[0], True)
                if contour_length == 0:
                    continue
                
                edge_pixels = cv2.findNonZero(region_mask & cv2.Canny(plate_mask, 100, 200))
                if edge_pixels is not None and len(edge_pixels) / contour_length > 0.1:
                    continue
            
            # If passed both criteria, add to filtered regions and draw contour
            filtered_regions[region] = pixels
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

        return image, filtered_regions

    

    def region_growing(self, img, threshold=50):
        # Parameters
        height, width, channels = img.shape

        # Data structures
        processed = np.zeros((height, width), dtype=np.uint8)
        segmentation = np.zeros((height, width), dtype=np.uint8)
        
        all_pixels = [(x, y) for y in range(height) for x in range(width)]
        random.shuffle(all_pixels)

        region_coordinates = {}
        region_number = 0

        for seed in all_pixels:
            # If the seed has already been processed, continue to the next seed
            if processed[seed[1], seed[0]] == 1:
                continue

            region_number += 1
            region_pixels = []
            active_pixels = [seed]
            while active_pixels:
                current_pixel = active_pixels.pop(0)

                x, y = current_pixel
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue
                if processed[y, x] == 1:
                    continue

                processed[y, x] = 1
                segmentation[y, x] = region_number  # Assign region number instead of 255
                region_pixels.append((x, y))

                neighbors = [(x-1, y-1), (x, y-1), (x+1, y-1),
                            (x-1, y),                 (x+1, y),
                            (x-1, y+1), (x, y+1), (x+1, y+1)]

                for neighbor in neighbors:
                    nx, ny = neighbor
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if processed[ny, nx] == 1:
                        continue

                    color_diff = np.linalg.norm(img[y, x] - img[ny, nx])
                    if color_diff < threshold:
                        active_pixels.append(neighbor)

            region_coordinates[region_number] = region_pixels

        # Draw contours on the original image
        contour_image = img.copy()
        for region, pixels in region_coordinates.items():
            mask = np.zeros((height, width), dtype=np.uint8)
            for x, y in pixels:
                mask[y, x] = 255
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

        return contour_image, region_coordinates


    def segment_food(self):
        # Crop the image around the detected ellipse
        cropped_image, relative_plate_coords = self.crop_image()
        # np.savetxt("test/ellipse_points.txt", relative_plate_coords)

        # Convert to CIELAB color space
        lab_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2Lab)

        # Apply pyramidal mean shift filtering and convert back to BGR
        mean_shifted_image = cv2.pyrMeanShiftFiltering(lab_image, sp=30, sr=30)
        mean_shifted_image = cv2.cvtColor(mean_shifted_image, cv2.COLOR_Lab2BGR)
        cv2.imshow('Mean Shifted Image', mean_shifted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Apply region growing algorithm
        segmented_image, region_coordinates = self.region_growing(mean_shifted_image, threshold=100)
        cv2.imshow('Segmented Image', segmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Merge small regions
        merged_image, merged_region_coords = self.merge_small_regions(mean_shifted_image, region_coordinates, area_threshold=500)
        cv2.imshow('Merged Regions Image', merged_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # Filter regions
        out_image, filtered_region_coords = self.filter_and_draw_contours(merged_image, relative_plate_coords, merged_region_coords)
        cv2.imshow('Filtered Segmented Image', out_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return out_image, filtered_region_coords

    

if __name__ == "__main__":

    # Load sample image
    path = "test/test_dish_3.png"
    image = load_image(path)
    
    # Init modules
    plate_detector = PlateDetector()
    food_segmenter = FoodSegmenter()

    # Detect plate
    result_draw, plate_coords = plate_detector.detect_plate(image, scale=1.0)

    # Segment food
    segmented_image, region_coords = food_segmenter(result_draw, plate_coords)
    # cv2.imwrite("test/segmented_image.jpg", segmented_image)
    # with open('test/region_dict.txt', 'w') as of:
    #     of.write(json.dumps(region_coords))
