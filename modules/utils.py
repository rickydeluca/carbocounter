import cv2
import random
import numpy as np
from collections import deque


class Stack:
    def __init__(self):
        self.items = []

    def push(self, value):
        self.items.append(value)

    def pop(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def is_empty(self):
        return self.size() == 0

    def clear(self):
        self.items = []


def load_image(img_path, max_size):
    """
    Rescale the input image while preserving its aspect ratio such that its 
    total number of pixels is less than or equal to "max_size".
    
    Args:
        img_path (str): Path to the input image.
        max_size (int): The maximum number of pixels allowed in the output image.
    
    Returns:
        numpy.ndarray: The resized image.
    """
    
    # Read the image.
    img = cv2.imread(img_path)

    # Get the current dimensions.
    height, width, _ = img.shape
    
    # Calculate the current number of pixels.
    current_pixels = height * width
    
    # If current pixels are already less than N, return the image.
    if current_pixels <= max_size:
        return img

    # Calculate the scaling factor.
    scaling_factor = np.sqrt(max_size / current_pixels)
    
    # Calculate the new dimensions.
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    
    # Resize the image.
    resized_img = cv2.resize(img, (new_width, new_height))
    
    return resized_img


def lab_pixel_distance(LAB1, LAB2):
    """ Calculate the distance between two LAB color values. """
    delta = LAB1 - LAB2
    return np.sqrt(abs(delta[0]) + delta[1]**2 + delta[2]**2)


def lab_distance(img1, img2):
    """ Calculate the distance between two LAB images element-wise. """
    delta = img1 - img2
    return np.sqrt(np.abs(delta[..., 0]) + delta[..., 1]**2 + delta[..., 2]**2)


def mean_shift(pixel, image, sth, cth):
    """
    Perform mean shift operation for a given pixel in the image.

    Args:
        pixel (tuple):
            A tuple containing (x, y, LAB) where (x, y) are the pixel 
            coordinates and LAB is the color.
        image (ndarray):
            LAB image to perform mean shift on.
        sth (int):
            Spatial threshold for neighborhood definition.
        cth (float):
            Color threshold for neighborhood definition.

    Returns:
        ndarray:
            The new LAB color after mean shift operation.
    """

    x, y, LAB = pixel
    h, w = image.shape[:2]
    
    # Define spatial neighborhood window boundaries.
    x_min, x_max = max(0, x - sth), min(w, x + sth + 1)
    y_min, y_max = max(0, y - sth), min(h, y + sth + 1)
    
    # Extract neighborhood.
    neighborhood = image[y_min:y_max, x_min:x_max]
    
    # Compute distances.
    distances = np.sqrt(np.abs(neighborhood[..., 0] - LAB[0]) + 
                        (neighborhood[..., 1] - LAB[1])**2 + 
                        (neighborhood[..., 2] - LAB[2])**2)
    
    # Filter by color threshold.
    mask = distances <= cth
    if np.sum(mask) == 0:  # No valid neighbors.
        return LAB

    # Calculate mean LAB value within the filtered neighborhood.
    return np.mean(neighborhood[mask], axis=0)


def pyramidal_mean_shift_filtering(image, sth=5, cth=10, gaussian_levels=4, max_iterations=10, cast_back=False):
    """
    Apply pyramidal mean shift filtering on the given image.

    Args:
        image (ndarray):
            Input BGR image to be processed.
        sth (int):
            Spatial threshold for mean shift (default: 5).
        cth (float):
            Color threshold for mean shift operation (default: 10).
        gaussian_levels (int, optional):
            Number of levels in the Gaussian pyramid (default: 4).
        max_iterations (int, optional):
            Maximum number of iterations for mean shift (default: 10).
        cast_back (bool, optional):
            If True cast back the image to its original color space
            (default: False).

    Returns:
        ndarray:
            Filtered BGR image after applying pyramidal mean shift.
    """

    # Convert to CIELAB color space.
    LAB_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(float)

    # Create a Gaussian pyramid.
    pyramid = [LAB_image]
    for _ in range(gaussian_levels - 1): 
        LAB_image = cv2.pyrDown(LAB_image)
        pyramid.append(LAB_image)

    # Perform mean shift on each pyramid level, starting from the smallest scale.
    for level in reversed(pyramid):
        h, w = level.shape[:2]
        for y in range(h):
            for x in range(w):
                LAB = level[y, x]
                for _ in range(max_iterations):
                    new_LAB = mean_shift((x, y, LAB), level, sth, cth)
                    if lab_pixel_distance(LAB, new_LAB) < 1e-2:  # Convergence threshold.
                        break
                    LAB = new_LAB
                level[y, x] = LAB

        # Propagate to the upper scale.
        if pyramid.index(level) > 0:
            upper = pyramid[pyramid.index(level) - 1]
            level_upsampled = cv2.pyrUp(level)

            # Ensuring same shape by cropping or padding as necessary
            if level_upsampled.shape != upper.shape:
                level_upsampled = cv2.resize(level_upsampled, (upper.shape[1], upper.shape[0]))

            mask = lab_distance(level_upsampled, upper) > cth
            upper[mask] = level_upsampled[mask]

    if cast_back:
        return cv2.cvtColor(pyramid[0].astype(np.uint8), cv2.COLOR_Lab2BGR)
    
    return  pyramid[0].astype(np.uint8)


def get_neighbour(x0, y0, h, w):
    neighbours = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if (i, j) == (0, 0):
                continue
            x = x0 + i
            y = y0 + j
            if 0 <= x < h and 0 <= y < w:
                neighbours.append((x, y))
    return neighbours


def distance(im, x, y, x0, y0):
    return (abs((int(im[x, y, 0]) - int(im[x0, y0, 0]))) + 
            (int(im[x, y, 1]) - int(im[x0, y0, 1]))**2 + 
            (int(im[x, y, 2]) - int(im[x0, y0, 2]))**2)**0.5


def bfs(im, x0, y0, passed_by, stack, thresh, h, w):
    region_num = passed_by[x0, y0]
    elems = []
    elems.append((int(im[x0, y0, 0]) + int(im[x0, y0, 1]) + int(im[x0, y0, 2]))/3)
    var = thresh
    neighbours = get_neighbour(x0, y0, h, w)

    for x, y in neighbours:
        if passed_by[x, y] == 0 and distance(im, x, y, x0, y0) < var:
            passed_by[x, y] = region_num
            stack.push((x, y))
            elems.append((int(im[x, y, 0]) + int(im[x, y, 1]) + int(im[x, y, 2]))/3)
            var = np.var(elems)
        var = max(var, thresh)

def region_growing(im, thresh):
    """
    https://github.com/Spinkoo/Region-Growing/blob/master/RegionGrowing.py
    """
    h, w, _ = im.shape
    passed_by = np.zeros((h, w), np.double)
    current_region = 0
    iterations = 0
    segs = np.zeros((h, w, 3), dtype='uint8')
    stack = Stack()

    for x0 in range(h):
        for y0 in range(w):
            if passed_by[x0, y0] == 0 and (int(im[x0, y0, 0]) * int(im[x0, y0, 1]) * int(im[x0, y0, 2]) > 0):
                current_region += 1
                passed_by[x0, y0] = current_region
                stack.push((x0, y0))
                while not stack.is_empty():
                    x, y = stack.pop()
                    bfs(im, x, y, passed_by, stack, thresh, h, w)
                    iterations += 1

    for i in range(h):
        for j in range(w):
            val = passed_by[i, j]
            if val == 0:
                segs[i, j] = 255, 255, 255
            else:
                segs[i, j] = val*35, val*90, val*30

    return segs


def merge_small_regions(img, segs, ath=100, cth=30):
    h, w, _ = segs.shape
    unique_regions = np.unique(segs.reshape(-1, 3), axis=0)
    
    # Compute the area for each region
    region_areas = {}
    for region in unique_regions:
        mask = np.all(segs == region, axis=2)
        region_areas[tuple(region)] = np.sum(mask)
    
    # For each small region, merge it with the nearest neighbor
    for region, area in region_areas.items():
        if area < ath:
            x, y = np.argwhere(np.all(segs == region, axis=2))[0]
            min_dist = float('inf')
            nearest_region = None
            
            # Check neighbors
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for nx, ny in neighbors:
                if 0 <= nx < h and 0 <= ny < w:
                    neighbor_region = tuple(segs[nx, ny])
                    if neighbor_region != region:
                        # color_dist = np.linalg.norm(np.array(region) - np.array(neighbor_region))
                        color_dist = distance(img, x, y, nx, ny)
                        if color_dist < min_dist:
                            min_dist = color_dist
                            nearest_region = neighbor_region
            
            # Merge the regions if the color distance is below a threshold
            if min_dist < cth and nearest_region:
                segs[np.all(segs == region, axis=2)] = nearest_region
                
    return segs