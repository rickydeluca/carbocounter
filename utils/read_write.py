import cv2 as cv
import numpy as np

def load_image(img_path, max_size=None, cvt=None):
    """
    Rescale the input image while preserving its aspect ratio such that its 
    total number of pixels is less than or equal to "max_size".
    
    Args:
        img_path (str): Path to the input image.
        max_size (int, optional): The maximum number of pixels allowed in the output image (default: None).
        cvt (str, optional): Convert image during reading (default: None)
    
    Returns:
        numpy.ndarray: The resized image.
    """
    
    # Read the image.
    if cvt is not None:
        img = cv.imread(img_path, cvt)
    else:
        img = cv.imread(img_path)

    # Get the current dimensions.
    if img.ndim == 3:
        height, width, _ = img.shape
    else:
        height, width, = img.shape
    
    # Calculate the current number of pixels.
    current_pixels = height * width
    
    # If current pixels are already less than N, return the image.
    if max_size is None or current_pixels <= max_size:
        return img

    # Calculate the scaling factor.
    scaling_factor = np.sqrt(max_size / current_pixels)
    
    # Calculate the new dimensions.
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    
    # Resize the image.
    resized_img = cv.resize(img, (new_width, new_height))
    
    return resized_img