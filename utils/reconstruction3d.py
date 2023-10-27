import cv2 as cv
import numpy as np
import torchvision.transforms as transforms

from models.AdaBins.infer import InferenceHelper


def distance_from_diameter(focal_length, actual_diameter, diameter_in_pixels):
    print("focal len", focal_length)
    print("actual diam", actual_diameter)
    
    return (focal_length * actual_diameter) / diameter_in_pixels


def infer_depth_map(img, inference_dataset='nyu'):
    # Get model from AdaBins
    infer_helper = InferenceHelper(dataset=inference_dataset)

    # Covert input CV image to PIL
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
    img_pil = transforms.ToPILImage()(img_rgb.astype(np.uint8))

    # Resize to standard shape
    img_pil = img_pil.resize((640, 480))

    # DEBUG
    # print("PIL image size:", img_pil.size)

    bin_centers, predicted_depth = infer_helper.predict_pil(img_pil)

    return predicted_depth[0][0]


def rescale_depth(depth_value, min_old, max_old, min_new, max_new):
    """
    Rescale depth_value from [min_old, max_old] to [min_new, max_new]
    """
    return min_new + ((depth_value - min_old) / (max_old - min_old)) * (max_new - min_new)


def smooth_mask(mask, kernel_size=3):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Closing
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    # Opening
    opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)
    return opened

