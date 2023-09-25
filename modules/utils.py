import cv2

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image from path: {path}")

    return image