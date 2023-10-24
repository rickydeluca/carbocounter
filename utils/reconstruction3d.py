import cv2 as cv
import numpy as np
import torchvision.transforms as transforms

from models.AdaBins.infer import InferenceHelper


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

