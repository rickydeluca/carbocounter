import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import torchvision.models as models
import torchvision.transforms as transforms

from joblib import dump, load
from PIL import Image

from skimage.color import label2rgb

from utils.read_write import load_image
from .plate_detection import PlateDetector
from .food_segmentation import FoodSegmenter


class FoodRecognizer:
    def __init__(self, model="inception_v3"):
        
        # Load trained model
        if model == "inception_v3":
            self.model = models.inception_v3(weights='IMAGENET1K_V1')
            self.model.fc = nn.Sequential(
                nn.BatchNorm1d(2048, eps=0.001, momentum=0.01),
                nn.Dropout(0.2),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 101),
                nn.Softmax(dim=1)
            )

            self.model.load_state_dict(torch.load(
                'best_models/inception_v3.pth',
                map_location=torch.device('cpu')))

            self.model.eval()

        elif model == "svm":
            self.model = load('best_models/svm.joblib')

        else:
            raise ValueError("Classification model not supported. Please use 'inception_v3' or 'svm'.")
    
    def preprocess_segment(self, segment):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        segment = segment.astype(np.uint8)  # Ensure data type is uint8
        return preprocess(segment)
    
    def predict(self, segment):
        input_tensor = self.preprocess_segment(segment).unsqueeze(0)  # add batch dim
        with torch.no_grad():
            output = self.model(input_tensor)
        return output
    
    def extract_and_predict(self, image, segments):
        for i, seg_val in enumerate(np.unique(segments)):
            if i == 0:      # Discard background
                continue

            mask = np.zeros_like(segments)
            mask[segments == seg_val] = 1
            
            # Extract region and preprocess
            region = image * np.expand_dims(mask, axis=-1)      # Extract region
            region = region[np.ix_(mask.any(1),mask.any(0))]    # Crop to bounding box
            
            # Predict
            prediction = self.predict(region)
            predicted_label = prediction.argmax().item()

            # Display
            self.display_segment(region, predicted_label)
            print("Segment:", seg_val, "Prediction:", predicted_label)


    def _extract_and_predict(self, image, segments):
        for i, seg_val in enumerate(np.unique(segments)):
            if i == 0:      # Discard background
                continue

            mask = np.zeros_like(segments)
            mask[segments == seg_val] = 1
            
            # Extract region and preprocess
            region = image * np.expand_dims(mask, axis=-1)  # Extract region
            region_background = image * np.expand_dims(1-mask, axis=-1) # Extract inverse region (background)
            region = region + region_background # Combine to keep original background
            
            # Crop to bounding box
            coords = np.column_stack(np.where(mask))
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            region = region[x_min:x_max+1, y_min:y_max+1]
            
            # Predict
            prediction = self.predict(region)
            predicted_label = prediction.argmax().item()

            # Display
            self.display_segment(region, predicted_label)
            print("Segment:", seg_val, "Prediction:", predicted_label)


    def display_segment(self, segment, predicted_label):
        # Note: you may map `predicted_label` to a string label if you have a mapping
        plt.imshow(segment)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()

    
    def visualize_results(self, image_rgb, image_segments, merged_segments=None):
        plt.subplot(1,3,1)
        plt.imshow(image_rgb)
        plt.title('Original Image')

        plt.subplot(1,3,2)
        plt.imshow(label2rgb(image_segments, image_rgb, kind='avg'))
        plt.title('SLIC Segments')

        if merged_segments is not None:
            plt.subplot(1,3,3)
            plt.imshow(label2rgb(merged_segments, image_rgb, kind='avg'))
            plt.title('Merged Segments')
        
        plt.show()


if __name__ == "__main__":

    # Load sample image
    path = "test/test_dish_4.jpg"
    image = load_image(path, max_size=120000)
    
    # Init modules
    plate_detector = PlateDetector()
    food_segmenter = FoodSegmenter(slic=True)
    food_recognizer = FoodRecognizer(model='inception_v3')

    # Segment food
    _, plate_mask = plate_detector.detect_plate_and_mask(image, scale=1.0)
    image_segments = food_segmenter(image, plate_mask, display=True)

    # Classfy food
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    merged_segments = food_recognizer.extract_and_predict(image_rgb, image_segments)
    food_recognizer.visualize_results(image_rgb, image_segments, merged_segments)