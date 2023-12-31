import os
import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from joblib import load
from skimage.color import label2rgb

from models.ResNet.resnet import get_resnet
from models.SVM.utils import extract_features

class FoodRecognizer:
    def __init__(self, model="resnet50", num_classes=104, class_names = None, device=None):
        self.model_name = model
        self.num_classes = num_classes
        self.class_names = class_names
        self.img = None
        self.segmentation_map = None
        self.merged_segmentation_map = None
        self.device = device
        self.display = False

        # Load trained model
        if "resnet" in self.model_name:

            # Get resnet version
            self.model = get_resnet(self.model_name,
                                    self.num_classes,
                                    pretrained=True,
                                    freeze_weights=True)
            
            # Load trained parameters
            self.model.load_state_dict(torch.load(f"trained_params/{self.model_name}.pth"))
            self.model = self.model.to(self.device)
            self.model.eval()

            # Get data transformations
            self.data_transforms = {

                'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),

                'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }

        elif self.model_name == "svm":
            self.model = load('trained_params/svm.joblib')

        else:
            raise ValueError("Classification model not supported. Please use 'resnet<18, 50 or 101>' or 'svm'.")
    
    def __call__(self, img, segmentation_map, display=False):
        self.img = img
        self.segmentation_map = segmentation_map
        self.display = display
        return self.extract_and_predict_crop()
    
    
    def predict(self, segment):
        
        if "resnet" in self.model_name:
            
            # Convert RGB segment to PIL image
            segment = transforms.ToPILImage()(segment.astype(np.uint8))

            # Apply transformation (in test mode)
            segment = self.data_transforms['val'](segment)
            segment = segment.unsqueeze(0)
            segment = segment.to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(segment)
                _, preds = torch.max(outputs, 1)                
                return preds[0]
        
        if self.model_name == "svm":

            # Save the found segment in a temporary folder (for further prediction)
            if not os.path.exists("tmp"):   # Ensure the "tmp/" directory exists
                os.makedirs("tmp")
            filename = f"tmp/segment.jpg"
            cv.imwrite(filename, cv.cvtColor(cv.convertScaleAbs(segment), cv.COLOR_RGB2BGR))

            segment_img = cv.imread(filename)
            segment_feat = np.array([extract_features(segment_img)])
            return self.model.predict(segment_feat)


    def extract_and_predict(self):

        # Convert input image to RGB
        img_rgb = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)

        # Init the merged segmentation map, where segments classified with the
        # same labels are merged. The pixel values in this new segmentation map
        # represent the predicted label value.
        self.merged_segmentation_map = np.zeros_like(self.segmentation_map)

        for i, seg_val in enumerate(np.unique(self.segmentation_map)):
            if i == 0:      # Discard background
                continue

            mask = np.zeros_like(self.segmentation_map)
            mask[self.segmentation_map == seg_val] = 1
            
            # Extract region and preprocess
            region = img_rgb * np.expand_dims(mask, axis=-1)    # Extract region
            region = region[np.ix_(mask.any(1),mask.any(0))]    # Crop to bounding box

            # Predict
            prediction = self.predict(region)
            predicted_label = self.class_names[int(prediction.item())]

            # Update the merged segmentation map and the predictions dictionary
            self.merged_segmentation_map[mask == 1] = prediction

            # Display
            if self.display:
                self.display_segment(region, predicted_label)
                print("Segment:", seg_val, "Prediction:", predicted_label)


        if self.display:
            self.visualize_results()

        return self.merged_segmentation_map


    def extract_and_predict_crop(self):
        # Convert input image to RGB
        img_rgb = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)

        # Init the merged segmentation map, where segments classified with the
        # same labels are merged. The pixel values in this new segmentation map
        # represent the predicted label value.
        self.merged_segmentation_map = np.zeros_like(self.segmentation_map)

        for i, seg_val in enumerate(np.unique(self.segmentation_map)):
            if i == 0:      # Discard background
                continue

            mask = np.zeros_like(self.segmentation_map)
            mask[self.segmentation_map == seg_val] = 1
            
            # Extract region and preprocess
            region = img_rgb * np.expand_dims(mask, axis=-1)  # Extract region
            region_background = img_rgb * np.expand_dims(1-mask, axis=-1) # Extract inverse region (background)
            region = region + region_background # Combine to keep original background
            
            # Crop to bounding box
            coords = np.column_stack(np.where(mask))
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            region = region[x_min:x_max+1, y_min:y_max+1]
            
            # Predict
            prediction = self.predict(region)
            predicted_label = self.class_names[int(prediction.item())]

            # Update the merged segmentation map and the predictions dictionary
            self.merged_segmentation_map[mask == 1] = prediction

            # Display
            if self.display:
                self.display_segment(region, predicted_label)
                print("Segment:", seg_val, "Prediction:", predicted_label)

        
        if self.display:
            self.visualize_results()

        return self.merged_segmentation_map


    def display_segment(self, segment, predicted_label):
        # Note: you may map `predicted_label` to a string label if you have a mapping
        plt.imshow(segment)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()

    
    def visualize_results(self):
        img_rgb = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        plt.subplot(1,3,1)
        plt.imshow(img_rgb)
        plt.title('Original Image')

        plt.subplot(1,3,2)
        plt.imshow(label2rgb(self.segmentation_map, img_rgb, kind='avg'))
        plt.title('SLIC Segments')

        plt.subplot(1,3,3)
        plt.imshow(label2rgb(self.merged_segmentation_map, img_rgb, kind='avg'))
        plt.title('Merged Segments')
        
        plt.show()