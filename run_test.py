import os
import csv
import torch
import argparse
import cv2 as cv
import numpy as np
import pandas as pd

from modules.plate_detection import PlateDetector
from modules.food_segmentation import FoodSegmenter
from modules.food_recognition import FoodRecognizer
from modules.volume_estimation import VolumeEstimator
from modules.carbo_estimation import CarboEstimator

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    """
    Read the terminal inputs.
    """

    parser = argparse.ArgumentParser(
        prog='Test CarboCount',
        description='Test the framework on the given images'
    )
    
    parser.add_argument('-i', '--image_folder', type=str, required=True, help='Path to image folder.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display the process step by step.')
    parser.add_argument('--segmenter', type=str, default='slic', help='Model used for food segmentation.')
    parser.add_argument('--classifier', type=str, default='resnet50', help='Model used for food classification.')
    parser.add_argument('--classes_file', type=str, default='data/food-101/meta/classes.txt', help='Path to the file containing the food class names. txt format; one class per row. (default: "data/food-101/meta/classes.txt")')
    parser.add_argument('--nutrients_file', type=str, default='data/food_nutrients.csv', help='Path to the file containing the food nutrient informations is csv format; one class per row. (default: "data/food_nutrients.csv")')
    parser.add_argument('--test_file', type=str, default='test/groundtruth.csv', help='Path to the CSV file containing the information of the test images and the real carbohydrate values. (default: "test/groundtruth.csv")')
    parser.add_argument('--outfile', type=str, default='results/metrics.csv', help='Where to save the result. (default: "result/accuracy.csv")')
    
    return parser.parse_args()


def resize_image(image, size):
    """
    Resize the input image. If `size` is a tuple resize the image to this dimension,
    otherwise id `size` is a scalar value, resize the shortest dimension to this value 
    while keeping the aspect ratio.
    """
    
    if isinstance(size, tuple):
        resized_image = cv.resize(image, size)

    elif isinstance(size, int) or isinstance(size, float):
        height, width = image.shape[:2]
        
        if height < width:
            new_height = int(size)
            new_width = int((size / height) * width)
        else:
            new_width = int(size)
            new_height = int((size / width) * height)

        resized_image = cv.resize(image, (new_width, new_height))
    
    else:
        raise ValueError("Invalid size parameter. Should be a tuple or a scalar value.")

    return resized_image


def main():

    # Read terminal input
    args = parse_args()

    # Get class names dictionary
    idx2class = {}
    with open(args.classes_file) as f:
        for idx, name in enumerate(f):
            idx2class[idx] = name.replace("\n", "")

    # Get device
    device = torch.device("cpu")

    # Load test DataFrames
    df = pd.read_csv(args.test_file, header=0)

    # Init framework modules
    plate_detector  = PlateDetector()
    food_segmenter  = FoodSegmenter(model=args.segmenter)
    food_recognizer = FoodRecognizer(model=args.classifier, num_classes=104, class_names=idx2class, device=device)
    volume_estimator = VolumeEstimator(idx2class=idx2class, inference_dataset='nyu')
    carbo_estimator = CarboEstimator(food_nutrients_file=args.nutrients_file)

    # Run experiments
    experiment_data = []
    
    for idx, row in df.iterrows():
        # Get dish image
        dish_number = f'0{idx+1}' if idx+1 < 10 else f'{idx+1}'
        try:
            f = os.path.join(args.image_folder, f"dish{dish_number}.jpg")
        except:
            f = os.path.join(args.image_folder, f"dish{dish_number}.png") 


        # DEBUG
        # if idx+1 != 15:
        #     continue
    
        # Get values from the current row
        focal = row['focal']
        diameter = row['diameter']

        if os.path.isfile(f):
            # Load and resize image
            image = cv.imread(f)
            image = resize_image(image, 480)

            # Run framework
            plate_coords, plate_mask = plate_detector(image, display=False)
            segmentation_map = food_segmenter(image, plate_mask, display=args.verbose)
            segmentation_map = food_recognizer(image, segmentation_map, display=args.verbose)
            food_volumes = volume_estimator(image, segmentation_map, plate_coords, focal_length=focal, plate_diameter=diameter, display=args.verbose)
            food_carbs = carbo_estimator(food_volumes=food_volumes)

            # Compute the total predicted carbohydrates
            pred_carbs = sum(food_carbs.values())

            experiment_data.append({'dish': idx+1, 'real': row['carbs'], 'estimated': pred_carbs})

    # Compute and store metrics
    with open(args.outfile, mode='w', newline='') as file:
        # Define the CSV writer
        writer = csv.writer(file)

        # Write header
        writer.writerow(['Dish', 'Real', 'Estimated', 'Absolute Error', 'Absolute Percentage Error'])

        # Calculate and write metrics for each dish
        total_absolute_error = 0
        total_real_values = 0

        for exp in experiment_data:
            dish_name = exp['dish']
            real_value = exp['real']
            estimated_value = exp['estimated']

            absolute_error = np.abs(real_value - estimated_value)
            absolute_percentage_error = (absolute_error / real_value) * 100

            # Update total metrics
            total_absolute_error += absolute_error
            total_real_values += real_value

            # Write the metrics to the CSV file
            writer.writerow([dish_name, real_value, estimated_value, absolute_error, absolute_percentage_error])

        # Calculate and write the mean metrics for all dishes
        mean_absolute_error_all = total_absolute_error / len(experiment_data)
        mean_absolute_percentage_error_all = (total_absolute_error / total_real_values) * 100

        # Write mean metrics to the CSV file
        writer.writerow(['Mean', '', '', mean_absolute_error_all, mean_absolute_percentage_error_all])

    print(f"CSV file '{args.outfile}' created successfully.")

if __name__ == "__main__":
    main()