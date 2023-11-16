import argparse
import torch
import cv2 as cv

from modules.plate_detection import PlateDetector
from modules.food_segmentation import FoodSegmenter
from modules.food_recognition import FoodRecognizer
from modules.volume_estimation import VolumeEstimator
from modules.carbo_estimation import CarboEstimator


def parse_args():
    """
    Read the terminal inputs.
    """

    parser = argparse.ArgumentParser(
        prog='CarboCount',
        description='Given two stereo images representing the same dish, segment the different foods and compute the quanity of carbohydrates within them.'
    )
    
    parser.add_argument('-i', '--image', type=str, required=True, help='Path to image.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display the process step by step.')
    parser.add_argument('-f', '--focal_length', type=int, default=50, help='Focal lenght of the camera (in mm). (default: 50).')
    parser.add_argument('-d', '--diameter', type=int, default=260, help='Actual diameter of the plate (in mm). (default: 260)')
    parser.add_argument('--segmenter', type=str, default='slic', help='Model used for food segmentation.')
    parser.add_argument('--classifier', type=str, default='resnet50', help='Model used for food classification.')
    parser.add_argument('--classes_file', type=str, default='data/food-101/meta/classes.txt', help='Path to the file containing the food class names. txt format; one class per row. (default: "data/food-101/meta/classes.txt")')
    parser.add_argument('--nutrients_file', type=str, default='data/food_nutrients.csv', help='Path to the file containing the food nutrient informations is csv format; one class per row. (default: "data/food_nutrients.csv")')

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
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Load and resize image
    image = cv.imread(args.image)
    image = resize_image(image, 480)

    # Init framework modules
    plate_detector  = PlateDetector()
    food_segmenter  = FoodSegmenter(model=args.segmenter)
    food_recognizer = FoodRecognizer(model=args.classifier, num_classes=104, class_names=idx2class, device=device)
    volume_estimator = VolumeEstimator(idx2class=idx2class, inference_dataset='nyu')
    carbo_estimator = CarboEstimator(food_nutrients_file=args.nutrients_file)

    # Run framework
    plate_coords, plate_mask = plate_detector(image, display=args.verbose)
    return
    segmentation_map = food_segmenter(image, plate_mask, display=args.verbose)
    segmentation_map= food_recognizer(image, segmentation_map, display=args.verbose)
    food_volumes = volume_estimator(image, segmentation_map, plate_coords, focal_length=args.focal_length, plate_diameter=args.diameter, display=args.verbose)
    food_carbs = carbo_estimator(food_volumes=food_volumes)


    # Compute the total carbohydrates quantity
    total_carbs = sum(food_carbs.values())
    print("\n\n")
    print("***********************************")
    print(f"Total Carbohydrates: {round(total_carbs,2)}")
    print("***********************************")
    print("\n\n")




if __name__ == "__main__":
    main()