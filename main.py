import argparse
import numpy as np

from modules.plate_detection import PlateDetector
from modules.food_segmentation import FoodSegmenter
from modules.food_recognition import FoodRecognizer
from modules.volume_estimation import VolumeEstimator

from utils.read_write import load_image

def parse_args():
    """
    Read the terminal inputs.
    """

    parser = argparse.ArgumentParser(
        prog='CarboCount',
        description='Given two stereo images representing the same dish, segment the different foods and compute the quanity of carbohydrates within them.'
    )
    
    parser.add_argument('-l', '--left_image', type=str, required=True, help='Path to the left stereo image.')
    parser.add_argument('-r', '--right_image', type=str, required=True, help='Path to the right stereo image.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display the process step by step.')
    parser.add_argument('--segmenter', type=str, default="slic", help="Model used for food segmentation.")
    parser.add_argument('--classifier', type=str, default="inception_v3", help="Model used for food classification.")

    return parser.parse_args()


def main():
    # Read terminal input
    args = parse_args()

    # Load images
    left_image = load_image(args.left_image)
    right_image = load_image(args.right_image)

    # Init framework modules
    plate_detector = PlateDetector()
    food_segmenter = FoodSegmenter(model=args.segmenter)
    food_recognizer = FoodRecognizer(model=args.classifier)
    volume_estimator = VolumeEstimator()

    # Run framework
    plate_coords, plate_mask = plate_detector(left_image)
    segmentation_map = food_segmenter(left_image, plate_mask, display=args.verbose)
    # segmentation_map = food_recognizer(left_image, segmentation_map, display=args.verbose)
    # volume_estimator(left_image, right_image, segmentation_map, reference_img=None, reference_size=None, display=True)

if __name__ == "__main__":
    main()




