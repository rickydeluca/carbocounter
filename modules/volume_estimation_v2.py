

class VolumeEstimator:

    def __init__(self, inference_dataset='nyu'):
        self.inference_dataset = inference_dataset

    def __call__(self, input_img, segmentation_map):
        self.input_img = input_img
        self.segmentation_map = segmentation_map



