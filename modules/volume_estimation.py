from utils.stereo_matching import *

class VolumeEstimator:

    def __init__(self):
        self.left_image = None
        self.right_image = None
        self.segmentation_map = None
        self.reference_img = None
        self.reference_size = None
        self.point_cloud = None
        self.display = False
    
    def __call__(self, left_image, right_image, segmentation_map, reference_img, reference_size, display=False):
        self.left_image = left_image
        self.right_image = right_image
        self.segmentation_map = segmentation_map
        self.reference_img = reference_img
        self.reference_size = reference_size
        self.display = display

        self.estimate_volume()

    def segment_point_cloud(self):
        segments = {}
        unique_labels = np.unique(self.segmentation_map)
        for label in unique_labels:
            mask = self.segmentation_map == label
            segment = self.point_cloud[mask]
            segments[label] = segment

        return segments
    
    def estimate_volume(self):
        # Get disparity map
        disp = compute_disparity(self.left_image,
                                self.right_image,
                                block_size=11,
                                min_disp=8,
                                num_disp=16,
                                disp_12_max_diff=12,
                                P1=638,
                                P2=1645,
                                uniqueness_ratio=0,
                                speckle_window_size=0,
                                speckle_range=0)

        # Generate 3D point cloud
        height, width = self.left_image.shape[:2]
        focal_length = 0.8 * width
        points, colors = generate_3d_point_cloud(self.left_image, height, width, focal_length, disp)

        # Display 3D reconstruncted image
        if self.display:
            visualize_3d_point_cloud(points, colors)
