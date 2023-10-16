import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def compute_disparity(img_l, img_r, block_size=None, min_disp=None, num_disp=None, disp_12_max_diff=None, P1=None, P2=None, uniqueness_ratio=None, speckle_window_size=None, speckle_range=None):

    # Instanitate StereoSGBM class
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
                                  numDisparities = num_disp,
                                  blockSize = block_size,
                                  P1 = P1,
                                  P2 = P2,
                                  disp12MaxDiff = disp_12_max_diff,
                                  uniquenessRatio = 0,
                                  speckleWindowSize = 0,
                                  speckleRange = 0)
    
    # Compute disparity
    disp = stereo.compute(img_l, img_r).astype(np.float32) / min_disp

    # Display disparity
    plt.imshow(disp,'gray')
    plt.show()

    return disp

def generate_3d_point_cloud(img, height, width, focal_length, disp):
    # Turn point 180 degrees so that the y-axis is in up
    Q = np.float32([[1, 0, 0,   -0.5*width],
                    [0,-1, 0,   0.5*height],
                    [0, 0, 0,-focal_length], 
                    [0, 0, 1,           0]])
    
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]

    return out_points, out_colors


def visualize_3d_point_cloud(points, colors):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors/255.0, s=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()