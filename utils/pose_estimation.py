"""
Source: https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
"""

import numpy as np
import cv2 as cv
import glob

def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)

    return img


def estimate_pose(camera_file):

    # Load camera informations
    with np.load(camera_file) as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    # TODO: Finish to implement this function
    pass

