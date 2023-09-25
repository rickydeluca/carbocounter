import cv2 as cv
img = cv.imread("data/detective_pikachu.jpg")

cv.imshow("Display window", img)
k = cv.waitKey(0)