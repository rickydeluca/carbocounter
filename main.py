import cv2

from utils.read_write import load_image

img = load_image("test/test_dish_3.png", 120000)
cv2.imshow("Test Dish", img)
cv2.waitKey(0)
cv2.destroyAllWindows()