import cv2
import numpy as np

rgb_image = cv2.imread("./demo/imgs/out.png", cv2.IMREAD_UNCHANGED)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2BGR)
rgb_image[np.all(rgb_image == [0, 0, 0], axis=-1)] = [255, 255, 255]
cv2.imwrite("out.jpg", rgb_image)