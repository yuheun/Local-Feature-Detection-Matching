import cv2 as cv
import numpy as np
import matplotlib as plt
from google.colab.patches import cv2_imshow

house = cv.imread('house.jpg')
house_gray = cv.cvtColor(house, cv.COLOR_BGR2GRAY)

harris = cv.cornerHarris(house_gray, 2, 3, 0.04)

# coordinates with a maximum value of 10% or more of the change result
coord = np.where(harris > 0.1 * harris.max())
coord = np.stack((coord[1], coord[0]), axis = -1)

# circle on corner
for x, y in coord:
  cv.circle(house, (x, y), 5, (0, 0, 255), 1, cv.LINE_AA)

# normalize
harris_norm = cv.normalize(harris, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

# display
harris_norm = cv.cvtColor(harris_norm, cv.COLOR_GRAY2BGR)
house_merge = np.hstack((harris_norm, house))

cv2_imshow(house_merge)

cv.waitKey()
cv.destroyAllWindows()
