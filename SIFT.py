import cv2 as cv
import numpy as np
import matplotlib as plt
from google.colab.patches import cv2_imshow

sift = cv.xfeatures2d.SIFT_create() 

keypoints, descriptor = sift.detectAndCompute(house_gray, None)
print('keypoint: ', len(keypoints), 'descriptor: ', descriptor.shape)
print(descriptor)

key_in_house = cv.drawKeypoints(house, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

taekwon = cv.imread('taekwonv3.jpg')
figures = cv.imread('figures.jpg')

taekwon_gray = cv.cvtColor(taekwon, cv.COLOR_BGR2GRAY)
figures_gray = cv.cvtColor(figures, cv.COLOR_BGR2GRAY)

sift2 = cv.SIFT_create() 

kp1, desc1 = sift2.detectAndCompute(taekwon_gray, None) 
kp2, desc2 = sift2.detectAndCompute(figures_gray, None)

# crossCheck only reflects mutual matches
matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True) 

matches = matcher.match(desc1, desc2)

res = cv.drawMatches(taekwon, kp1, figures, kp2, matches, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2_imshow(key_in_house)
cv2_imshow(res)

cv.waitKey()
cv.destroyAllWindows()
