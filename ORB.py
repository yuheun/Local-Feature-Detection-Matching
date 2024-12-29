import cv2 as cv
import numpy as np
import matplotlib as plt
from google.colab.patches import cv2_imshow

box = cv.imread('box.png')
orb = cv.ORB_create() 

# FAST key detector
kp = orb.detect(box, None)

# draw key points
box_kp = cv.drawKeypoints(box, kp, None, color=(0, 255, 0), flags=0)

plt.imshow(box_kp), plt.show() #plt.imshow in matplotlib.pylab

# BRIEF descriptor
house = cv.imread('house.jpg')
house_orb_kp, house_orb_descriptor = orb.detectAndCompute(house, None)

house_orb_draw_rich = cv.drawKeypoints(house, house_orb_kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
house_orb_draw_default = cv.drawKeypoints(house, house_orb_kp, None, flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
house_orb_draw_single_point = cv.drawKeypoints(house, house_orb_kp, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
house_orb_draw = cv.drawKeypoints(house, house_orb_kp, None, color=(0,255,0), flags=0)

cv2_imshow(house_orb_draw_default)
cv2_imshow(house_orb_draw_rich)
cv2_imshow(house_orb_draw_single_point)
cv2_imshow(house_orb_draw)

cv.waitKey()
cv.destroyAllWindows()

"""
matching
"""

orb_kp, orb_desc = orb.detectAndCompute(taekwon_gray, None)
orb_kp2, orb_desc2 = orb.detectAndCompute(figures_gray, None)

# brute-force matching with orb descriptors

bf_orb = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# matching computing
matches_orb = bf_orb.match(orb_desc, orb_desc2)

# sort
matches_orb = sorted(matches_orb, key = lambda x:x.distance)

# draw
orb_match_draw = cv.drawMatches(taekwon, orb_kp, figures, orb_kp2, matches_orb[:10], None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(orb_match_draw), plt.show

min, max = matches_orb[0].distance, matches_orb[-1].distance

# create a critical point at 20% of the minimum distance
ratio = 0.1
good_thres = (max - min) * ratio + min

good_matches = [m for m in matches_orb if m.distance < good_thres]
print('matches: %d/%d, min: %.2f, max: %.2f, thresh: %.2f'
        %(len(good_matches), len(matches_orb), min, max, good_thres))

orb_good_match_draw = cv.drawMatches(taekwon, orb_kp, figures, orb_kp2, good_matches, None, flags= cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(orb_good_match_draw), plt.show

