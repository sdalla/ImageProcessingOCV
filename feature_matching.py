import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img = cv2.imread('NAME', 0)
template = cv2.imread('TEMPLATE', 0)

orb = cv2.ORB_create()

kp1 des1 = orb.detectAndCompute(img, None)
kp2 des2 = orb.detectAndCompute(template, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

# number 10 is number of matches
img3 = cv2.drawMatches(img, kp1, template, kp2, matches[:10], None, flags = 2)
plt.imshow(img3)
plt.show()