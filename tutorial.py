import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('csr.jpg', 1) #img name here
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", img)

# initialize the AKAZE descriptor, then detect keypoints and extract
# local invariant descriptors from the image
detector = cv2.AKAZE_create()
(kps, descs) = detector.detectAndCompute(gray, None)
print("keypoints: {}, descriptors: {}".format(len(kps), descs.shape))
 
# draw the keypoints and show the output image
cv2.drawKeypoints(gray, kps, img, (0, 255, 0))

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()