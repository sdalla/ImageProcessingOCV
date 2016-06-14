# import the necessary packages
from __future__ import print_function
import cv2
import numpy as np 
from matplotlib import pyplot as plt 
 
# load the image and convert it to grayscale
image = cv2.imread("jurassic_world.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
laplacian = cv2.Laplacian(image,cv2.CV_64F)
cv2.imshow("Original", image)
 
# initialize the AKAZE descriptor, then detect keypoints and extract
# local invariant descriptors from the image
detector = cv2.AKAZE_create()
(kps, descs) = detector.detectAndCompute(gray, None)
print(cv2.__version__ + "keypoints: {}, descriptors: {}".format(len(kps), descs.shape))
 
# draw the keypoints and show the output image
cv2.drawKeypoints(image, kps, image, (0, 255, 0))
cv2.imshow("laplacian", laplacian)
cv2.imshow("Output", image)
cv2.waitKey(0)