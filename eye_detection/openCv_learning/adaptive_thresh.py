import numpy as np
import cv2
import cv2.cv as cv
import matplotlib.pyplot as plt

img = cv2.imread('../Datasets/testImages/1.jpeg')
img = cv2.resize(img,(500,500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,gray_t = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
ret,gray_t = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
cv2.imshow("gray", gray_t)
cv2.waitKey(0)
cv2.destroyAllWindows()
