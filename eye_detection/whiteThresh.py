import cv2
import numpy as np

img = cv2.imread("22.pgm")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,110,255,cv2.THRESH_BINARY)
thresh2 = 255 - thresh
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.imshow("thresh2", thresh2)
cv2.waitKey(0)
