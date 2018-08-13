import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("../Datasets/testImages/1.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img1 = cv2.imread('BioID_0000.pgm', 0)
#img2 = cv2.imread('BioID_0001.pgm', 0)


canny = cv2.Canny(gray,100,200)
laplacian = cv2.Laplacian(gray,cv2.CV_64F)
sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

plt.subplot(2,2,1),plt.imshow(sobel,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(thresh,cmap = 'gray')
plt.title('thresh'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(canny,cmap = 'gray')
plt.title('canny'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(laplacian,cmap = 'gray')
plt.title('laplacian'), plt.xticks([]), plt.yticks([])

cv2.imwrite("sobel.jpeg", sobel)
cv2.imwrite("cann.jpeg", canny)
cv2.imwrite("lapl.jpeg", laplacian)
cv2.imwrite("thresh.jpeg", thresh)

plt.show()



