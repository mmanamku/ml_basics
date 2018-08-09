import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('BioID_0000.pgm',0)
#ret,thresh1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
#sobely1 = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=5)
#img2 = cv2.imread('BioID_0001.pgm',0)
ret,thresh2 = cv2.threshold(img1,127,255,cv2.THRESH_BINARYINV)
#sobely2 = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=5)

#plt.subplot(2,3,1),plt.imshow(img1,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,3,2),plt.imshow(thresh1,cmap = 'gray')
#plt.title('Thresh'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,3,3),plt.imshow(sobely1,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,3,4),plt.imshow(img2,cmap = 'gray')
#plt.title('Thresh'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,3,5),plt.imshow(thresh2,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,3,6),plt.imshow(sobely2,cmap = 'gray')
#plt.title('Thresh'), plt.xticks([]), plt.yticks([])
plt.hist(thresh2.ravel(), 256, [0, 256])

plt.show()
