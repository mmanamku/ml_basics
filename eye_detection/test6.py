import cv2
import numpy as np
from matplotlib import pyplot as plt

#img = cv2.imread('BioID_0000.pgm',0)
#sobel = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#cv2.imwrite('sobel.pgm',img)

img = cv2.imread('messigray.png',0)
thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


titles = ['ORIGINAL','BINARY']
images = [img, thresh]

for i in xrange(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
