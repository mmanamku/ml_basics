import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('/home/arjun/Desktop/ger/training-repo/Team_E/shakir/code_understanding/index.jpeg', 1)
print (img == None)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("img",img_rgb)#hsv
cv.waitKey(0)
cv.destroyAllWindows()

