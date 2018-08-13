import numpy as np
import cv2
import cv2.cv as cv
import matplotlib.pyplot as plt

img = cv2.imread('../Datasets/testImages/1.jpeg')
img = cv2.resize(img,(500,500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,gray_t = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
gray_t = 255-gray_t
cv2.imshow("gray", gray_t)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_row_sum = np.sum(gray_t,axis=1).tolist()
plt.plot(img_row_sum)
plt.show()
