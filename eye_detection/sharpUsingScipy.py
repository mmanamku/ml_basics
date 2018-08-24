import scipy
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

#f = scipy.misc.face(gray=True).astype(float)
blurred_f = cv2.imread("blur.png", 0)
#blurred_f = ndimage.gaussian_filter(f, 3)

filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)

alpha = 30
sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

cv2.imshow("sha", sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
