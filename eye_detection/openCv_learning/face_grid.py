import numpy as np
import cv2
import cv2.cv as cv
import matplotlib.pyplot as plt

gray = cv2.imread("image_face.jpeg", 0)
(w, h) = gray.shape[:2]
cv2.rectangle(gray,(0,0),(w,h/4),(255,0,0),2)
cv2.rectangle(gray,(0,h/4),(w,2*h/4),(255,0,0),2)
cv2.addWeighted(gray, 0.3, output, 0.7, 0, output)
cv2.rectangle(gray,(0,2*h/4),(w,3*h/4),(255,0,0),2)
cv2.rectangle(gray,(0,3*h/4),(w,h),(255,0,0),2)
cv2.imwrite("image_eye_marked_grid.jpeg", gray);
#cv2.imshow("gray", gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


