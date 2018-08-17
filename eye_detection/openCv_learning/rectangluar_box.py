# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
 
# load the image
image = cv2.imread("image_face.jpeg")
(w, h) = image.shape[:2]
# loop over the alpha transparency values
alpha = 0.3
# create two copies of the original image -- one for
# the overlay and one for the final output image
overlay = image.copy()
output = image.copy()

# draw a red rectangle surrounding Adrian in the image
# along with the text "PyImageSearch" at the top-left
# corner
cv2.rectangle(overlay, (0,h/4),(w,2*h/4),
	(0, 0, 255), -1)
#cv2.putText(overlay, "PyImageSearch: alpha={}".format(alpha),
	#(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
# apply the overlay
cv2.addWeighted(overlay, alpha, output, 1 - alpha,
	0, output)
cv2.imwrite("image_eye_grid_marked.jpeg", output);
# show the output image
#print("alpha={}, beta={}".format(alpha, 1 - alpha))
cv2.imshow("Output", output)
cv2.waitKey(0)
