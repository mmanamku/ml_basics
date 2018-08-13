import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('../Datasets/video/Eye_slow.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(400,400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.hist(gray.ravel(), 256, [0, 256])
    plt.subplot(1,2,2),plt.imshow(gray,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



