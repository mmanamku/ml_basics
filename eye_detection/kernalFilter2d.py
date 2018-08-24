import cv2
import numpy as np

cap = cv2.VideoCapture("video1.mkv")

while True:
    ret, frame = cap.read()
    (w, h) = frame.shape[:2]
    img = cv2.resize(frame, (500, 500*w/h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = cv2.filter2D(gray, -1, kernel)
    cv2.imshow("image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
