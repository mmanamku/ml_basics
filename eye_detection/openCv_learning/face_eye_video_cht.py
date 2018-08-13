import numpy as np
import cv2
import cv2.cv as cv
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('../haar/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('../haar/haarcascade_eye_tree_eyeglasses.xml')
cap = cv2.VideoCapture('../Datasets/video/videoplayback')
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            circles = cv2.HoughCircles(roi_gray,cv.CV_HOUGH_GRADIENT,1,20,param1=150,param2=30,minRadius=20,maxRadius=100)
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(roi_gray_t,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(roi_gray_t,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
