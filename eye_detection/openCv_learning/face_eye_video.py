import numpy as np
import cv2
import cv2.cv as cv
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('../haar/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('../haar/haarcascade_eye_tree_eyeglasses.xml')
cap = cv2.VideoCapture('../Datasets/video/Eye_slow.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    img = cv2.resize(frame,(500,500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_gray_t,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


