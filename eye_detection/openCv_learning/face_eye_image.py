import numpy as np
import cv2
import cv2.cv as cv
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('../haar/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('../haar/haarcascade_eye_tree_eyeglasses.xml')
gray = cv2.imread("image_gray.jpeg", 0)
#img = cv2.resize(frame,(500,500))
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    cv2.imwrite("image_face_marked.jpeg", gray);
    cv2.imwrite("image_face.jpeg", roi_gray);
    roi_color = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    eyev = 1;
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        roi_gray_t_e = roi_gray[ey:ey+eh, ex:ex+ew]
        cv2.imwrite("image_eye_marked.jpeg", roi_gray);
        cv2.imwrite("image_eye_"+ str(eyev) +".jpeg", roi_gray_t_e);
        eyev = 2;
#cv2.imshow("gray", gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


