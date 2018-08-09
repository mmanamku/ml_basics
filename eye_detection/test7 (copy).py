import numpy as np
import cv2
import cv2.cv as cv
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade_right = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
img = cv2.imread('4.jpeg')
#img = cv2.medianBlur(img,5)
#img1 = cv2.resize(img,(200,500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    crop_img = img[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes_right = eye_cascade_right.detectMultiScale(roi_gray) 
    for (ex,ey,ew,eh) in eyes_right:
	crop_img_left = roi_gray[ey:ey+eh, ex:ex+ew]	
	cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
	#circles = cv2.HoughCircles(crop_img_left,cv.CV_HOUGH_GRADIENT,1,20,
                           # param1=150,param2=30,minRadius=20,maxRadius=100)
	#circles = np.uint16(np.around(circles))
	#for i in circles[0,:]:
         #   cv2.circle(crop_img,(i[0],i[1]),i[2],(255,0,0),2)
         #   cv2.circle(crop_img,(i[0],i[1]),2,(255,0,0),3)
	cv2.imshow("cropped", crop_img_left)
	cv2.waitKey(0)
        cv2.destroyAllWindows()
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

