import numpy as np
import cv2
import cv2.cv as cv

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
cap = cv2.VideoCapture('videoplayback')
#img = cv2.imread('BioID_0056.pgm')
#img = cv2.imread('BioID_0000.pgm')
#img = cv2.imread('2.jpeg')
#img1 = cv2.resize(img,(200,200))
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,gray_t = cv2.threshold(gray,100,180,cv2.THRESH_BINARY)
    canny = cv2.Canny(gray,100,200)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(gray_t,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_t = canny[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_gray_t,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
            #circles = cv2.HoughCircles(roi_gray,cv.CV_HOUGH_GRADIENT,1,20,
                           # param1=150,param2=30,minRadius=20,maxRadius=100)
            #circles = np.uint16(np.around(circles))
            #for i in circles[0,:]:
                # draw the outer circle
                #cv2.circle(roi_gray_t,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                #cv2.circle(roi_gray_t,(i[0],i[1]),2,(0,0,255),3)
    #cv2.imshow("gray", gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imshow("gray", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imshow("gray", gray_t)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#for (x,y,w,h) in faces:
#    #cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi_color = img[y:y+h, x:x+w]
#    eyes = eye_cascade.detectMultiScale(roi_gray)
#    for (ex,ey,ew,eh) in eyes:
#	crop_img = roi_gray[ey:ey+eh, ex:ex+ew]	
#	#cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
#	ret,crop_img_t = cv2.threshold(crop_img,125,255,cv2.THRESH_BINARY)
#	cv2.imshow("cropped", crop_img)
#	cv2.waitKey(0)
#        cv2.destroyAllWindows()
#	cv2.imshow("threshold", crop_img_t)
#	cv2.waitKey(0)
#        cv2.destroyAllWindows()
#ret,gray_t = cv2.threshold(gray,125,255,cv2.THRESH_BINARY)
#cv2.imshow("gray", gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imshow("gray", gray_t)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()


