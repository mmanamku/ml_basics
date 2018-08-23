import numpy as np
import cv2
import matplotlib.pyplot as plt

eyeCascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')
faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')

for n in range(1, 22):
    img = cv2.imread(str(n) + ".pgm", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        k = 1
        for (ex, ey, ew, eh) in eyes:
            roi_gray_eye = roi_gray[ey:ey+eh, ex:ex+ew]
            sobel = cv2.Sobel(roi_gray_eye,cv2.CV_64F,0,1,ksize=5)
            cv2.imshow('kuch nahi', sobel)
            sobel = sobel.astype(np.uint8)
            #cv2.imshow('kuch nahi', sobel)
            cv2.waitKey(0)
            ret, gray_t = cv2.threshold(sobel, 80, 255, cv2.THRESH_BINARY_INV)
            gray_t = 255-gray_t
            cv2.imshow("image_", gray_t)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
