import numpy as np
import cv2
import cv2.cv as cv
import argparse

#Debug
import matplotlib.pyplot as plt

#Command line argument passing
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='path to file, leave if camera')
args = vars(parser.parse_args())

#Initialization of variables
eyeCascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')
faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
eyesLoc = []                    #list of eye locations
faceLocation = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
facesLoc = []                   #list of face locations

#Debug
frameC = 0              
maxIns = [0, 0]         

if not args.get('file', False):
    cap = cv2.VideoCapture(0)             
else:
    cap = cv2.VideoCapture(args['file'])       

#Begin
while True:
    ret, frame = cap.read()
    
    frameC += 1
    
    if(frame == None):
        break
    
    (w, h) = frame.shape[:2]
    frame = cv2.resize(frame, (500, 500*w/h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    if(len(faces) == 0):
        if(len(facesLoc) > 0): 
            print("Previous face present")
            print("No face detected")
            continue
        else:
            print("No face detected")
            continue
    for (x, y, w, h) in faces:
        
        if(len(facesLoc) > 1):
            del facesLoc[0]            
        faceLocation = {'x': x, 'y': y, 'w': w, 'h': h}
        facesLoc.append(faceLocation)
        
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        
        if(len(eyes) == 0):
            if(len(facesLoc) > 1):
                if(len(eyesLoc) > 0):
                    for i in range(len(eyesLoc)):
                        ex = eyesLoc[0]['x'] - (facesLoc[0]['x'] - faceLocation['x'])
                        ey = eyesLoc[0]['y'] - (facesLoc[0]['y'] - faceLocation['y'])
                        eyesLoc.append({'x': ex, 'y': ey, 'w': int((h/4)), 'h': int((h/4))})
                        del eyesLoc[0]
                        
                else:
                    print("No eye present")
                    break
            else:
                print("No eye present")
                break
                
        else:
            eyesLoc = []
            i = 0
            for (ex, ey, ew, eh) in eyes:
                if i >= 2:
                    break
                eyesLoc.append({'x': ex, 'y': ey, 'w': (h/4), 'h': (h/4)})
                i += 1
                
        i = 0
        for eye in eyesLoc:
        
            roi_gray_e = roi_gray[eye['y']:eye['y']+eye['h'], eye['x']:eye['x']+eye['w']]
            roi_gray_e = cv2.resize(roi_gray_e, (100, 100*w/h))
            if(eye['x'] > (faceLocation['w']/3)):
                i = 1
            else:
                i = 0
            
            ret,gray_t = cv2.threshold(roi_gray_e,110,255,cv2.THRESH_BINARY_INV)
            gray_t = 255-gray_t
            equ = cv2.equalizeHist(roi_gray_e)
            ret,gray_t1 = cv2.threshold(equ,80,255,cv2.THRESH_BINARY)
            gray_t1 = 255-gray_t1
            cv2.imshow("eye_"+str(i), roi_gray_e)
            cv2.waitKey(0)
            cv2.imshow("eye_"+str(i), gray_t)
            cv2.waitKey(0)
            cv2.imshow("eye_"+str(i), equ)
            cv2.waitKey(0)
            cv2.imshow("eye_"+str(i), gray_t1)
            cv2.waitKey(0)
             
        cv2.imshow("Image", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
#End
cap.release()
cv2.destroyAllWindows()
