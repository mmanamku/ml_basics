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

image = [] 
fillZero = 0

if not args.get('file', False):
    cap = cv2.VideoCapture(0)             
else:
    cap = cv2.VideoCapture(args['file'])       
    
def cluster(x, y, fillVal, left, right, top, bottom):
    if y<left or y>right or x<top or x>bottom:
        fillZero = 1
        try:
            image[i][j] = 0
        except:
            print("range out of bound")
    try:
        image[x][y] = fillVal
        if image[x+1][y] == 255:
            cluster(x+1, y, fillVal)
        if image[x-1][y] == 255:
            cluster(x-1, y, fillVal)
        if image[x+1][y+1] == 255:
            cluster(x+1, y+1, fillVal)
        if image[x-1][y-1] == 255:
            cluster(x-1, y-1, fillVal)
        if image[x][y+1] == 255:
            cluster(x, y+1, fillVal)
        if image[x][y-1] == 255:
            cluster(x, y-1, fillVal)
        if image[x-1][y+1] == 255:
            cluster(x-1, y+1, fillVal)
        if image[x+1][y-1] == 255:
            cluster(x+1, y-1, fillVal)
    except:
        print("range out of bound")

def frameCrunch(img, left, right, top, bottom):
    image = img
    img = 0
    fillVal = 0
    (w, h) = image.shape[:2]
    fillZero = 0
    for i in range(0, h):
        for j in range(0, w):
            if image[i][j] == 255:
                fillVal -= 1
                cluster(i ,j, fillVal, left, right, top, bottom)
                if fillZero == 1:
                    for k in range(0, h):
                        for l in range(0, w):
                            if image[k][l] == fillVal:
                                image[k][l] = 0
                                
                    fillZero = 0
    img = image
    return img

def clearAll(img, x, y):
    (w, h) = img.shape[:2]
    if x<h-1 and y<w-1 and x>0 and y>0:
        if img[i][j] == 255:
            
            return clearAll[i+1]

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
            roi_gray_e = roi_gray[eye['y']+(eye['h']*0.30):eye['y']+(eye['h']*0.70), eye['x']+(eye['w']*0.15):eye['x']+(eye['w']*0.85)]
            (w1, h1) = roi_gray_e.shape[:2]
            roi_gray_e = cv2.resize(roi_gray_e, (100*h1/w1, 100))
            cv2.imshow("roi_gray_e", roi_gray_e)
            cv2.waitKey(0)
            equ = cv2.equalizeHist(roi_gray_e)
            cv2.imshow("equ", equ)
            cv2.waitKey(0)
            #cv2.putText(gray ,str(roi_gray_e.mean()), (10, 20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(roi_gray_e)
            cv2.imshow("cl1", cl1)
            cv2.waitKey(0)
            cl2 = clahe.apply(equ)
            cv2.imshow("cl2", cl2)
            cv2.waitKey(0)
            #ret,gray_t_cl1 = cv2.threshold(cl1,40,255,cv2.THRESH_BINARY)
            #gray_t_cl1 = 255 - gray_t_cl1
            #cv2.imshow("gray_t_cl1", gray_t_cl1)
            #cv2.waitKey(0)
            #ret,gray_t_cl2 = cv2.threshold(cl2,40,255,cv2.THRESH_BINARY)
            #gray_t_cl2 = 255 - gray_t_cl2
            #cv2.imshow("gray_t_cl2", gray_t_cl2)
            #cv2.waitKey(0)
            ret,gray_t_cl1 = cv2.threshold(cl1,40,255,cv2.THRESH_BINARY)
            ret,gray_t_cl1I = cv2.threshold(cl1,240,255,cv2.THRESH_BINARY_INV)
            ret,gray_t_cl2 = cv2.threshold(cl2,40,255,cv2.THRESH_BINARY)
            ret,gray_t_cl2I = cv2.threshold(cl2,240,255,cv2.THRESH_BINARY_INV)
            ret,gray_t = cv2.threshold(equ,240,255,cv2.THRESH_BINARY_INV)
            gray_t_cl1 = 255-gray_t_cl1
            cv2.imshow("gray_t_cl1", gray_t_cl1)
            cv2.waitKey(0)
            gray_t_cl1I = 255-gray_t_cl1I
            cv2.imshow("gray_t_cl1I", gray_t_cl1I)
            cv2.waitKey(0)
            gray_t_cl2 = 255-gray_t_cl2
            cv2.imshow("gray_t_cl2", gray_t_cl2)
            cv2.waitKey(0)
            gray_t_cl2I = 255-gray_t_cl2I
            cv2.imshow("gray_t_cl2I", gray_t_cl2I)
            cv2.waitKey(0)
            gray_t2 = 255-gray_t
            cv2.imshow("gray_t2", gray_t2)
            cv2.waitKey(0)
            for i in [0, 1, 2, 3, 4, h-4, h-3, h-2, h-1, h]:
                for j in [0, 1, 2, 3, 4, w-4, w-3, w-2, w-1, w]:
                    if gray_t2[i][j] == 255:
                        gray_t2 = clearAll(gray_t2, i , j)
            i += 1
             
        cv2.imshow("Image", gray)
        cv2.waitKey(0)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
        
#End
cap.release()
cv2.destroyAllWindows()
