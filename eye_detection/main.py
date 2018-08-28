import numpy as np
import cv2
import cv2.cv as cv
import argparse

#Command line argument passing
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='path to file, leave if camera')
args = vars(parser.parse_args())

#Initialization of variables
eyeCascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')
faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
eyesLoc = []                                    #list of eyes
facesLoc = []                                   #list of faces

#Debug & Reference
frameC = 0                                      #Frame Number

#Capture Video
if not args.get('file', False):
    cap = cv2.VideoCapture(0)                   #camera
else:
    cap = cv2.VideoCapture(args['file'])        #video
    
while True:
    ret, frame = cap.read()
    
    #Debug | Reference
    frameC += 1
    
    #No frame
    if(frame == None)
        print("End of the Video Stream")
        break
        
    #Preprocessing of frame
    (fw, fh) = frame.shape[:2]
    frame = cv2.resize(frame, (500, 500*fw/fh))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Face detection
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    if(len(faces) == 0):                        #no face detected
        if(len(facesLoc) > 0):                  #any previous face data
            ###NEED TO CHECK STANDARD DEVIATION TO FIND FACE
            print("Previous face present")
            print("No face detected")
            continue
        else:
            print("No face detected")
            continue
    else:                                       #face detected
        for (x, y, w, h) in faces:
            
            #Storing face coordinated
            if(len(facesLoc) > 1):              #only 2 face data is required
                del facesLoc[0]
            facesLoc.append({'x': x, 'y': y, 'w': w, 'h': h})
    
    #Processing the face
    for face in facesLoc:
        x = face['x']
        y = face['y']
        w = face['w']
        h = face['h']
        roi_gray = gray[y:y+h, x:x+w]
        
        #Eye Detection
        eyes = eyeCascade.detectMultiScale(roi_gray)
        if(len(eyes) == 0):                     #no eye detected
            
            #Predict Eye Location
            if(len(facesLoc) > 1):              
                if(len(eyesLoc) > 0):
                    for i in range(len(eyesLoc)):
                        ex = eyesLoc[0]['x'] - (facesLoc[0]['x'] - x)
                        ey = eyesLoc[0]['y'] - (facesLoc[0]['y'] - y)
                        ewh = int(h/4)        
                        eyesLoc.append({'x': ex, 'y': ey, 'w': ewh, 'h': ewh})
                        del eyesLoc[0]
                else:
                    print("No eye detected")
                    break
            else:
                print("No eye detected")
                break
        else:                                   #eye detected
            eyesLoc = []                        #flushing the buffer
            eyeIndex = 0
            for (ex, ey, ew, eh) in eyes:
                if eyeIndex >= 2:
                    break
                eyesLoc.append({'x': ex, 'y': ey, 'w': (h/4), 'h': (h/4)})
                eyeIndex += 1

        #Processing the eye
        percents = []
        for eye in eyesLoc:
            ex = eye['x']
            ey = eye['y']
            ew = eye['w']
            eh = eye['h']
            
            #Preprocessing
            eye_gray = roi_gray[ey+(eh*0.30):ey+(eh*0.70), ex+(ew*0.15):ex+(ew*0.85)]
            (h1, w1) = eye_gray.shape[:2]
            eye_gray = cv2.resize(roi_gray_e, (100*w1/h1, 100))
            (h1, w1) = eye_gray.shape[:2]
            if(ex > (w/3)):                     #left/right eye
                i = 1
            else:
                i = 0
                
#*****Now we have an gray eye image[left(1:real)/right(0:real)] with height 100*****#
            
            #*******For white part of eye********#
            
            #Histogram Equlization
            gray_e = cv2.equalizeHist(eye_gray)
            
            #Thresholding
            ret, gray_t = cv2.threshold(gray_e,240,255,cv2.THRESH_BINARY_INV)
            gray_t = 255-gray_t                 #Inversion
            
            #Eliminate unwanted white pixels in the boarder
            for k in range(0, h1):
                for l in range(0, w1):
                    if(k<10 or k >= (h1-3)):
                        clearFrame() 
