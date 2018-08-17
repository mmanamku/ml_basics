import numpy as np
import cv2
import cv2.cv as cv
import argparse

#For viewing vertical histogram | Debugging
import matplotlib.pyplot as plt

#for Passing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='Path to file')
args = vars(parser.parse_args())
fileName = args['file']             #name of the video file

#Flags and Threshold Values
NEW = 1                             #initialize to new entry
PRESET = [0, 0, 0, 0]               #normal open eye width value [eye1, eye2, err, err]
MOD = 0                             #normal open eye detect
FACE = 0                            #face present
EYE = 0                             #eye present
eyeDeviationLimit = 8               #% of change : detect normal eye open stage
deviationLimit = 0                  #detect motion of face
minThreshold = 20                   #optimise the threshold value
maxThreshold = 50                   #optimise the threshold value

#Parameters initialization
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
centPer = [0, 0, 0, 0]              #Cent Percent attained when width equals this [eye1, eye2, err, err]
perC = [100, 100, 100, 100]         #Current frame's percentage [eye1, eye2, err, err]
cap = cv2.VideoCapture(fileName)    #fileName from command line : 'Datasets/videos/Face_2.mp4'
prevEyePercent = [100, 100, 100, 100]    #previous percent for comparing with current percentage [eye1, eye2, err, err]
timer = 10                          #timer to find normal eye
globalTimer = 50                    #upper time limit to find normal eye
eyeLocation = {'x': 0, 'y': 0, 'w': 0, 'h': 0}      #initializing eye location
eyesLoc = []
faceLocation = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
facesLoc = []

while(cap.isOpened()):
    ret, frame = cap.read()
    
    #error rectification
    if(frame == None):
        break
    
    #preprocessing of image frame
    (w, h) = frame.shape[:2]
    img = cv2.resize(frame,(500,500*w/h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #face detection from image frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print("No face detected")
    for (x,y,w,h) in faces:
        if(len(facesLoc) > 1):
            del facesLoc[0]
        faceLocation = {'x': x, 'y': y, 'w': w, 'h': h}
        facesLoc.append()

        #eye detection from face image
        roi_gray = gray[y:y+h, x:x+w]               #crop region of interest
        eyes = eye_cascade.detectMultiScale(roi_gray)
        percents = []                               #keeping percentage of both eyes
        i = 0                                       #points left (1) and right (0) eyes
        for (ex,ey,ew,eh) in eyes:
            if i >= 2:                                 #morethan 2 eye detected
                break
            eyesLoc = []
            eyeLocation = {'x': ex, 'y': ey, 'w': ew, 'h': eh}
            eyesLoc.append()
                
            #thresholding
            roi_gray_e = roi_gray[ey:ey+eh, ex:ex+ew]       #crop the eye
            ret,gray_t = cv2.threshold(roi_gray_e,80,255,cv2.THRESH_BINARY)     #threshold the eye image
            gray_t = 255-gray_t                     #Invert the graph
            
            #finding vertical projection histogram
            img_row_sum = np.sum(gray_t,axis=1).tolist()    #row wise sum
            
            #finding width of projection
            start,end,maxIn,opened = 0, 0, 0, 0
            for p in range(len(img_row_sum)):
                check = img_row_sum[p];
                if  check > 0 and start == 0:       #beginning of peak
                    opened = 1
                    start = p-1
                elif start != 0 and check <= 0 and opened == 1:     #end of peak
                    end = p
                    opened = 0
                diff = end - start
                if maxIn <= diff and diff != 0:                     #max width in the graph
                    maxIn = diff
                    start = 0
                    end = 0
                    
            #Initializing the normal open eye width
            if PRESET[i] == 0:                      #checking initialization
                centPer[i] = maxIn
                PRESET[i] = 1
                
            #finding percentage of eye closure
            perC[i] = (maxIn*100)/centPer[i]
            percents.append(perC[i])
            
            #updating the normal open eye width, if user was initialized with another value
            if MOD == 0:                            #normal open eye finding
                if abs(perC[i] - prevEyePercent[i]) >= eyeDeviationLimit :
                    if timer == 0:
                        centPer[i] = maxIn
                        MOD = 1
                else:
                    timer = 10
                timer -= 1
                
            
            cv2.imshow("Image", gray)
            i += 1
        if len(eyes) == 0:
            #print("No eye detected")
            if(len(facesLoc) > 0):
                faceLocDiff = {'x': facesLoc[0]['x']-faceLocation['x'], 'y': facesLoc[0]['y']-faceLocation['y'], 'w': facesLoc[0]['w']-faceLocation['w'], 'h': facesLoc[0]['h']-faceLocation['h']}
                eyeLoc1Temp = {'x': eyesLoc[0]['x']+faceLocDiff['x'], 'y': eyesLoc[0]['y']+faceLocDiff['y'], 'w': eyesLoc[0]['w']+faceLocDiff['w'], 'h': eyesLoc[0]['h']+faceLocDiff['h']}
                eyeLoc2Temp = {'x': eyesLoc[1]['x']+faceLocDiff['x'], 'y': eyesLoc[1]['y']+faceLocDiff['y'], 'w': eyesLoc[1]['w']+faceLocDiff['w'], 'h': eyesLoc[1]['h']+faceLocDiff['h']}
            
        else:
            print(str(sum(percents)/len(percents)) + "%")                        
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
