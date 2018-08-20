import numpy as np
import cv2
import cv2.cv as cv
import argparse
import matplotlib.pyplot as plt

#Command line argument passing
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='Path to file')
args = vars(parser.parse_args())
fileName = args['file']         #path to video file

#Flags and Threshold values for comparison
MODIFIED = [0, 0]               #normalization of eye state [left, right]
PREVINIT = [0, 0]            #if eye state has a previous value
PRESET = [0, 0]                    #if eye state initialized
eyeDeviationLimit = 8           #Used for normalization of eye state; represented as % 

#Initialization of variables
realOpen = [0, 0]                #normal open    [left, right]
realClose = [60, 60]            #normal closed  [left, right]
eyeCascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')
faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
currentEye = [0, 0]             #current eye state
prevEyePercent = [0, 0]         #previous eye state
timer = 10                        #timer for modification
globalTimer = 50                #utmost time for modification
eyesLoc = []                    #list of eye locations
faceLocation = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
facesLoc = []                    #list of face locations

#Capture Video
cap = cv2.VideoCapture(fileName)

while(cap.isOpened()):
    ret, frame = cap.read()
    
    #no frame
    if(frame == None):
        break
    
    #preprocessing of image frame
    (w, h) = frame.shape[:2]
    img = cv2.resize(frame, (500, 500*w/h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #face detection
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    if(len(faces) == 0):
        if(len(facesLoc) > 0):         #looking for previous face
            '''need to check standard deviation of frames, once found, append to faces and remove break'''
            print("Previous face present")
            print("No face detected")
            break
        else:
            print("No face detected")
            break
    for (x, y, w, h) in faces:
        
        #storing face coordinates
        if(len(facesLoc) > 1):
            del facesLoc[0]             #remove unwanted datas
        faceLocation = {'x': x, 'y': y, 'w': w, 'h': h}
        facesLoc.append(faceLocation)
        print("facesLoc:")
        print(facesLoc)
        
        #eye detection
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        percents = []
        
        #if no eyes found, predict it's location
        if(len(eyes) == 0):
            if(len(facesLoc) > 1):
                if(len(eyesLoc) > 0):
                    for i in range(len(eyesLoc)):
                        print("current eye taken %d" %(i))
                        ex = eyesLoc[i]['x'] - (facesLoc[0]['x'] - faceLocation['x'])
                        ey = eyesLoc[i]['y'] - (facesLoc[0]['y'] - faceLocation['y'])
                        ew = eyesLoc[i]['w'] + eyesLoc[i]['w']*((faceLocation['w'] - facesLoc[0]['w'])/facesLoc[0]['w']) #% change
                        eh = eyesLoc[i]['h'] + eyesLoc[i]['h']*((faceLocation['h'] - facesLoc[0]['h'])/facesLoc[0]['h']) #% change
                        print("Predicated eye %d:" %(i))
                        print('{x: ' + str(ex) + ', y: ' + str(ey) + ', w: ' + str(ew) + ', h:' + str(eh) + '}')
                        eyesLoc.append({'x': ex, 'y': ey, 'w': ew, 'h': eh})
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
                eyesLoc.append({'x': ex, 'y': ey, 'w': ew, 'h': eh})
                print("eyesLoc %d:" %(i))
                print(eyesLoc)
                i += 1
        i = 0
        cv2.imshow("Name", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for eye in eyesLoc:
            roi_gray_e = roi_gray[eye['y']:eye['y']+eye['h'], eye['x']:eye['x']+eye['w']]
            ret,gray_t = cv2.threshold(roi_gray_e,80,255,cv2.THRESH_BINARY)
            gray_t = 255-gray_t
            start,end,maxIn,opened, dataFlag = 0, 0, 0, 0, 0
            img_row_sum = np.sum(gray_t,axis=1).tolist()
            for p in range(len(img_row_sum)):
                if img_row_sum[p] > 0 and start == 0:
                    opened = 1
                    start = p-1
                elif img_row_sum[p] <= 0 and opened == 1:
                    end = p
                    opened = 0
                    dataFlag = 1
                if dataFlag == 1 or p == len(img_row_sum)-1 :
                    diff = end - start
                    if maxIn <= diff and diff != 0:
                        maxIn = diff
                        start = 0
                        end = 0
            
            print("*****Eye %d*****\nstart: %d\nend: %d\nmaxIn: %d\n" %(i, start, end, maxIn))
            plt.plot(img_row_sum)
            plt.show()
                        
            if PRESET[i] == 0:
                realOpen[i] = maxIn
                print("realOpen[%d]: %d\n" %(i, realOpen[i]))
                PRESET[i] = 1
            
            realClose[i] = 0.6*realOpen[i]
            print("realClose[%d]: %f\n" %(i, realClose[i]))
            currentEye[i] = ((maxIn-realClose[i])*100)/(realOpen[i]-realClose[i])
            if currentEye[i] < 0:
                realClose[i] = maxIn
                currentEye[i] = ((maxIn-realClose[i])*100)/(realOpen[i]-realClose[i])
            percents.append(currentEye[i])
            
            print("realClose[%d]: %f\ncurrentEye[%d]: %f\n" %(i, realClose[i], i, currentEye[i]))
              
            if MODIFIED[i] == 0 and globalTimer != 0:
                if (PREVINIT[i] != 0):
                    if (abs(currentEye[i] - prevEyePercent[i]) >= eyeDeviationLimit):
                        if timer == 0:
                            realOpen[i] = maxIn
                            print("realOpen[%d]: %d\n" %(i, realOpen[i]))
                            MODIFIED[i] = 1
                    else:
                        timer = 10
                        print("timer reset\n")
                    timer -= 1
                    globalTimer -= 1
                    prevEyePercent[i] = currentEye[i]
                else:
                    prevEyePercent[i] = currentEye[i]
                    print("previous eye set\n")
                    PREVINIT[i] = 1
            i += 1
        print(str(sum(percents)/len(percents))+'%')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
