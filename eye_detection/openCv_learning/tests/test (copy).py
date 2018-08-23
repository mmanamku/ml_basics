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
PREVINIT = [0, 0]               #if eye state has a previous value
PRESET = [0, 0]                 #if eye state initialized
OBSEYE = 0                      #Eye for observation
eyeDeviationLimit = 8           #Used for normalization of eye state; represented as % 

#Initialization of variables
realOpen = [0, 0]               #normal open    [left, right]
realClose = [60, 60]            #normal closed  [left, right]
eyeCascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')
faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
currentEye = [0, 0]             #current eye state
obsEyePercent = [0, 0]          #previous eye state
timer = 10                      #timer for modification
globalTimer = 50                #utmost time for modification
eyesLoc = []                    #list of eye locations
faceLocation = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
facesLoc = []                   #list of face locations

#Debug
frameC = 0                      #frame count
maxIns = [0, 0]                 #maximum values found in each frames
#wholeSum = 0
#wholeMax = [0, 0]
#wholeMin = [0, 0]
#wholePer = [0, 0]
#obsEyePercent2 = [0, 0]          #previous eye state
#timer2 = 10                      #timer for modification
#OBSEYE2 = 0                      #Eye for observation
#MODIFIED2 = [0, 0]               #normalization of eye state [left, right]

#Capture Video
cap = cv2.VideoCapture(fileName)

while(cap.isOpened()):
    ret, frame = cap.read()
    frameC += 1
    
    #if frameC < 160 :
     #   continue
    
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
        if(len(facesLoc) > 0):          #looking for previous face
            '''need to check standard deviation of frames, once found, append to faces and remove break'''
            print("Previous face present")
            print("No face detected")
            break
        else:
            print("No face detected")
            break
    for (x, y, w, h) in faces:
        
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        
        #storing face coordinates
        if(len(facesLoc) > 1):
            del facesLoc[0]             #remove unwanted datas
        faceLocation = {'x': x, 'y': y, 'w': w, 'h': h}
        facesLoc.append(faceLocation)
        #print("facesLoc:")
        #print(facesLoc)
        
        #eye detection
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        percents = []
        #percents2 = []
        
        #if no eyes found, predict it's location
        if(len(eyes) == 0):
            if(len(facesLoc) > 1):
                if(len(eyesLoc) > 0):
                    for i in range(len(eyesLoc)):
                        ex = eyesLoc[0]['x'] - (facesLoc[0]['x'] - faceLocation['x'])
                        ey = eyesLoc[0]['y'] - (facesLoc[0]['y'] - faceLocation['y'])
                        ew = eyesLoc[0]['w'] + eyesLoc[0]['w']*float(float(faceLocation['w'] - facesLoc[0]['w'])/float(facesLoc[0]['w'])) #% change
                        eyesLoc.append({'x': ex, 'y': ey, 'w': int(ew), 'h': int(ew)})
                        cv2.rectangle(roi_gray,(ex,ey),(ex+int(ew),ey+int(ew)),(255,0,0),2)
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
                cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                i += 1
        i = 0
        for eye in eyesLoc:
            roi_gray_e = roi_gray[eye['y']:eye['y']+eye['h'], eye['x']:eye['x']+eye['w']]
            ret,gray_t = cv2.threshold(roi_gray_e,80,255,cv2.THRESH_BINARY)
            gray_t = 255-gray_t
            #gray_t = cv2.resize(gray_t, (100, 100))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            start,end,maxIn,opened, dataFlag, sumOf = 0, 0, 0, 0, 0, 0
            img_row_sum = np.sum(gray_t,axis=1).tolist()
            
            for p in range(len(img_row_sum)):
                if img_row_sum[p] > 0 and start == 0:
                    opened = 1
                    start = p-1
                    #sumOf += img_row_sum[p]
                elif img_row_sum[p] <= 0 and opened == 1:
                    end = p
                    opened = 0
                    dataFlag = 1
                if dataFlag == 1 or p == len(img_row_sum)-1 :
                    diff = end - start
                    if maxIn <= diff and diff != 0:
                        maxIn = diff
                      #  wholeSum = sumOf
                        start = 0
                        end = 0
                       # sumOf = 0
                        
            maxIns[i] = maxIn            
            if PRESET[i] == 0:
                realOpen[i] = maxIn
                #wholeMax[i] = wholeSum
                PRESET[i] = 1
                realClose[i] = 0.6*realOpen[i]
                #wholeMin[i] = 0.6*wholeMax[i]
            currentEye[i] = ((maxIn-realClose[i])*100)/(realOpen[i]-realClose[i])
            #print("\n*********eye%d*********" %(i+1))
            #print(wholeSum)
            #print(wholeMin[i])
            #print(wholeMax[i])
            #wholePer[i] = ((wholeSum-wholeMin[i])*100)/(wholeMax[i]-wholeMin[i])
            #print(wholePer[i])
            cv2.imshow("Name", gray_t)
            if currentEye[i] < 0:
                realClose[i] = maxIn
                currentEye[i] = ((maxIn-realClose[i])*100)/(realOpen[i]-realClose[i])
            #if wholePer[i] < 0:
            #    wholeMin[i] = wholeSum
            #    wholePer[i] = ((wholeSum-wholeMin[i])*100)/(wholeMax[i]-wholeMin[i])
            percents.append(currentEye[i])
           # percents2.append(wholePer[i])
            
            if MODIFIED[i] == 0 and globalTimer != 0: #''' and MODIFIED2[i] == 0 '''
                if (PREVINIT[i] != 0):
                    if (abs(currentEye[i] - obsEyePercent[i]) >= eyeDeviationLimit) and MODIFIED[i] == 0:
                        if timer == 0:
                            realOpen[i] = maxIn
                            MODIFIED[i] = 1
                        else:
                            if OBSEYE == 0:     #need modifcation, change of eye state again not included
                                obsEyePercent[i] = currentEye[i]
                                OBSEYE = 1
                    else:
                        timer = 10
                    
                    '''if (abs(wholePer[i] - obsEyePercent2[i]) >= eyeDeviationLimit) and MODIFIED2[i] == 0:
                        if timer2 == 0:
                            wholeMax[i] = wholeSum
                            MODIFIED2[i] = 1
                        else:
                            if OBSEYE2 == 0:     #need modifcation, change of eye state again not included
                                obsEyePercent2[i] = wholePer[i]
                                OBSEYE2 = 1
                    else:
                        timer2 = 10
                    timer2 -= 1'''
                    
                    timer -= 1
                    globalTimer -= 1
                else:
                    obsEyePercent[i] = currentEye[i]
                    #obsEyePercent2[i] = wholePer[i]
                    PREVINIT[i] = 1
            i += 1
        #cv2.imshow("Name", gray)
        #print(str(frameC) + " : " + str(int(min(percents))) + '% : ' + str(int(sum(percents2)/len(percents2))) + '%')
        print(str(frameC) + " : " + str(int(min(percents))) + '%')
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
