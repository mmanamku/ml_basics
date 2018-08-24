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

#Flags and Threshold values for comparison
MODIFIED = [0, 0]               #normalization of eye state             [left, right]
PREVINIT = [0, 0]               #if eye state has a previous value      [left, right]
PRESET = [0, 0]                 #if eye state initialized               [left, right]
OBSEYE = 0                      #eye for observation
eyeDeviationLimit = 8           #for normalization of eye state;        % 
thresholdLowerLimit = 3.2       #to find the optimal threshold value;   B-W Ratio
thresholdUpperLimit = 10        #to find the optimal threshold value;   B-W Ratio

#Initialization of font
font                    = cv2.FONT_HERSHEY_SIMPLEX
location                = (10,20)
fontScale               = 0.5
fontColor               = (0,0,0)
lineType                = 1

#Initialization of variables
eyeCascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')
faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
realOpen = [0, 0]               #normal open                            [left, right]
realClose = [60, 60]            #normal closed                          [left, right]
currentEye = [0, 0]             #current eye state                      [left, right]
obsEyePercent = [0, 0]          #previous eye state                     [left, right]
timer = [10, 10]                #timer for modification; currently no. of frames
globalTimer = 50                #utmost time for modification; currently no. of frames
eyesLoc = []                    #list of eye locations
faceLocation = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
facesLoc = []                   #list of face locations

#Debug
frameC = 0                      #frame count
maxIns = [0, 0]                 #maximum values found in each eyes
percentTracker = []
widthTracker = []
DECREASING = 0
DONE = [0, 0]
BackUp = 0

#Capture Video
if not args.get('file', False):
    cap = cv2.VideoCapture(0)                   #camera
else:
    cap = cv2.VideoCapture(args['file'])        #video

#Begin
while True:
    ret, frame = cap.read()
    
    #Debug
    frameC += 1
    
    #No frame
    if(frame == None):
        break
    
    #Preprocessing of frame
    (w, h) = frame.shape[:2]
    frame = cv2.resize(frame, (500, 500*w/h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    
    #Face detection
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    if(len(faces) == 0):
        if(len(facesLoc) > 0):          #looking for previous face
        
            #Debug          #need to check standard deviation of frames, once found, append to faces and remove break
            print("Previous face present")
            print("No face detected")
            continue
        else:
            print("No face detected")
            continue
    for (x, y, w, h) in faces:
        
        #Debug
        #cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        
        #Storing face coordinates
        if(len(facesLoc) > 1):
            del facesLoc[0]             #remove unwanted datas
        faceLocation = {'x': x, 'y': y, 'w': w, 'h': h}
        facesLoc.append(faceLocation)
        
        #Eye detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray1 = gray[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        percents = []
        
        #No eye Found? Predict eye location
        if(len(eyes) == 0):
            if(len(facesLoc) > 1):
                if(len(eyesLoc) > 0):
                    for i in range(len(eyesLoc)):
                        ex = eyesLoc[0]['x'] - (facesLoc[0]['x'] - faceLocation['x'])
                        ey = eyesLoc[0]['y'] - (facesLoc[0]['y'] - faceLocation['y'])
                        eyesLoc.append({'x': ex, 'y': ey, 'w': int((h/4)), 'h': int((h/4))})
                        del eyesLoc[0]
                        
                        #Debug
                        #cv2.rectangle(roi_gray,(ex,ey),(ex+int((h/4)),ey+int((h/4))),(255,0,0),2)
                else:
                    print("No eye present")
                    break
            else:
                print("No eye present")
                break
                
        #Eye found
        else:
            eyesLoc = []
            i = 0
            for (ex, ey, ew, eh) in eyes:
                if i >= 2:
                    break
                eyesLoc.append({'x': ex, 'y': ey, 'w': (h/4), 'h': (h/4)})
                i += 1
                        
                #Debug
                #cv2.rectangle(roi_gray,(ex,ey),(ex+(h/4),ey+(h/4)),(255,0,0),2)
                
        #Heart part
        i = 0
        for eye in eyesLoc:
            
            #Preprocessing
            roi_gray_e = roi_gray1[eye['y']:eye['y']+eye['h'], eye['x']:eye['x']+eye['w']]
            roi_gray_e = cv2.resize(roi_gray_e, (100, 100*w/h))
            if(eye['x'] > (faceLocation['w']/3)):       #fixing which eye
                i = 1
            else:
                i = 0
                
            #Debug
            #print(eye['x'], (faceLocation['w'])/3 , i)
            
            #Thresholding
            ret,gray_t = cv2.threshold(roi_gray_e,80,255,cv2.THRESH_BINARY)
            gray_t = 255-gray_t
            (gw, gh) = gray_t.shape[:2]
            #cv2.imshow("Eye", roi_gray_e)
            #cv2.waitKey(0)
            #cv2.imshow("Eye", gray_t)
            #cv2.waitKey(0)
            
            #Finding vertical histogram and max width
            img_row_sum = np.sum(gray_t,axis=1).tolist()
            start,end,maxIn,opened,dataFlag,storeS,storeE,x = 0, 0, 0, 0, 0, 0, 0, 0
            for p in range(len(img_row_sum)):
                if img_row_sum[p] - x > 0 and start == 0:
                    opened = 1
                    start = p-1
                elif img_row_sum[p] - x <= 0 and opened == 1:
                    end = p
                    opened = 0
                    dataFlag = 1
                #if end == len(img_row_sum)-1:
                #    x += 100
                #    p = 0
                #    start = 0
                #    end = 0
                #    continue
                if dataFlag == 1:
                    diff = end - start
                    #if start == -1 or (float(diff)*100)/float(len(img_row_sum)) > 60):
                    #    x += 100
                    #    start = 0
                    #    end = 0
                    #    p =0
                    #    continue
                    if maxIn <= diff and diff != 0:
                        maxIn = diff
                        storeS = start
                        storeE = end
                        start = 0
                        end = 0
                        
            #Eliminate unwanted regions and find B-W ratio
            for p in range(len(img_row_sum)):
                if p < storeS or p > storeE:
                    img_row_sum[p] = 0
                else:
                    img_row_sum[p] += x
            sumOfWhite = sum(img_row_sum)/255
            sumOfBlack = (gh*gw) - sumOfWhite                
            maxIns[i] = maxIn            
            
            #Initialization of open and closed eye
            if PRESET[i] == 0:
                realOpen[i] = maxIn
                PRESET[i] = 1
                realClose[i] = 0.6*realOpen[i]
                
            #Percentage of one eye
            currentEye[i] = ((maxIn-realClose[i])*100)/(realOpen[i]-realClose[i])
            
            #Updating close eye
            if currentEye[i] < 0:
                realClose[i] = maxIn
                currentEye[i] = ((maxIn-realClose[i])*100)/(realOpen[i]-realClose[i])
            percents.append(currentEye[i])
            
            #Debug
            print(i, storeS, storeE, maxIn, currentEye[i])
            #plt.plot(img_row_sum)
            #plt.show()
            
            #Modifiying open eye with timer
            if MODIFIED[i] == 0 and globalTimer != 0:
                if (PREVINIT[i] != 0):              #previous eye present
                    if (abs(currentEye[i] - obsEyePercent[i]) >= eyeDeviationLimit):    #eye deviation exceed a limit
                        if timer[i] == 0:              #deviation occured through out the timer
                            BackUp = realOpen[i]
                            realOpen[i] = maxIn
                            MODIFIED[i] = 1
                        else:
                            if OBSEYE == 0:         #need modifcation, change of eye state again not included
                                obsEyePercent[i] = currentEye[i]
                                OBSEYE = 1
                                
                        #Debug
                        DECREASING = 1
                    else:                           #no deviation, reset timer
                        timer[i] = 10
                        
                        #Debug
                        DECREASING = 0
                    
                    timer[i] -= 1
                    globalTimer -= 1
                else:                               #initialization of previous eye
                    obsEyePercent[i] = currentEye[i]
                    PREVINIT[i] = 1
            if MODIFIED[i] == 0 and globalTimer != 0 and DECREASING == 1:
                cv2.putText(gray, str(timer[i]), (10,40), font, fontScale, fontColor, lineType) 
            elif MODIFIED[i] == 1 and DONE[i] == 0:
                cv2.putText(gray, "Eye" + str(i) + ': ' + str(realOpen[i]) + ' (' + str(BackUp) + ')', (10,60), font, fontScale, fontColor, lineType)
                DONE[i] = 1
            
            #Debug
            if cv2.waitKey(1) & 0xFF == ord('p'):
                plt.plot(img_row_sum)
                plt.show()
            
        #Debug
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        percentTracker.append(min(percents))
        #widthTracker.append(eyesLoc[0]['w'] - ((faceLocation['w'])/4))
        
        #Output
        cv2.putText(gray ,str(frameC) + " : " + str(int(min(percents))) + '% : ' + str(faceLocation['w']), location, font, fontScale, fontColor, lineType)  
        cv2.imshow("Image", gray)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #print(str(int(min(percents))) + '%')
    if cv2.waitKey(1000/24) & 0xFF == ord('q'):
        break
        
#Debug        
plt.plot(percentTracker)
plt.show()
#plt.plot(widthTracker)
#plt.show()

#End
cap.release()
cv2.destroyAllWindows()
