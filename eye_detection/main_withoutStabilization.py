import numpy as np
import cv2
import cv2.cv as cv
import argparse
import matplotlib.pyplot as plt

#Command line argument passing
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='path to file, leave if camera')
args = vars(parser.parse_args())

#Flags, constants and Thresholds
PRESET = [0,0]
MODIFIED = [[0,0,0],[0,0,0]]
PREVINIT = [[0,0,0],[0,0,0]]
OBSEYE = [[0,0,0],[0,0,0]]
font = cv2.FONT_HERSHEY_SIMPLEX
eyeDeviationLimit = [8,8,20]

#Initialization of variables
eyeCascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')
faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
eyesLoc = []                                    #list of eyes
facesLoc = []                                   #list of faces
realOpen = [[0,0,0],[0,0,0]]
realClose = [[0,0,0],[0,0,0]]
currentEye = [[0,0,0],[0,0,0]]
obsEyePercent = [[0,0,0],[0,0,0]]
timer = [[10,10,10],[10,10,10]]
globalTimer = 50

#Debug | Reference
frameC = 0                                      #frame Number
percentTrackerZZ = []
percentTrackerZO = []
percentTrackerZT = []
percentTrackerOZ = []
percentTrackerOO = []
percentTrackerOT = [] 

#Functions
def clearFrame(image, x, y):                    #eliminate unwanted part iteratively
    (h, w) = image.shape[:2]
    if image[x][y] == 255:
        image[x][y] = 0
    if (x+1) < h and image[x+1][y] == 255:
        clearFrame(image, x+1, y)
    if (x-1) >= 0 and image[x-1][y] == 255:
        clearFrame(image, x-1, y)
    if (y+1) < w and image[x][y+1] == 255 :
        clearFrame(image, x, y+1)
    if (y-1) >= 0 and image[x][y-1] == 255:
        clearFrame(image, x, y-1)
    if (y+1) < w and (x+1) < h and image[x+1][y+1] == 255:
        clearFrame(image, x+1, y+1)
    if (y-1) >= 0 and (x+1) < h and image[x+1][y-1] == 255:
        clearFrame(image, x+1, y-1)
    if (y+1) < w and (x-1) >= 0 and image[x-1][y+1] == 255:
        clearFrame(image, x-1, y+1)
    if (y-1) >= 0 and (x-1) >= 0 and image[x-1][y-1] == 255:
        clearFrame(image, x-1, y-1)
       
def eliminateBoarderData(image, left, right, top, bottom):
    (h, w) = image.shape[:2]
    for k in range(0, h):
        for l in range(0, w):
            if(k<top or k >= (h-bottom)):       #defining boundary
                clearFrame(image, k, l)
            else:
                if(l<left or l >= (w-right)):
                    clearFrame(image, k, l)
                    
def maxWidth(image, vertHist):
    start,end,maxIn,opened,dataFlag,storeS,storeE = 0,0,0,0,0,0,0
    for p in range(len(vertHist)):
        if vertHist[p] > 0 and opened == 0:
            opened = 1
            start = p-1
        elif vertHist[p] <= 0 and opened == 1:
            end = p
            opened = 0
            dataFlag = 1
        if dataFlag == 1:
            diff = end - start
            if maxIn <= diff and diff != 0:
                maxIn = diff
                storeS = start
                storeE = end
                start = 0
                end = 0
    for p in range(len(vertHist)):              #limiting the frame to max width region
        if p < storeS or p > storeE:
            vertHist[p] = 0
            image[p] = 0
    return maxIn
    
def storePercent(currentValue, eye, option):
    try:
        currentEye[eye][option] = ((currentValue-realClose[eye][option])*100)/(realOpen[eye][option]-realClose[eye][option])
    except:
        currentEye[eye][option] = 0

def modifyOpen(currentValue, eye, option):
    if MODIFIED[eye][option] == 0:
        if PREVINIT[eye][option] != 0:
            if (abs(currentEye[eye][option] - obsEyePercent[eye][option]) >= eyeDeviationLimit[option]):
                if timer[eye][option] == 0:
                    realOpen[eye][option] = currentValue
                    MODIFIED[eye][option] = 1
                else:
                    if OBSEYE[eye][option] == 0:
                        obsEyePercent[eye][option] = currentEye[eye][option]
                        OBSEYE[eye][option] = 1
            else:
                timer[eye][option] = 10
                OBSEYE[eye][option] = 0
            timer[eye][option] -= 1
        else:
            obsEyePercent[eye][option] = currentEye[eye][option]
            PREVINIT[eye][option] = 1
            
#Capture Video
if not args.get('file', False):
    cap = cv2.VideoCapture(0)                   #camera
else:
    cap = cv2.VideoCapture(args['file'])        #video
    
#Processing each frame
while True:
    ret, frame = cap.read()
    
    #Debug | Reference
    frameC += 1
    
    #No frame
    if(frame == None):
        print("End of the Video Stream")
        break
        
    #Preprocessing of frame
    (fh, fw) = frame.shape[:2]
    frame = cv2.resize(frame, (500, 500*fh/fw))
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
    face = facesLoc.pop()
    facesLoc.append(face)
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
                continue
        else:
            print("No eye detected")
            continue
    else:                                   #eye detected
        eyesLoc = []                        #flushing the buffer
        eyeIndex = 0
        for (ex, ey, ew, eh) in eyes:
            if eyeIndex >= 2:
                break
            eyesLoc.append({'x': ex, 'y': ey, 'w': (h/4), 'h': (h/4)})
            eyeIndex += 1

    #Processing the eye
    for eye in eyesLoc:
        ex = eye['x']
        ey = eye['y']
        ew = eye['w']
        eh = eye['h']
        
        #Preprocessing
        eye_gray = roi_gray[ey+(eh*0.30):ey+(eh*0.70), ex+(ew*0.15):ex+(ew*0.85)]
        (h1, w1) = eye_gray.shape[:2]
        eye_gray = cv2.resize(eye_gray, (100*w1/h1, 100))
        (h1, w1) = eye_gray.shape[:2]
        if(ex > (w/3)):                     #left/right eye
            i = 1
        else:
            i = 0
            
#*****Now we have a gray eye image[left(1:real)/right(0:real)] with height 100*****#
        
        #*******For white part of eye********#
        
        #Histogram Equlization
        gray_e = cv2.equalizeHist(eye_gray)
        
        #Thresholding
        ret, gray_et = cv2.threshold(gray_e,240,255,cv2.THRESH_BINARY_INV)
        gray_et = 255-gray_et             #inversion
        
        #Eliminate unwanted white pixels in the boarder
        eliminateBoarderData(gray_et, 3, 3, 10, 3)
                    
        #Finding Vertical Histogram
        gray_et_row_sum = np.sum(gray_et, axis=1).tolist()
        
        #Find sum of white
        sumOfWhite = sum(gray_et_row_sum)/255
        currentEye[i][2] = (sumOfWhite*100)/(w1*h1) 
        
        #Finding max width of vertical histogram
        max_et = maxWidth(gray_et, gray_et_row_sum)
                            
        #Percentage of eye openness using sum of white pixels in updated img
        #sumOfWhite2 = sum(gray_et_row_sum)/255
        #percentUsingSW2 = (sumOfWhite2*100)/(w1*h1)
                
        #*******For black part of eye********#
        
        #Adaptive Histogram Equilization using clahe
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_c = clahe.apply(eye_gray)
        
        #Thresholding the equalized image
        ret, gray_ct = cv2.threshold(gray_c,40,255,cv2.THRESH_BINARY)
        gray_ct = 255-gray_ct
        
        #Eliminate unwanted black pixels in the boarder
        eliminateBoarderData(gray_ct, 3, 3, 3, 3)
        
        #Finding Vertical Histogram
        gray_ct_row_sum = np.sum(gray_ct, axis=1).tolist()
        
        #Finding max width of vertical histogram
        max_ct = maxWidth(gray_ct, gray_ct_row_sum)
        
        #Initialization of open and closed eyes
        if PRESET[i] == 0:
            realOpen[i][0] = max_ct
            realOpen[i][1] = max_et
            realOpen[i][2] = sumOfWhite
            realClose[i][0] = 0.4*realOpen[i][0]
            realClose[i][1] = 0.4*realOpen[i][1]
            realClose[i][2] = 0
            PRESET[i] = 1
            
        #Percentage of one eye
        storePercent(max_ct, i, 0)
        storePercent(max_et, i, 1)
        storePercent(sumOfWhite, i, 2)
        
        #Update normal close eye
        if currentEye[i][0] < 0:
            currentEye[i][0] = 0
            if currentEye[i][1] < 15 or currentEye[i][2] < 15:
                realClose[i][0] = max_ct
        if currentEye[i][1] < 0:
            currentEye[i][1] = 0
            if currentEye[i][0] < 15 or currentEye[i][2] < 15:
                realClose[i][1] = max_et
        
        #Modifying open eye with timer
        if globalTimer != 0:
            globalTimer -= 1
            modifyOpen(max_ct, i, 0)
            modifyOpen(max_et, i, 1)
            modifyOpen(sumOfWhite, i, 2)   
        
    #Debug
    #if cv2.waitKey(1) & 0xFF == ord('p'):
    #    for eye in range(len(eyesLoc)):
    #        cv2.imshow("EyeClahe"+str(eye), gray_ct[eye])
    #        plt.plot(gray_ct_row_sum[eye])
    #        plt.show()
    #        cv2.imshow("EyeEqu"+str(eye), gray_et[eye])
    #        plt.plot(gray_et_row_sum[eye])
    #        plt.show()
    #        cv2.destroyAllWindows()
    
    #output
    percentTrackerZZ.append(currentEye[0][0])
    percentTrackerZO.append(currentEye[0][1])
    percentTrackerZT.append(currentEye[0][2])
    percentTrackerOZ.append(currentEye[1][0])
    percentTrackerOO.append(currentEye[1][1])
    percentTrackerOT.append(currentEye[1][2])
    cv2.putText(gray,str(frameC), (10,20), font, 0.5, (0,0,0), 1)
    cv2.putText(gray,str(currentEye[0][0])+'% : '+str(currentEye[1][0])+'%', (10,40), font, 0.5, (0,0,0), 1)
    cv2.putText(gray,str(currentEye[0][1])+'% : '+str(currentEye[1][1])+'%', (10,60), font, 0.5, (0,0,0), 1)
    cv2.putText(gray,str(currentEye[0][2])+'% : '+str(currentEye[1][2])+'%', (10,80), font, 0.5, (0,0,0), 1)
    cv2.imshow("Frame", gray)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

     
plt.plot(percentTrackerZZ)
plt.show()
plt.plot(percentTrackerZO)
plt.show()
plt.plot(percentTrackerZT)
plt.show()
plt.plot(percentTrackerOZ)
plt.show()
plt.plot(percentTrackerOO)
plt.show()
plt.plot(percentTrackerOT)
plt.show()

cap.release()
cv2.destroyAllWindows()
