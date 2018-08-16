import numpy as np
import cv2
import cv2.cv as cv
import matplotlib.pyplot as plt

eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
#parser = argparse.ArgumentParser()
#parser.add_argument('-f', '--file', help='Path to file')
#args = vars(parser.parse_args())
#fileName = args['file']
#NEW = 1
centPer = [0, 0, 0, 0]
cap = cv2.VideoCapture('Datasets/videos/Face_2.mp4')
eyeIn = ["right", "left"]
while(cap.isOpened()):
    ret, frame = cap.read()
    (w, h) = frame.shape[:2]
    img = cv2.resize(frame,(500,500*w/h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print("No face")
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        percents = []
        i = 0
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
            roi_gray_e = roi_gray[ey:ey+eh, ex:ex+ew]
            ret,gray_t = cv2.threshold(roi_gray_e,80,255,cv2.THRESH_BINARY)
            gray_t = 255-gray_t
            #cv2.imshow(str(eyeIn[i])+"eye", gray_t)
            img_row_sum = np.sum(gray_t,axis=1).tolist()
            start,end,maxIn,opened = 0, 0, 0, 0
            for p in range(len(img_row_sum)):
                check = img_row_sum[p]-1000;
                if  check >= 0 and start == 0:
                    opened = 1
                    start = p-1
                elif start != 0 and check <= 0 and opened == 1:
                    end = p
                    opened = 0
                diff = end - start
                if maxIn <= diff:
                    maxIn = diff
            if centPer[i] <= maxIn:
                centPer[i] = maxIn
            print("***************\nStart: %d\nEnd: %d\nMaxIn: %d\ncentPer: %d" % (start, end, maxIn, centPer[i]))
            percents.append((diff*100)/centPer[i])
            cv2.imshow("Image", gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            #plt.plot(img_row_sum)
            #plt.show()
            i += 1
        if len(eyes) == 0:
            print("No eye")
        else:
            print(str(sum(percents)/len(percents)) + "%")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
