import numpy as np
import cv2
import matplotlib.pyplot as plt

eyeCascade = cv2.CascadeClassifier('haar/haarcascade_eye_tree_eyeglasses.xml')
faceCascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')

for n in range(1, 22):
    print("\n\n************** %d **************" %n)
    img = cv2.imread(str(n) + ".pgm", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        k = 1
        for (ex, ey, ew, eh) in eyes:
            print("\n\n***** %d *****" %k)
            roi_gray_eye = roi_gray[ey:ey+eh, ex:ex+ew]
            for val in range(0, 255):
                ret, gray_t = cv2.threshold(roi_gray_eye, val, 255, cv2.THRESH_BINARY_INV)
                gray_t = 255-gray_t
                (gw, gh) = gray_t.shape[:2]
                start,end,maxIn,opened,dataFlag,storeS,storeE,x = 0, 0, 0, 0, 0, 0, 0, 0
                img_row_sum = np.sum(gray_t,axis=1).tolist()
                for p in range(len(img_row_sum)):
                    if img_row_sum[p] - x > 0 and start == 0:
                        opened = 1
                        start = p-1
                    elif img_row_sum[p] - x <= 0 and opened == 1:
                        end = p
                        opened = 0
                        dataFlag = 1
                    if end == len(img_row_sum)-1:
                        x += 100
                        p = 0
                        start = 0
                        end = 0
                        continue
                    if dataFlag == 1:
                        diff = end - start
                        if start == -1 or start == 0 or (float(diff)*100)/float(len(img_row_sum)-1) > 50:
                            x += 100
                            start = 0
                            end = 0
                            p =0
                            continue
                        if maxIn <= diff and diff != 0:
                            maxIn = diff
                            storeS = start
                            storeE = end
                            start = 0
                            end = 0
                if maxIn == 0:
                    continue
                for p in range(len(img_row_sum)):
                    if p < storeS or p > storeE:
                        img_row_sum[p] = 0
                sumOfWhite = sum(img_row_sum)/255
                sumOfBlack = (gh*gw) - sumOfWhite
                for i in range(0,gh):
                    if i < storeS or i > storeE:
                        for j in range(0,gw):
                            gray_t[i][j] = 0
                try:
                    prnt = float(float(sumOfBlack)/float(sumOfWhite))
                except:
                    continue
                if prnt < 1.5:
                    continue
                else:
                    print(str(prnt) + ' : ' + str(val) + ' : ' + str(storeS) + ' : ' + str(storeE))
                cv2.imshow("image_"+str(val), gray_t)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                plt.plot(img_row_sum)
                plt.show()
            k+=1
