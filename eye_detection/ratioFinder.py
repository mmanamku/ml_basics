import numpy as np
import cv2

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
			    ret, gray_t = cv2.threshold(roi_gray_eye, val, 255, cv2.THRESH_BINARY)
			    gray_t = 255-gray_t
			    sumB = 0
			    sumW = 0
			    for i in gray_t:
				    for j in i:
					    if j == 0:
						    sumB += 1
					    else:
						    sumW += 1
			    try:
			        prnt = float(float(sumB)/float(sumW))
			    except:
			        continue
			    if sumW == 0:
			        continue
			    elif prnt < 1.5:
			        continue
			    else:
			        print(str(prnt) + ' : ' + str(val))
			    cv2.imshow("image_"+str(val), gray_t)
			    cv2.waitKey(0)
			    cv2.destroyAllWindows()
		    k+=1
