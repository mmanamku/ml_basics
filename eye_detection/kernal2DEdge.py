import cv2
import numpy as np

cap = cv2.VideoCapture("video1.mkv")

while True:
    ret, frame = cap.read()
    (w, h) = frame.shape[:2]
    img = cv2.resize(frame, (500, 500*w/h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray,100,200)
    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = cv2.filter2D(gray, -1, kernel)
    new = gray-im
    #new = gray + edge
    newC = gray-im
    #newC = gray + edgeC
    newL = gray-im
    #newL = gray + edgeL
    newS = gray-im
    #newS = gray + edgeS
    cv2.imshow("image", new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("imageC", newC)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("imageL", newL)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("imageS", newS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ret,thresh = cv2.threshold(new,127,255,cv2.THRESH_BINARY)
    ret,threshC = cv2.threshold(newC,127,255,cv2.THRESH_BINARY)
    ret,threshL = cv2.threshold(newL,127,255,cv2.THRESH_BINARY)
    ret,threshS = cv2.threshold(newS,127,255,cv2.THRESH_BINARY)
    cv2.imshow("image", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("imageC", threshC)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("imageL", threshL)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("imageS", threshS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
