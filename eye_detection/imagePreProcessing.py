import cv2

cap = cv2.VideoCapture("video1.mkv")

while True:
    ret, frame = cap.read()
    (w, h) = frame.shape[:2]
    img = cv2.resize(frame, (500, 500*w/h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("resize", gray)
    #cv2.waitKey(0)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #cv2.imshow("blur", blur)
    #cv2.waitKey(0)
    smooth = cv2.addWeighted(blur,1.5,gray,-0.5,0)
    edgeS = gray-smooth
    newS = gray + edgeS
    cv2.imshow("smooth", newS)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
