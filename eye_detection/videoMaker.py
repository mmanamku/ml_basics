import cv2
import os

video_name = 'video.avi'

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if(frame == None):
        break
    (height, width) = frame.shape[:2]
    video = cv2.VideoWriter(video_name, -1, 1, (width,height))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	    
cap.release()
cv2.destroyAllWindows()
video.release()
