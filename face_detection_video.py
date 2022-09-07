import cv2
import numpy as np

video_cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    # get the next frame, resize it, and convert it to grayscale
    succes, frame = video_cap.read()
    frame = cv2.resize(frame, (640, 640))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_rects = face_detector.detectMultiScale(gray, 1.04, 5, minSize=(30, 30))
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    # wait for 1 milliseconde and if the q key is pressed, we break the loop
    if cv2.waitKey(1) == ord("q"):
        break
    
# release the video capture and close all windows
video_cap.release()
cv2.destroyAllWindows()