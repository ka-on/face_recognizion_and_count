import cv2
import os
import numpy as np

cascPath = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cascPath = os.path.dirname(cv2.__file__)+"/data/haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
while True:

    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
    display_text = "Number of faces detected: {}".format(len(faces))
    coords = (50, frames.shape[1]-200)
    cv2.putText(frames,display_text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

video_capture.release()
cv2.destroyAllWindows()
