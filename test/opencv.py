# Importing libraries
#import pandas
import numpy as np
import cv2
import os

# Getting required directories
# Model Directory
mod_dir = "/".join(os.getcwd().split("/")[0:-1] + ['model/'])

# Initiating the video capture
cap = cv2.VideoCapture(0)

# Importing the haar cascade models for front face any eyes

face_cascade = cv2.CascadeClassifier(mod_dir+'haarcascade_frontalface_default.xml')
eyes_cascade = eye_cascade = cv2.CascadeClassifier(mod_dir+'haarcascade_eye.xml')


while(True):
    #font = cv2.FONT_HERSHEY_SIMPLEX
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detecting faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Getting boundaries for the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detecting eyes inside every face
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Getting boundaries for every eye
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
