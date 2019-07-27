# Importing libraries
#import pandas
import numpy as np
import cv2

# Initiating the video capture
cap = cv2.VideoCapture(0)

while(True):
    #font = cv2.FONT_HERSHEY_SIMPLEX
    ret, frame = cap.read()

    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
