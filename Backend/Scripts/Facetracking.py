import numpy as np
import cv2

cap = cv2.VideoCapture(0) #Opens an output stream for video capture
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #This is the actual algorithm

while True: 
    ret, frame = cap.read() #Reads in the cap object, and returns data from video to the frame object

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.imshow("cropped", roi_color) #Tracks the cropped face
        cv2.imshow('frame', frame) #Tracks the full screen
  
    if cv2.waitKey(1) == ord('q'):
        break

cap.release() #Prevents mem leaks
cv2.destroyAllWindows() #Destroys all windows and clears screen