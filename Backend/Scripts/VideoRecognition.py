import cv2  

cap = cv2.VideoCapture('SteveJobsCut.mp4')
face_cascade = cv2.CascadeClassifier('../opencv/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#The above points to the harrcascade algorithm folder. It points to the algorithm inside the opencv git that
#was imported. Ship the files as a bunch.

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # CV has a video write method that outputs various codecs
out = cv2.VideoWriter('CutVideo.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
#

# Read the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Frame 2 Greyscale for algorithm to work properly
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply algorithm and detected faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

    # Write frame to video file
    out.write(frame)

#Basically this code goes FRAME BY FRAME, treating each frame as an indiviual image. Whats really cool about this is
#you do not need to learn anything new to video related processing, as it is literally just images over and over
#again, analysis being repeated thousands of times for a single video.

# Release
cap.release()
out.release()
cv2.destroyAllWindows()
