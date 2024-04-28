import cv2
import numpy as np

direction = input("Please enter video path : ")
cap = cv2.VideoCapture(direction)
face_cascade = cv2.CascadeClassifier('../opencv/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#Above haar cascade points to the file inside the opencv library in our git installation. We really should containarize things but idk
#how to do that

out_s = 0 #Output size of the frame. This will be contsant
out = 0 #Output size of the actual video, this is dynamic
buffer_percent = 0.3
#Ok so this is a weird solution. In the facetracking script, there was the problem of the frame dynamically
#changing to ensure the box is kept on the persons face. Here, instead, there is a "slider precentage" that 
#adjusts the buffer of the video accordingly. Setthing this to 1 has a generous buffer around the face,
#setting it to 0 leaves no buffer. I reccomend about 0.3, which looks natural. Also keep the value between
#1 and 0, as larger values distort the output and spit out a large file

# Read the video frame by frame. Recall opencv is a frame by frame workflow
while cap.isOpened():
    ret, frame = cap.read() #frame stored, ret is the return value and is boolean. Frame is the image itself
    if not ret:
        break #error handling

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Algorithm works with greyscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Tune the algorithm itself. feel free to play with this

    if len(faces) > 0: #This is to stave off against multi-face videos. Since the video "focuses" on one face, having
        #multiple faces would cause the focus to be split off between various frames. In this case, though, we just take
        #the first face. Otherwise it goes to black

        x, y, w, h = faces[0] #takes the coordinates from the first face it detects
        
        buf_x = int(buffer_percent * w) #this is added later to the frame to add a buffer zone 
        buf_y = int(buffer_percent * h) #likewise

        # Adjust coordinates to include buffer, ensuring they stay within frame bounds
        x_min = max(x - buf_x, 0) #Prevents x-buf_x negative values from causing odd interactions
        y_min = max(y - buf_y, 0) #likewise
        x_max = min(x + w + buf_x, frame.shape[1]) #.shape[1] returns num of columns in pixels
        y_max = min(y + h + buf_y, frame.shape[0]) #.shape[0] returns num of rows in pixels

        #The above creates new x and y coordinates that replace x,y,w,h. This is what determines the size of the frame

        #Up until here, out_s shouldn't be touched, but incase previous use didn't destroy data for some reason
        if out_s == 0:
            out_s = (x_max - x_min, y_max - y_min)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('CutVideo.mp4', fourcc, 20.0, out_s)

        #The above creates the frame of the video, essentially the container that actually writes the video. Opencv
        #lets you export a lot of formats which is neat. mp4v seems the most convinent. Unsure about the rest of the
        #parameters, just ripped off the documentation as reccomended and it works 

        #the x,y values, now set by the algorithm, are ready to cut the frame into an appropriate size
        face_frame = frame[y_min:y_max, x_min:x_max]
        face_frame = cv2.resize(face_frame, out_s) 
        #Ensures that the face frame fits what the algorithm detected as a face. Prevents jitters and mismatch

        # Write the frame. Congratulations, one down, 1799 frames to go. Video processing is very resource intensive
        out.write(face_frame)
    else:
        # Throws black to prevent weird jitters
        if out != 0:
            black_frame = np.zeros((out_s[1], out_s[0], 3), dtype="uint8")
            out.write(black_frame)

cap.release()
if out != 0:
    out.release()
cv2.destroyAllWindows()
#Releases all resources