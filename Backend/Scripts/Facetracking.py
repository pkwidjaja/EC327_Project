import cv2
import numpy as np

cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

#OpenCV gives you tools to capture data. VideoCapture accesses the webcam. 0 is the first value, so if you have multiple
#cams theyd likely be 1,2,3. 

face_cascade = cv2.CascadeClassifier('../opencv/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#Starts the algorithm/accesses it.

if face_cascade.empty():
    print("Error in cascade")
    exit()
# Error handling
    

# To prevent the jitter problem, we set the frame size. Note this fits the entire image into whatever frame size you specify.
#Stick to square values, with identical width and height, to prevent stretching the image. Adjust as you see fit
output_size = (500, 500)  

#Ok so this is a weird solution. Previously, there was the problem of the Haar cascade zooming in too tighttly
#on the persons face, making it uncomfortable. Here, its a slider percentage instead, which
#adjusts the buffer of the video accordingly. Setthing this to 1 has a generous buffer around the face,
#setting it to 0 leaves no buffer. Keep the value between 1 and 0, as larger values distort the output.
buffer_percentage = 0.5


# Read the video frame by frame. Recall opencv is a frame by frame workflow
while True:
    ret, frame = cap.read() #frame stored, ret is the return value and is boolean. Frame is the image itself
    if not ret:
        break #error handling

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Algorithm works with greyscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Tunes the algorithm. Feel free to adjust this if you want to test accuracy

    if len(faces) > 0: 
        #This is to stave off against multi-face videos. Since the video "focuses" on one face, having
        #multiple faces would cause the focus to be split off between various frames. In this case, though, we just take
        #the first face. Otherwise it goes to black
        x, y, w, h = faces[0]
        
        buffer_x = int(buffer_percentage * w)#this is added later to the frame to add a buffer zone 
        buffer_y = int(buffer_percentage * h)#likewise

        # Adjust coordinates to include margin, ensuring they stay within frame bounds
        x_min = max(x - buffer_x, 0) #Prevents x-buf_x negative values from causing odd interactions
        y_min = max(y - buffer_y, 0)
        x_max = min(x + w + buffer_x, frame.shape[1]) #.shape[1] returns num of columns in pixels
        y_max = min(y + h + buffer_y, frame.shape[0])#.shape[0] returns num of rows in pixels

        #the x,y values, now set by the algorithm, are ready to cut the frame into an appropriate size
        face_frame = frame[y_min:y_max, x_min:x_max]

        #Ensures that the face frame fits what the algorithm detected as a face. Prevents jitters and mismatch
        face_frame = cv2.resize(face_frame, output_size)

        # Write the frame. Congratulations, one down, 1799 frames to go. Video processing is very resource intensive
        cv2.imshow('Cropped Face', face_frame)
    else:
        # We were running into issues where if the face was undetected for whatever reasonm, the video stream lags or
        #just gives up. I looked for a solution online and found this, which generates an entirelyblack frame when 
        #the length of faces is 0, ie nothing detected. This makes it so theres constant streaming isntead of
        #it cutting off every time it doesnt detect a face
        black_frame = np.zeros(output_size, dtype="uint8")
        cv2.imshow('Cropped Face', black_frame)

    #Break loop when user wants
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Releases all resources
cap.release()
cv2.destroyAllWindows()
