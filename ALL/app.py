from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def detect_faces():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cropped_face = frame[y:y+h, x:x+w]
            canvas = np.zeros((frame.shape[0], frame.shape[1] + cropped_face.shape[1], 3), dtype=np.uint8)
            canvas[:frame.shape[0], :frame.shape[1]] = frame
            canvas[:cropped_face.shape[0], frame.shape[1]:] = cropped_face
            ret, buffer = cv2.imencode('.jpg', canvas)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
