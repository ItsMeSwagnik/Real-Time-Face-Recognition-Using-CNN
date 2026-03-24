import cv2
import numpy as np
import os
from collections import deque
from keras.models import load_model

model = load_model("models/trained_model.h5")
classes = np.load("models/classes.npy")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if not os.path.exists("detected_faces"):
    os.makedirs("detected_faces")

img_count = 0
pred_buffer = deque(maxlen=5)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60,60))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))
        face_norm = face_resized / 255.0
        face_input = np.reshape(face_norm, (1, 100, 100, 3))

        pred = model.predict(face_input, verbose=0)
        pred_buffer.append(pred[0])
        avg_pred = np.mean(pred_buffer, axis=0)
        confidence = np.max(avg_pred)

        label = "Unknown" if confidence < 0.6 else classes[np.argmax(avg_pred)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imwrite(f"detected_faces/face_{img_count}.jpg", face_resized)
        img_count += 1

    cv2.imshow("Face Recognition CNN", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()