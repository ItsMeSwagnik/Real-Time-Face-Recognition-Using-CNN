import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

data = []
labels = []

for person in os.listdir("dataset"):
    for img_name in os.listdir(f"dataset/{person}"):
        img = cv2.imread(f"dataset/{person}/{img_name}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            continue
        (x, y, w, h) = faces[0]
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (100,100))
        data.append(img)
        labels.append(person)

data = np.array(data) / 255.0

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

n_classes = len(lb.classes_)
activation = 'sigmoid' if n_classes == 1 else 'softmax'
loss = 'binary_crossentropy' if n_classes == 1 else 'categorical_crossentropy'

model = Sequential([
    Input(shape=(100,100,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(n_classes, activation=activation)
])

model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

model.fit(data, labels, epochs=10)

model.save("models/trained_model.h5")
np.save("models/classes.npy", lb.classes_)