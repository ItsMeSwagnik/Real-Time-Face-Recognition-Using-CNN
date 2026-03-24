# Real Time Face Recognition Using CNN

A real-time face recognition system built with a Convolutional Neural Network (CNN) using TensorFlow/Keras and OpenCV. The system detects faces via webcam, trains a custom CNN model on collected face data, and performs live recognition with confidence scoring.

---

## Key Features

- **Live face detection** using Haar Cascade classifier
- **Custom CNN model** trained on your own face data
- **Unknown face detection** with configurable confidence threshold
- **Prediction smoothing** via rolling average over 5 frames to reduce flickering
- **Automatic face saving** — detected faces are saved to `detected_faces/`
- **Multi-class support** — recognizes multiple people
- **Automatic loss function selection** — switches between binary and categorical crossentropy based on number of classes

---

## System Requirements

- Windows 10/11
- Python 3.10 – 3.12 (Python 3.12 recommended)
- Webcam
- Minimum 4GB RAM
- No GPU required (CPU inference supported)

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Face detection and webcam capture |
| `tensorflow` | CNN backend |
| `keras` | Model building and training |
| `numpy` | Array operations |
| `scikit-learn` | Label encoding |
| `matplotlib` | Optional plotting |

---

## Installation

**1. Clone the repository**
```bash
git clone <repository-url>
cd "Real Time Face Recognition Using CNN"
```

**2. Create a virtual environment with Python 3.12**
```bash
py -3.12 -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
pip install tensorflow
```

> Note: `tensorflow-intel` in `requirements.txt` is a placeholder. Install `tensorflow` separately as shown above.

---

## Usage Guide

### Step 1 — Collect face data
```bash
python src/data_collection/face_dataset.py
```
- Enter your name when prompted
- Press `C` to capture a face image
- Capture at least 50 images with varied angles and lighting
- The window closes automatically after 50 captures
- Images are saved to `dataset/<your_name>/`

### Step 2 — Train the model
```bash
python src/training/face_training.py
```
- Automatically detects and crops faces from dataset images
- Trains the CNN for 10 epochs
- Saves the model to `models/trained_model.h5`
- Saves class labels to `models/classes.npy`

### Step 3 — Run live recognition
```bash
python src/recognition/face_recognition.py
```
- Opens webcam and starts detecting faces in real time
- Displays name and confidence percentage on screen
- Press `ESC` to exit

---

## Configuration

### Confidence threshold (`src/recognition/face_recognition.py`)
```python
label = "Unknown" if confidence < 0.6 else classes[np.argmax(avg_pred)]
```
- Increase `0.6` (e.g. `0.75`) for stricter recognition — fewer false positives
- Decrease `0.6` (e.g. `0.5`) if your face is being shown as Unknown too often

### Prediction smoothing buffer (`src/recognition/face_recognition.py`)
```python
pred_buffer = deque(maxlen=5)
```
- Increase `maxlen` for more stable but slower label updates
- Decrease `maxlen` for faster but more flickery updates

### Face detection sensitivity (`src/recognition/face_recognition.py`)
```python
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60,60))
```
- Increase `minNeighbors` to reduce false detections
- Decrease `minSize` to detect smaller/farther faces

### Training epochs (`src/training/face_training.py`)
```python
model.fit(data, labels, epochs=10)
```
- Increase epochs (e.g. `20`) if recognition accuracy is low

---

## Project Structure

```
Real Time Face Recognition Using CNN/
├── dataset/                        # Training images, one folder per person
│   └── <person_name>/
│       ├── 0.jpg
│       └── ...
├── models/                         # Saved model and class labels
│   ├── trained_model.h5
│   └── classes.npy
├── detected_faces/                 # Faces captured during live recognition
├── src/
│   ├── data_collection/
│   │   └── face_dataset.py         # Webcam face capture script
│   ├── training/
│   │   └── face_training.py        # CNN model training script
│   └── recognition/
│       └── face_recognition.py     # Live face recognition script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tips for Better Accuracy

- Collect images in the same lighting conditions you'll use for recognition
- Include varied angles (slightly left, right, up, down)
- Collect at least 100 images per person for better accuracy
- Add an `Unknown` folder with random face images from the internet to improve unknown detection
- Retrain the model whenever you add new people to the dataset

---

## License

This project is licensed under the MIT License. See below for details.

```
MIT License

Copyright (c) 2026 Swagnik Ganguly

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Privacy Policy

- This system processes **live webcam footage** solely for the purpose of face recognition
- **No video or audio is recorded or transmitted** to any external server or service
- Face images saved to `detected_faces/` are stored **locally on your machine only**
- Training data stored in `dataset/` remains **entirely under your control**
- This software does **not collect, share, or sell** any personal data
- It is your responsibility to obtain consent from any individuals whose face data is used for training

> **Warning:** Deploying this system to recognize individuals without their knowledge or consent may violate privacy laws in your jurisdiction (e.g. GDPR, CCPA). Use responsibly and ethically.

---

## Copyright

© 2026 Swagnik Ganguly. All rights reserved.

Unauthorized copying, distribution, or modification of this project without explicit permission from the author is prohibited.
