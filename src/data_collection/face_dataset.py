import cv2
import os

name = input("Enter your name: ")
path = f"dataset/{name}"

if not os.path.exists(path):
    os.makedirs(path)

cam = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite(f"{path}/{count}.jpg", frame)
        count += 1
        print(f"Captured {count}")

    if count >= 100:
        break

cam.release()
cv2.destroyAllWindows()