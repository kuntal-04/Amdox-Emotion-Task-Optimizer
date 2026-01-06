import cv2
import os

def detect_face_emotion(image_path):
    if not os.path.exists(image_path):
        return "Image file not found"

    image = cv2.imread(image_path)

    if image is None:
        return "Image could not be read"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return "No face detected"

    # Placeholder visual emotion (CV logic)
    return "stressed"
