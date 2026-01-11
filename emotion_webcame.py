import cv2
import numpy as np
import time
import json
from tensorflow.keras.models import load_model

# Load model
model = load_model("emotion_detection_model.h5")

# Emotion labels
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Starting webcam...")

TIMEOUT = 10
start_time = time.time()

last_emotion = "unknown"   # <-- ALWAYS STORED

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)

        preds = model.predict(face_roi, verbose=0)
        idx = np.argmax(preds)
        emotion = class_names[idx]
        conf = np.max(preds)

        last_emotion = emotion.lower()   # <-- ALWAYS UPDATED

        cv2.putText(frame, f"{emotion} ({conf*100:.1f}%)", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0,255,0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    # manual close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # auto timeout
    if time.time() - start_time >= TIMEOUT:
        print("[INFO] Timeout. Closing camera.")
        break

cap.release()
cv2.destroyAllWindows()

# ALWAYS SAVE EMOTION
try:
    with open("current_emotion.json", "w") as f:
        json.dump({"emotion": last_emotion}, f)
    print("[SAVED] Emotion saved:", last_emotion)
except:
    print("[ERROR] Could NOT save emotion!")
