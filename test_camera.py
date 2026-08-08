import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Download model if not present
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model... please wait...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded successfully.")

# Hand connections for drawing lines between dots
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_landmarks(frame, hand_landmarks_list):
    h, w = frame.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 200, 255), 2)
    return frame

# Setup detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Start camera — 0 = laptop webcam
cap = cv2.VideoCapture(0)
print("Camera started. Show your hand. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found. Check connection.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        frame = draw_landmarks(frame, result.hand_landmarks)
        cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Show your hand...",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("ISL Project - Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()