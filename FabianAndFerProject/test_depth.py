import cv2
import numpy as np
import time
import threading
import pyttsx3
import os
import sys
from cvzone.HandTrackingModule import HandDetector
from face_detector_dnn import FaceDetectorDNN

# ==========================================
# 0. FILE CHECK
# ==========================================
midas_path = "FabianAndFerProject/model-small.onnx"
if not os.path.exists(midas_path):
    print(f"ERROR: Could not find file '{midas_path}'.")
    print("Make sure it is in the same folder as this script.")
    sys.exit()

# ==========================================
# 1. INITIALIZE AUDIO ENGINE
# ==========================================
print("Initializing Audio Engine...")
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_beep(volume_level):
    def _speak():
        engine.setProperty('volume', float(volume_level))
        engine.say("Beep")
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# ==========================================
# 2. INITIALIZE MODELS (ALL VIA OPENCV)
# ==========================================
print("Loading MiDaS (via OpenCV DNN)...")
midas_net = cv2.dnn.readNet(midas_path)
print("MiDaS loaded successfully.")

print("Loading Face Detector (Person Depth)...")
face_detector = FaceDetectorDNN(
    model_path="res10_300x300_ssd_iter_140000.caffemodel",
    config_path="deploy.prototxt"
)
print("Face Detector loaded successfully.")

print("Loading Hand Detector (Sign Depth)...")
hand_detector = HandDetector(maxHands=1, detectionCon=0.8)
print("Hand Detector loaded successfully.")

# ==========================================
# 3. VARIABLES
# ==========================================
cap = cv2.VideoCapture(0)

last_wall_alert = 0
last_beep_time = 0
FOCAL_LENGTH_CONSTANT = 15000
MAX_HAND_WIDTH = 300.0

print("-" * 50)
print("SYSTEM ACTIVE: Running Hands, Faces, and Walls simultaneously.")
print("Press 'q' in the video window to exit.")
print("-" * 50)

while True:
    success, frame = cap.read()
    if not success:
        break

    current_time = time.time()
    display_frame = frame.copy()
    h_frame, w_frame = frame.shape[:2]

    # ==========================================
    # SYSTEM 1: MIDAS (WALLS AND OBJECTS) VIA OPENCV
    # ==========================================
    blob = cv2.dnn.blobFromImage(
        frame,
        1/255.0,
        (256, 256),
        (123.675, 116.28, 103.53),
        swapRB=True,
        crop=False
    )

    midas_net.setInput(blob)
    depth_map = midas_net.forward()

    depth_map = depth_map[0, 0]
    depth_map = cv2.resize(depth_map, (w_frame, h_frame))

    depth_map_visual = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )

    y1, y2 = int(h_frame * 0.4), int(h_frame * 0.6)
    x1, x2 = int(w_frame * 0.4), int(w_frame * 0.6)
    center_region = depth_map_visual[y1:y2, x1:x2]
    avg_wall_depth = np.mean(center_region)

    if (current_time - last_wall_alert) > 1.0:
        if avg_wall_depth > 180:
            print(f"WALL ALERT: Obstacle imminent (Score: {int(avg_wall_depth)})")
        last_wall_alert = current_time

    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        display_frame,
        f"Depth: {int(avg_wall_depth)}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    # ==========================================
    # SYSTEM 2: FACE DETECTOR
    # ==========================================
    faces = face_detector.detect(frame)
    for face in faces:
        fx, fy, fw, fh = face["bbox"]
        if fw > 0:
            person_distance = FOCAL_LENGTH_CONSTANT / fw

            cv2.rectangle(display_frame, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"Person distance: {int(person_distance)}",
                (fx, fy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

    # ==========================================
    # SYSTEM 3: HAND DETECTOR AND AUDIO
    # ==========================================
    hands, display_frame = hand_detector.findHands(display_frame, draw=True)

    if hands:
        hand = hands[0]
        hw = hand["bbox"][2]

        dynamic_volume = min(1.0, max(0.1, hw / MAX_HAND_WIDTH))

        if (current_time - last_beep_time) > 1.0:
            speak_beep(dynamic_volume)
            last_beep_time = current_time

            print(f"HAND DETECTED: Playing at {int(dynamic_volume * 100)}% volume")

    # ==========================================
    # DISPLAY
    # ==========================================
    cv2.imshow("Unified Echolocation System", display_frame)

    depth_colormap = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_INFERNO)
    small_depth = cv2.resize(depth_colormap, (w_frame // 2, h_frame // 2))
    cv2.imshow("MiDaS Thermal Radar", small_depth)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()