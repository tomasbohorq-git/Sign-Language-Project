import cv2
import time
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cvzone.HandTrackingModule import HandDetector

from person import PoseData
from tracker import PersonTracker
from associator import Associator
from face_detector_dnn import FaceDetectorDNN
from logger_jsonl import JsonlLogger

from gesture_classifier import HammingGestureClassifier

# CONFIG


MODEL_PATH = "FabianAndFerProject/pose_landmarker_lite.task"

NUM_POSES = 3              # Start stable with 2; increase later if CPU allows
MAX_HANDS = 8              # Up to 8 hands in the scene (multiple people)
FLIP_TYPE = True          # False = error with the hand labeling, left=right and viceversa

CAP_W, CAP_H = 1280, 720   # Camera capture resolution (best effort)
COUPLING_FACTOR = 0.35     # Hand->Pose coupling threshold (relative to shoulder distance)

LOG_PATH = "FabianAndFerProject/emmaeye_log.jsonl"


# Gesture reference matrix (G1–G16)
GESTURE_MATRIX = np.array([
    [1,0,0,1,1,1,1,1,1,0,0,0],
    [1,0,0,0,0,1,1,1,1,0,0,0],
    [0,0,0,0,0,1,1,1,1,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,1,1,0,0,0],
    [1,0,0,0,0,0,1,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1],
    [0,1,1,1,1,1,1,1,1,0,0,0],
    [0,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0],
    [0,0,0,1,1,1,1,1,1,0,0,0],
    [1,1,1,1,1,1,1,0,0,0,0,0],
    [1,0,0,0,0,1,1,1,1,0,1,0]
])



# HELPERS


def euclid2(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def lm_px(lm, w, h):
    return (lm.x * w, lm.y * h)


def build_pose_data(lm_list, w, h, coupling_factor):
    """
    Convert MediaPipe normalized landmarks into PoseData (pixel coords + useful points).
    Uses visibility to drop wrists/head if they are not reliable.
    """
    shL = lm_px(lm_list[11], w, h)
    shR = lm_px(lm_list[12], w, h)
    shoulder_dist = max(euclid2(shL, shR), 1.0)

    # Left/right wrist (only if visible enough)
    wrist_L = lm_px(lm_list[15], w, h) if getattr(lm_list[15], "visibility", 1.0) > 0.5 else None
    wrist_R = lm_px(lm_list[16], w, h) if getattr(lm_list[16], "visibility", 1.0) > 0.5 else None

    # Head proxy: nose landmark (index 0)
    head = lm_px(lm_list[0], w, h) if getattr(lm_list[0], "visibility", 1.0) > 0.5 else None

    # Pose center (mid-shoulder)
    center = ((shL[0] + shR[0]) / 2.0, (shL[1] + shR[1]) / 2.0)

    # Store all landmarks for logging
    full = []
    for lm in lm_list:
        vis = float(getattr(lm, "visibility", 1.0))
        full.append((lm.x * w, lm.y * h, vis))

    return PoseData(
        landmarks=full,
        center=center,
        left_wrist=wrist_L,
        right_wrist=wrist_R,
        head=head,
        shoulder_dist=shoulder_dist,
        coupling_thresh=coupling_factor * shoulder_dist
    )


def gesture_id_to_name(gid):
    if gid == -1:
        return "Unknown"
    if gid == 2:
        return "G2 (Peace)"
    if gid == 3:
        return "G3 (Fist)"
    if gid == 4:
        return "G4 (Open Hand)"
    if gid == 5:
        return "G5 (Thumbs Up)"
    # if gid == 6:
    # if gid == 7:
    # if gid == 8:
    # if gid == 9:
    if gid == 10:
        return "G7 (Thumbs Down)"
    # if gid == 11:
    # if gid == 12:
    # if gid == 13:
    # if gid == 14:
    # if gid == 15:
    # if gid == 16:
    
    return f"G{gid} (TODO)"

# INITIALIZATION


# PoseLandmarker (MediaPipe Tasks)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=NUM_POSES,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# Camera (Windows stable backend: DirectShow)
# Camera Initialization
print("Initializing camera...")

# Removed CAP_DSHOW to match your working test script
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Camera index 0 failed, trying index 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("No camera available. Close Zoom/Teams/Discord and try again.")

# Temporarily comment out the resolution forcing to prevent freezing
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

# Hand detector (cvzone)
hand_detector = HandDetector(detectionCon=0.8, maxHands=MAX_HANDS)

# Face detector (OpenCV DNN)
face_detector = FaceDetectorDNN(conf_threshold=0.5)

# Persistent tracker
tracker = PersonTracker(max_people=20, max_missed=100, match_thresh_px=200.0)

# Associator: hands + face -> person, with gesture stabilization
associator = Associator(gesture_classifier=HammingGestureClassifier(GESTURE_MATRIX, max_hamming=1))

#  Logger (JSONL)
logger = JsonlLogger(LOG_PATH)

print("EMMAeye (persistent IDs) running. Press ESC to exit.")
t0 = time.time()


last_t_ms = -1  


# MAIN LOOP


while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera frame read failed. Exiting.")
        break
        
    # Safety catch for empty warm-up frames
    if frame is None or frame.size == 0:
        continue

    h, w = frame.shape[:2]
    
    # Strictly increasing timestamp for MediaPipe
    t_ms = int((time.time() - t0) * 1000)
    if t_ms <= last_t_ms:
        t_ms = last_t_ms + 1
    last_t_ms = t_ms

    # Multi-pose detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pose_result = pose_landmarker.detect_for_video(mp_image, t_ms)

    detected_poses = []
    if pose_result.pose_landmarks:
        for lm_list in pose_result.pose_landmarks[:NUM_POSES]:
            detected_poses.append(build_pose_data(lm_list, w, h, COUPLING_FACTOR))

    # Update persistent IDs
    people = tracker.update(detected_poses)

    # Hands (cvzone) 
    hands, frame = hand_detector.findHands(frame, draw=True, flipType=FLIP_TYPE)
    hands_list = hands if hands else []

    #  Faces (OpenCV DNN)
    faces = face_detector.detect(frame)

    # Associate faces and hands to each person + gesture stabilization 
    associator.assign_hands_and_faces(people, hands_list, faces, t_ms)

    # Basic UI overlay
    for p in people:
        if not p.pose:
            continue

        cx, cy = int(p.pose.center[0]), int(p.pose.center[1])
        cv2.putText(frame, f"Person {p.id}", (cx - 50, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Face bbox
        if p.face:
            # Safely check if face is an object or dictionary
            x, y, bw, bh = p.face.bbox if hasattr(p.face, 'bbox') else p.face["bbox"]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        # Hands stable gesture text
        for side in ["BODY_LEFT", "BODY_RIGHT"]:
            hd = p.hands[side]
            if hd:
                # Safely get bbox
                x, y, bw, bh = hd.bbox if hasattr(hd, 'bbox') else hd["bbox"]
                
                # Safely get gesture from the object
                gesture = getattr(hd, 'stable_gesture', 'Unknown')

                cv2.putText(frame, f"{side} {gesture_id_to_name(gesture)}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    #  Log for later analysis
    try:
        logger.log_people(people)
    except Exception as e:
        # If the logger crashes on custom objects, it will print here instead of killing the app
        print(f"Logger Error: {e}")

    cv2.imshow("EMMAeye", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
