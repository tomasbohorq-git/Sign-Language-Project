import cv2
import math
import time
import numpy as np
import mediapipe as mp
import logging

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cvzone.HandTrackingModule import HandDetector
from person import PoseData
from tracker import PersonTracker
from associator import Associator
from face_detector_dnn import FaceDetectorDNN
from gesture_classifier import HammingGestureClassifier
from audio_manager import GestureAudioManager


# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)

MODEL_PATH = "FabianAndFerProject/pose_landmarker_lite.task"

NUM_POSES      = 3       # Start stable with 2; increase later if CPU allows
MAX_HANDS      = 8       # Up to 8 hands in the scene (multiple people)
FLIP_TYPE      = True    # False = error with the hand labeling, left=right and viceversa

CAP_W, CAP_H   = 1280, 720  # Camera capture resolution (best effort)
COUPLING_FACTOR = 0.35       # Hand->Pose coupling threshold (relative to shoulder dist)

# ── Depth / echolocation (from CamTest) ──────────────────────
WALL_ALERT_THRESHOLD_MM = 600   # Warn if obstacle closer than 1 m
ALERT_COOLDOWN_SEC      = 1.0    # Min seconds between wall-alert prints
MAX_VOLUME_DIST         = 300.0  # 30 cm  -> 100 % volume
MIN_VOLUME_DIST         = 2000.0 # 200 cm -> 10 % volume
VOLUME_FADE_RANGE       = 0.9    # Dynamic range between those extremes

# Audio

# Enables clipping audio to the center or the edges of the frame of the camara.
# This supports an idea that seperating sounds more clearly might make the learning curve easier for users.
CLIP_AUDIO = True

# ── Gesture reference matrix (G1-G16) ────────────────────────
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


# ──────────────────────────────────────────────────────────────
# DEPTH HELPERS  (ported from CamTest.py)
# ──────────────────────────────────────────────────────────────

def get_physical_xyz(xyz_mat, x:int, y:int):
    """
    Query a ZED XYZ point cloud map at pixel (x, y).
    Returns absolute Euclidean distance in millimetres, or None on invalid reading.
    """
    import pyzed.sl as sl
    err, point_cloud_value = xyz_mat.get_value(x, y)
    
    if err == sl.ERROR_CODE.SUCCESS:
        x_val, y_val, z_val, _ = point_cloud_value
        
        # Guard against invalid point cloud data
        if (not math.isnan(x_val) and not math.isinf(x_val) and
            not math.isnan(y_val) and not math.isinf(y_val) and
            not math.isnan(z_val) and not math.isinf(z_val)):
            
            return (x_val, y_val, z_val)
                
    return None

def get_euclidean_depth_zed(xyz_mat, x: int, y: int):
    """
    Query a ZED XYZ point cloud map at pixel (x, y).
    Returns absolute Euclidean distance in millimetres, or None on invalid reading.
    """
    physical_xyz = get_physical_xyz(xyz_mat, x, y)
    if physical_xyz:            
        # Calculate absolute 3D Euclidean distance
        distance = math.sqrt(physical_xyz[0]**2 + physical_xyz[1]**2 + physical_xyz[2]**2)

        if distance > 0:
            return distance

    return None

def distance_to_volume(dist_mm: float) -> float:
    """
    Map a real-world distance (mm) to an audio volume between 0.1 and 1.0.
    Linear interpolation between MAX_VOLUME_DIST and MIN_VOLUME_DIST.
    Clamps outside the defined range.
    """
    if dist_mm <= MAX_VOLUME_DIST:
        return 1.0
    if dist_mm >= MIN_VOLUME_DIST:
        return 0.1
    normalized = (dist_mm - MAX_VOLUME_DIST) / (MIN_VOLUME_DIST - MAX_VOLUME_DIST)
    return 1.0 - (normalized * VOLUME_FADE_RANGE)


# ──────────────────────────────────────────────────────────────
# POSE HELPERS
# ──────────────────────────────────────────────────────────────

def euclid2(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def lm_px(lm, w, h):
    return (lm.x * w, lm.y * h)


def build_pose_data(lm_list, w, h, coupling_factor):
    """
    Convert MediaPipe normalised landmarks into PoseData (pixel coords + derived points).
    Uses visibility to drop wrists/head if they are not reliable.
    """
    shL = lm_px(lm_list[11], w, h)
    shR = lm_px(lm_list[12], w, h)
    shoulder_dist = max(euclid2(shL, shR), 1.0)

    wrist_L = lm_px(lm_list[15], w, h) if getattr(lm_list[15], "visibility", 1.0) > 0.5 else None
    wrist_R = lm_px(lm_list[16], w, h) if getattr(lm_list[16], "visibility", 1.0) > 0.5 else None
    head    = lm_px(lm_list[0],  w, h) if getattr(lm_list[0],  "visibility", 1.0) > 0.5 else None
    center  = ((shL[0] + shR[0]) / 2.0, (shL[1] + shR[1]) / 2.0)

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


# ──────────────────────────────────────────────────────────────
# GESTURE LABEL HELPERS
# ──────────────────────────────────────────────────────────────

def print_gesture(gid: int) -> str:
    labels = {
        2:  "G2 (Peace)",
        3:  "G3 (Fist)",
        4:  "G4 (Open Hand)",
        5:  "G5 (Thumbs Up)",
        10: "G10 (Thumbs Down)",
    }
    if gid == -1:
        return "Unknown"
    return labels.get(gid, f"G{gid} (TODO)")


def gid_to_sign_name(gid: int) -> str:
    sign_names = {
        2:  "Victory",
        4:  "Stop",
        10: "Dislike",
        5:  "Okay",
        # TODO: implement Point
    }
    return sign_names.get(gid, "UNKNOWN")


# ──────────────────────────────────────────────────────────────
# Audio Helpers
# ──────────────────────────────────────────────────────────────

def clip_position_to_points(pos, target_angles_deg=[0.0, -30.0, 30.0]):
    """
    Snaps a 3D position to the nearest angular plane in the XZ axis.
    
    Args:
        pos: Tuple of (x, y, z) coordinates.
        target_angles_deg: List of angles in degrees to snap to. 
                           0 is straight ahead (the Z axis).
                           Positive values are to the right, negative to the left.
    """
    x, y, z = pos
    
    # 1. Get the horizontal distance from the camera (radius in XZ plane)
    # math.hypot calculates sqrt(x^2 + z^2) safely
    r_xz = math.hypot(x, z)
    
    # 2. Calculate the current angle in degrees
    # atan2(x, z) assumes Z is forward, X is right/left.
    current_angle = math.degrees(math.atan2(x, z))
    
    # 3. Find the closest target angle from the provided list
    closest_angle = min(target_angles_deg, key=lambda a: abs(a - current_angle))
    
    # 4. Calculate the new X and Z using the snapped angle
    snapped_rad = math.radians(closest_angle)
    
    new_x = r_xz * math.sin(snapped_rad)
    new_z = r_xz * math.cos(snapped_rad)
    
    # 5. Return the new position, keeping the original "true up down" (Y)
    return (new_x, y, new_z)

# ──────────────────────────────────────────────────────────────
# CAMERA INITIALISATION  (ZED-first, webcam fallback)
# ──────────────────────────────────────────────────────────────

def _try_init_zed():
    """
    Attempt to open the ZED 2i with performance-optimised settings.
    Returns (zed, image_mat, depth_mat) on success,
    or (None, None, None) if the ZED SDK is absent or the camera unavailable.
    """
    try:
        import pyzed.sl as sl
    except ImportError:
        print("[Camera] pyzed not found - skipping ZED init.")
        return None, None, None

    zed    = sl.Camera()
    params = sl.InitParameters()
    params.depth_mode        = sl.DEPTH_MODE.NEURAL
    params.coordinate_units  = sl.UNIT.MILLIMETER
    params.camera_resolution = sl.RESOLUTION.HD720
    params.camera_fps        = 30

    err = zed.open(params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[Camera] ZED open failed ({err}) - falling back to webcam.")
        return None, None, None

    print("[Camera] ZED 2i ready - real metric depth enabled.")
    return zed, sl.Mat(), sl.Mat()


print("Initializing camera...")
zed, image_zed, xyz_zed = _try_init_zed()
USE_ZED = zed is not None

if not USE_ZED:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera index 0 failed, trying index 1...")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("No camera available. Close Zoom/Teams/Discord and try again.")
    print("[Camera] Webcam ready (no real depth - volume will use fixed fallback).")
else:
    cap = None  # not used when ZED is active

left_edge_angle = -30
right_edge_angle = 30
if zed is not None:
    # 1. Get the camera information
    cam_info = zed.get_camera_information()
    
    # 2. Extract the Horizontal and Vertical FOV (returns degrees)
    # We use the left camera because that is what you are retrieving images from
    h_fov = cam_info.camera_configuration.calibration_parameters.left_cam.h_fov
    
    print(f"ZED Horizontal FOV: {h_fov:.2f} degrees")
    # 3. Calculate the edges (since 0 is the exact center line)
    # If your h_fov is 110 degrees, the left edge is -55 and right is +55
    left_edge_angle  = -(h_fov / 2.0)
    right_edge_angle = (h_fov / 2.0)


# ──────────────────────────────────────────────────────────────
# ML MODEL INITIALISATION
# ──────────────────────────────────────────────────────────────

base_options    = python.BaseOptions(model_asset_path=MODEL_PATH)
mp_options      = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=NUM_POSES,
    min_pose_detection_confidence=0.7,#0.5
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.6
)
pose_landmarker = vision.PoseLandmarker.create_from_options(mp_options)

hand_detector  = HandDetector(detectionCon=0.8, maxHands=MAX_HANDS)
face_detector  = FaceDetectorDNN(conf_threshold=0.5)
tracker        = PersonTracker(max_people=20, max_missed=15, match_thresh_px=120.0)
associator     = Associator(gesture_classifier=HammingGestureClassifier(GESTURE_MATRIX, max_hamming=1))
audio_manager  = GestureAudioManager()

print("EMMAeye running. Press ESC to exit.")
t0              = time.time()
last_t_ms       = -1
last_wall_alert = 0.0   # throttle wall-alert console spam


# ──────────────────────────────────────────────────────────────
# FRAME ACQUISITION  (unified ZED / webcam interface)
# ──────────────────────────────────────────────────────────────

def grab_frame():
    """
    Returns (frame_bgr, depth_fn) where:
      depth_fn(x, y) -> float | None
        Returns the true Euclidean distance in mm at pixel (x, y), or None if unavailable.
        Uses 3D point cloud when using ZED; always None for webcam.
    """
    if USE_ZED:
        import pyzed.sl as sl
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            return None, None
            
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        # Retrieve XYZ point cloud instead of just DEPTH
        zed.retrieve_measure(xyz_zed, sl.MEASURE.XYZ) 
        
        frame = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
        
        # Use our new Euclidean calculator
        depth_fn = lambda x, y: get_euclidean_depth_zed(xyz_zed, x, y)
        xyz_fn  = lambda x, y: get_physical_xyz(xyz_zed, x, y)
    else:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            return None, None, None
            
        depth_fn = lambda x, y: None  # no depth sensor available
        xyz_fn   = lambda x, y: None  # no depth sensor available

    return frame, depth_fn, xyz_fn


# ──────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────
if (USE_ZED):
    print("using zed")

while True:
    frame, depth_fn, xyz_fn = grab_frame()
    if frame is None:
        print("Frame read failed. Exiting.")
        break

    h, w = frame.shape[:2]
    cam_cx = w // 2
    cam_cy = h // 2
    current_time = time.time()

    # Strictly increasing timestamp for MediaPipe
    t_ms = int((current_time - t0) * 1000)
    if t_ms <= last_t_ms:
        t_ms = last_t_ms + 1
    last_t_ms = t_ms

    # ── Pose detection ────────────────────────────────────────
    mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pose_result = pose_landmarker.detect_for_video(mp_image, t_ms)

    detected_poses = []
    if pose_result.pose_landmarks:
        for lm_list in pose_result.pose_landmarks[:NUM_POSES]:
            detected_poses.append(build_pose_data(lm_list, w, h, COUPLING_FACTOR))

    people = tracker.update(detected_poses)
    # Deduplicate by depth: if two people are within 80px AND within 150mm depth, drop the newer ID
    if USE_ZED:
        to_remove = set()
        plist = [p for p in people if p.pose]
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                a, b = plist[i], plist[j]
                px_dist = euclid2(a.pose.center, b.pose.center)
                if px_dist < 80:
                    da = depth_fn(int(a.pose.center[0]), int(a.pose.center[1]))
                    db = depth_fn(int(b.pose.center[0]), int(b.pose.center[1]))
                    if da and db and abs(da - db) < 150:
                        to_remove.add(max(a.id, b.id))  # <- all inside the if px_dist block
        people = [p for p in people if p.id not in to_remove]  # <- outside the loops

    # ── Hand & face detection ─────────────────────────────────
    hands, frame = hand_detector.findHands(frame, draw=True, flipType=FLIP_TYPE)
    hands_list   = hands if hands else []
    faces        = face_detector.detect(frame)

    associator.assign_hands_and_faces(people, hands_list, faces, t_ms)

    # ── SYSTEM 1: Wall / obstacle echolocation (ZED only) ─────
    if USE_ZED:
        cx_wall   = w // 2
        cy_wall   = h // 2
        wall_dist = depth_fn(cx_wall, cy_wall)
        enable_warning = False
        if wall_dist is not  None:
            if wall_dist < WALL_ALERT_THRESHOLD_MM:
                intensity = 1.0 - (wall_dist/WALL_ALERT_THRESHOLD_MM)
                if (current_time - last_wall_alert) >= ALERT_COOLDOWN_SEC:
                    print(f"[WALL]  Obstacle at {int(wall_dist)} mm")
                    last_wall_alert = current_time
                    enable_warning = True
            cv2.circle(frame, (cx_wall, cy_wall), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Wall: {int(wall_dist)} mm",
                        (cx_wall - 50, cy_wall - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if enable_warning:
            audio_manager.toggle_warning(state=True, intensity=intensity)
            print("warning toggled.")

        else:
            audio_manager.toggle_warning(state=False)


    # ── SYSTEM 2 & 3: Per-person UI + depth-aware audio ───────
    for p in people:
        if not p.pose:
            continue

        px, py = int(p.pose.center[0]), int(p.pose.center[1])
        cv2.putText(frame, f"Person {p.id}", (px - 50, py - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Face bbox + optional depth label
        if p.face:
            fx, fy, fw, fh = p.face.bbox if hasattr(p.face, 'bbox') else p.face["bbox"]
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            face_dist = depth_fn(fx + fw // 2, fy + fh // 2)
            if face_dist is not None:
                cv2.putText(frame, f"{int(face_dist)} mm",
                            (fx, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Hands: gesture label + depth-scaled audio trigger
        for side in ["BODY_LEFT", "BODY_RIGHT"]:
            hd = p.hands[side]
            if not hd:
                continue

            hx, hy, hw, hh = hd.bbox if hasattr(hd, 'bbox') else hd["bbox"]
            gesture         = getattr(hd, 'stable_gesture', -1)

            cv2.putText(frame, f"{side} {print_gesture(gesture)}", (hx, hy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            if hd.changed:
                sign_name = gid_to_sign_name(gesture)

                # Derive audio position; use real depth for Z when available
                hand_dist = depth_fn(hx + hw // 2, hy + hh // 2)
                hand_xyz  = xyz_fn(hx + hw // 2, hy + hh // 2)
                if hand_dist is not None and hand_xyz is not None:
                    volume    = distance_to_volume(hand_dist)
                    hand_x    = hand_xyz[0] if hand_xyz else 0
                    hand_y    = hand_xyz[1] if hand_xyz else 0
                    hand_z    = hand_xyz[2] if hand_xyz else 0
                    audio_pos = (hand_x/100.0, hand_y/100.0, hand_z / 100.0)
                    if CLIP_AUDIO:
                        audio_pos = clip_position_to_points(audio_pos, target_angles_deg=[left_edge_angle, 0, right_edge_angle])
                    print(f"[HAND]  Person {p.id} | {side}"
                          f" | dist={int(hand_dist)} mm"
                          f" | vol={int(volume * 100)}%"
                          f" | {sign_name}"
                          f" | pos={audio_pos}")
                else:
                    # No depth sensor - fixed neutral position / full volume
                    audio_pos = (px / 100.0, py / 100.0, 0)
                    if CLIP_AUDIO:
                        audio_pos = clip_position_to_points(audio_pos, target_angles_deg=[left_edge_angle, 0, right_edge_angle])

                audio_manager.trigger_gesture(p.id, audio_pos, sign_name, volume)

    cv2.imshow("EMMAeye", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        print("break on waitKey")
        break


# ──────────────────────────────────────────────────────────────
# CLEANUP
# ──────────────────────────────────────────────────────────────

if USE_ZED:
    zed.close()
else:
    cap.release()

cv2.destroyAllWindows()
audio_manager.cleanup()