import pyzed.sl as sl
import cv2
import time
import math
from cvzone.HandTrackingModule import HandDetector
from face_detector_dnn import FaceDetectorDNN

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================

WALL_ALERT_THRESHOLD_MM = 1000   # Trigger wall alert if obstacle closer than 1 meter
ALERT_COOLDOWN_SEC       = 1.0   # Minimum seconds between wall alert prints
BEEP_COOLDOWN_SEC        = 1.0   # Minimum seconds between hand-detection beeps

MAX_VOLUME_DIST = 300.0          # 30 cm -> 100% volume
MIN_VOLUME_DIST = 2000.0         # 200 cm -> 10% volume
VOLUME_FADE_RANGE = 0.9          # Volume range between max and min distance

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_valid_depth(depth_mat, x, y):
    """
    Query the ZED depth map at pixel (x, y).
    Returns the distance in millimeters, or None if the reading is invalid.
    Guards against ZED error codes, NaN, Inf, and negative values.
    """
    err, value = depth_mat.get_value(x, y)
    if (
        err == sl.ERROR_CODE.SUCCESS
        and not math.isnan(value)
        and not math.isinf(value)
        and value > 0
    ):
        return value
    return None


def distance_to_volume(dist_mm):
    """
    Map a real-world distance in millimeters to a volume level between 0.1 and 1.0.
    Linear interpolation between MAX_VOLUME_DIST and MIN_VOLUME_DIST.
    Clamps to bounds outside the defined range.
    """
    if dist_mm <= MAX_VOLUME_DIST:
        return 1.0
    if dist_mm >= MIN_VOLUME_DIST:
        return 0.1
    normalized = (dist_mm - MAX_VOLUME_DIST) / (MIN_VOLUME_DIST - MAX_VOLUME_DIST)
    return 1.0 - (normalized * VOLUME_FADE_RANGE)


def play_beep(volume):
    """
    Placeholder for the audio subsystem (handled separately).
    Prints a terminal representation of what the beep would sound like.
    """
    bar_length = int(volume * 20)
    bar = "#" * bar_length + "-" * (20 - bar_length)
    print(f"[AUDIO] BEEP  volume={int(volume * 100):>3}%  [{bar}]")


# ==========================================
# CAMERA INITIALIZATION
# ==========================================

def init_camera():
    """
    Initialize and open the ZED 2i camera with performance-optimized settings.
    Exits the program if the camera cannot be opened.
    """
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.depth_mode        = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units  = sl.UNIT.MILLIMETER
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps        = 30

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera. Error: {err}")
        print("Check that the camera is on a USB 3.0 port and the ZED SDK is installed.")
        exit(1)

    print("ZED camera ready.")
    return zed


# ==========================================
# DETECTOR INITIALIZATION
# ==========================================

def init_detectors():
    """
    Load the DNN face detector and the cvzone hand detector.
    Returns both detector objects.
    """
    face_detector = FaceDetectorDNN(
        model_path="res10_300x300_ssd_iter_140000.caffemodel",
        config_path="deploy.prototxt"
    )
    hand_detector = HandDetector(maxHands=1, detectionCon=0.8)
    print("Detectors ready.")
    return face_detector, hand_detector


# ==========================================
# SUBSYSTEM PROCESSORS
# ==========================================

def process_wall(display_frame, depth_zed, frame_shape, current_time, last_wall_alert):
    """
    System 1 — Wall / Obstacle Echolocation.
    Samples the depth at the center pixel of the frame.
    Prints a terminal alert if an obstacle is within WALL_ALERT_THRESHOLD_MM
    and the cooldown period has elapsed.
    Returns the updated last_wall_alert timestamp.
    """
    h, w = frame_shape[:2]
    center_x, center_y = w // 2, h // 2

    distance = get_valid_depth(depth_zed, center_x, center_y)
    if distance is not None:
        if distance < WALL_ALERT_THRESHOLD_MM:
            if (current_time - last_wall_alert) >= ALERT_COOLDOWN_SEC:
                print(f"[WALL]  Obstacle detected at {int(distance)} mm")
                last_wall_alert = current_time

        cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(
            display_frame,
            f"Wall: {int(distance)}mm",
            (center_x - 50, center_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    return last_wall_alert


def process_faces(display_frame, depth_zed, face_detector, frame_bgr):
    """
    System 2 — Face / Person Detection.
    Runs the DNN face detector on the current frame.
    For each detected face, queries its real-world depth from the ZED depth map
    and overlays the bounding box and distance on the display frame.
    """
    faces = face_detector.detect(frame_bgr)
    for face in faces:
        fx, fy, fw, fh = face["bbox"]
        if fw <= 0:
            continue

        face_cx = fx + (fw // 2)
        face_cy = fy + (fh // 2)
        distance = get_valid_depth(depth_zed, face_cx, face_cy)

        if distance is not None:
            cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"Person: {int(distance)}mm",
                (fx, fy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )


def process_hands(display_frame, depth_zed, hand_detector, current_time, last_beep_time):
    """
    System 3 — Hand Detection with Dynamic Audio Feedback.
    Runs the cvzone hand detector on the display frame.
    If a hand is found, queries its real-world depth from the ZED depth map,
    maps that distance to a volume level, and triggers a beep on cooldown.
    Returns the updated last_beep_time timestamp.
    """
    hands, display_frame = hand_detector.findHands(display_frame, draw=True)

    if hands:
        hand = hands[0]
        cx, cy = hand["center"]
        distance = get_valid_depth(depth_zed, cx, cy)

        if distance is not None:
            volume = distance_to_volume(distance)

            if (current_time - last_beep_time) >= BEEP_COOLDOWN_SEC:
                play_beep(volume)
                print(f"[HAND]  Distance: {int(distance)} mm  |  Volume: {int(volume * 100)}%")
                last_beep_time = current_time

    return last_beep_time


# ==========================================
# MAIN LOOP
# ==========================================

def main():
    zed = init_camera()
    face_detector, hand_detector = init_detectors()

    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    last_wall_alert = 0.0
    last_beep_time  = 0.0

    print("-" * 50)
    print("System active. Press 'q' to quit.")
    print("-" * 50)

    try:
        while True:
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

            frame_bgr    = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
            display_frame = frame_bgr.copy()
            current_time  = time.time()

            last_wall_alert = process_wall(
                display_frame, depth_zed, frame_bgr.shape, current_time, last_wall_alert
            )
            process_faces(display_frame, depth_zed, face_detector, frame_bgr)
            last_beep_time = process_hands(
                display_frame, depth_zed, hand_detector, current_time, last_beep_time
            )

            cv2.imshow("ZED 2i Echolocation System", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()