import cv2
import numpy as np
import math

# -------------------------
# Small helpers
# -------------------------
def _safe_norm(v):
    n = math.hypot(v[0], v[1])
    return n if n > 1e-9 else 1e-9

def calculate_angle_deg(v1, v2):
    """Angle in degrees between 2D vectors."""
    x1, y1 = v1
    x2, y2 = v2
    dot = x1 * x2 + y1 * y2
    n = _safe_norm(v1) * _safe_norm(v2)
    c = max(-1.0, min(1.0, dot / n))
    return math.degrees(math.acos(c))

def rotate_hand_points_xy(hand_points, rotation_angle_deg):
    """
    Proper 2D rotation in X-Y plane; keep z unchanged.
    hand_points: list/array of [x,y,z]
    """
    a = math.radians(rotation_angle_deg)
    c, s = math.cos(a), math.sin(a)
    rotated = []
    for x, y, z in hand_points:
        xr = x * c - y * s
        yr = x * s + y * c
        rotated.append([xr, yr, z])
    return rotated

# -------------------------
# Core functions
# -------------------------
def shift_to_origin(lm_list):
    """
    Shift wrist(0) to (0,0) and invert Y to Cartesian-like coords.
    Input: cvzone [x,y,z]
    Output: numpy array shape (21,3)
    """
    lm = np.array(lm_list, dtype=float)
    origin = lm[0].copy()
    shifted = lm - origin
    shifted[:, 1] *= -1.0
    return shifted

def rotate_hand_around_reference(shifted_lm_list):
    """
    Align hand so wrist->middleMCP (0->9) points 'up' (0,1).
    Uses 2D rotation only (x,y).
    """
    # Reference 'up'
    ref = (0.0, 1.0)

    # Vector from wrist(0) to middleMCP(9)
    vx = shifted_lm_list[9][0] - shifted_lm_list[0][0]
    vy = shifted_lm_list[9][1] - shifted_lm_list[0][1]
    hand_vec = (vx, vy)

    angle = calculate_angle_deg(ref, hand_vec)

    # Determine sign using x direction (so we rotate the correct way)
    # If vx > 0, the vector leans right -> rotate left, etc.
    if vx > 0:
        rot = -(180.0 - angle)
    elif vx < 0:
        rot = (180.0 - angle)
    else:
        rot = 0.0

    return rotate_hand_points_xy(shifted_lm_list.tolist(), rot)

def expand_hand_area(rotated_lm_list):
    """
    Palm polygon with RELATIVE margin (stable across distances).
    rotated_lm_list: list of [x,y,z] (already shifted + rotated)
    Returns list of (x,y) points
    """
    # Use wrist->indexMCP distance as scale
    ax, ay = rotated_lm_list[0][0], rotated_lm_list[0][1]
    bx, by = rotated_lm_list[5][0], rotated_lm_list[5][1]
    scale = max(1.0, math.hypot(bx - ax, by - ay))
    m = 0.10 * scale  # margin factor (tune for stability)

    palm_polygon = [
        (rotated_lm_list[0][0], rotated_lm_list[0][1]),
        (rotated_lm_list[2][0] + m, rotated_lm_list[0][1]),
        (rotated_lm_list[2][0] + m, rotated_lm_list[5][1]),
        (rotated_lm_list[5][0], rotated_lm_list[5][1]),
        (rotated_lm_list[9][0],  rotated_lm_list[9][1]  + m),
        (rotated_lm_list[13][0], rotated_lm_list[13][1] + m),
        (rotated_lm_list[17][0], rotated_lm_list[17][1] + m),
        (rotated_lm_list[18][0], rotated_lm_list[17][1] + m),
        (rotated_lm_list[18][0], rotated_lm_list[0][1]),
    ]
    return palm_polygon

def check_all_points_position(lm_list, palm_polygon):
    """
    Returns feature vector:
    - For each landmark: inside palm polygon? (0/1)
    - +3 extra special features for ambiguous gestures
    """
    # Reference scale for relative thresholds
    ref = math.hypot(lm_list[5][0] - lm_list[0][0], lm_list[5][1] - lm_list[0][1])
    ref = max(ref, 1.0)

    point_positions = []
    for p in lm_list:
        inside = cv2.pointPolygonTest(palm_polygon, (p[0], p[1]), False) >= 0
        point_positions.append(1 if inside else 0)

    # Special A (Gesture 13-ish): closeness between joints
    # Keep your original intent but make it relative
    dx = abs(lm_list[3][0] - lm_list[7][0])
    point_positions.append(1 if dx < 0.18 * ref else 0)

    # Special B (Gesture 16): crossed fingers heuristic (kept)
    thumb_x, pinky_x = lm_list[4][0], lm_list[17][0]
    index_x, middle_x = lm_list[8][0], lm_list[12][0]
    crossed = ((index_x < middle_x and thumb_x > pinky_x) or
               (index_x > middle_x and thumb_x < pinky_x))
    point_positions.append(1 if crossed else 0)

    # Special C (Gesture 9): pinch (relative)
    dx = abs(lm_list[4][0] - lm_list[8][0])
    dy = abs(lm_list[4][1] - lm_list[8][1])
    pinch = (dx < 0.45 * ref) and (dy < 0.30 * ref)
    point_positions.append(1 if pinch else 0)

    return point_positions

def compare_to_gesture_matrix(point_positions, gesture_matrix, max_hamming=1):
    """
    Stable matching with Hamming distance tolerance.
    max_hamming=1 (very strict but tolerates jitter)
    """
    relevant_indices = [3, 7, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23]
    current = np.array([point_positions[i] for i in relevant_indices], dtype=int)

    best_id = -1
    best_dist = 999

    for i, row in enumerate(gesture_matrix):
        row = np.array(row, dtype=int)
        dist = int(np.sum(current != row))
        if dist < best_dist:
            best_dist = dist
            best_id = i + 1

    return best_id if best_dist <= max_hamming else -1

def shift_hand_area_back(hand_area, origin):
    """
    Convert rotated-space polygon back to image coords.
    origin must be [x,y,z] in image coords (cvzone wrist)
    """
    ox, oy = origin[0], origin[1]
    back = []
    for x, y in hand_area:
        # undo shift: +origin, undo inverted y: -y
        back.append((x + ox, -y + oy))
    return back
