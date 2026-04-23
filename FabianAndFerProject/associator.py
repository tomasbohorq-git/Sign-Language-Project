import math
from typing import List, Dict, Optional, Tuple
from collections import deque

from person import Person, HandData, FaceData, PoseData
from gesture_logic import (
    shift_to_origin, rotate_hand_around_reference, expand_hand_area,
    check_all_points_position, compare_to_gesture_matrix
)

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0]-b[0], a[1]-b[1]))

class GestureStabilizer:
    def __init__(self, history_len=12, votes_to_accept=8, drop_to_none_after=15):
        self.history = deque(maxlen=history_len)
        self.stable = -1
        self.no_gesture = 0
        self.votes_to_accept = votes_to_accept
        self.drop_to_none_after = drop_to_none_after

    def update(self, gid: int) -> Tuple[bool, int]:
        if gid != -1:
            self.history.append(gid)
            self.no_gesture = 0
        else:
            self.no_gesture += 1

        new_stable = self.stable
        if self.history:
            vals = list(self.history)
            winner = max(set(vals), key=vals.count)
            if vals.count(winner) >= self.votes_to_accept:
                new_stable = winner

        if self.no_gesture >= self.drop_to_none_after:
            new_stable = -1
            self.history.clear()

        changed = (new_stable != self.stable)
        self.stable = new_stable
        return changed, new_stable

def compute_gesture_id(lm_list_xyz, gesture_matrix, max_hamming=1):
    shifted = shift_to_origin(lm_list_xyz)
    rotated = rotate_hand_around_reference(shifted)
    palm_poly = expand_hand_area(rotated)
    palm_np = __import__("numpy").array(palm_poly, dtype=__import__("numpy").int32)
    features = check_all_points_position(rotated, palm_np)
    gid = compare_to_gesture_matrix(features, gesture_matrix, max_hamming=max_hamming)
    return gid

class Associator:
    def __init__(self, gesture_matrix, max_hamming=1, coupling_factor=0.35):
        self.gesture_matrix = gesture_matrix
        self.max_hamming = max_hamming
        self.coupling_factor = coupling_factor

        # Stabilizers keyed by (person_id, side)
        self.stabilizers: Dict[Tuple[int, str], GestureStabilizer] = {}

    def _get_stab(self, person_id: int, side: str) -> GestureStabilizer:
        key = (person_id, side)
        if key not in self.stabilizers:
            self.stabilizers[key] = GestureStabilizer()
        return self.stabilizers[key]

    def assign_hands_and_faces(self, people: List[Person], hands: List[dict], faces: List[dict], t_ms: int):
        """
        people: Person list already updated with PoseData by the tracker
        hands: cvzone hands raw list (each has lmList, bbox)
        faces: DNN faces list (bbox, center, confidence)
        """
        # Clear previous frame associations
        for p in people:
            p.t_ms = t_ms
            p.face = None
            p.hands["BODY_LEFT"] = None
            p.hands["BODY_RIGHT"] = None

        # ----- Face -> Pose assignment (nearest head within threshold)
        for p in people:
            if not p.pose or not p.pose.head:
                continue
            best_face = None
            best_d = 1e9
            thresh = max(120.0, 1.2 * p.pose.shoulder_dist)
            for f in faces:
                d = _dist((f["center"][0], f["center"][1]), p.pose.head)
                if d < best_d and d <= thresh:
                    best_d = d
                    best_face = f
            if best_face:
                p.face = FaceData(
                    bbox=best_face["bbox"],
                    center=best_face["center"],
                    conf=best_face["confidence"],
                    dist_to_head=best_d
                )

        # ----- Hands -> Pose wrist assignment
        for hand in hands:
            lm = hand["lmList"]
            bbox = hand["bbox"]
            hand_wrist = (lm[0][0], lm[0][1])

            # gesture raw
            raw_gid = compute_gesture_id(lm, self.gesture_matrix, max_hamming=self.max_hamming)

            best = None  # (dist, person, side)
            for p in people:
                if not p.pose:
                    continue
                pose = p.pose

                # Skip if wrists not visible
                candidates = []
                if pose.left_wrist:
                    candidates.append(("BODY_LEFT", pose.left_wrist))
                if pose.right_wrist:
                    candidates.append(("BODY_RIGHT", pose.right_wrist))
                if not candidates:
                    continue

                # threshold scaled by shoulders
                thresh = pose.coupling_thresh
                for side, wrist in candidates:
                    d = _dist(hand_wrist, wrist)
                    if d <= thresh:
                        if best is None or d < best[0]:
                            best = (d, p, side)

            if best is None:
                continue

            d, person, side = best

            # Only keep closest hand per side
            existing = person.hands[side]
            if existing is not None and existing.dist_to_wrist <= d:
                continue

            # stable gesture per person+side
            stab = self._get_stab(person.id, side)
            changed, stable_gid = stab.update(raw_gid)

            person.hands[side] = HandData(
                lm=lm,
                bbox=tuple(bbox),
                raw_gesture=raw_gid,
                stable_gesture=stable_gid,
                assigned_side=side,
                dist_to_wrist=d
            )

            if changed:
                print(f"[Person {person.id}] {side} -> STABLE G{stable_gid}")
