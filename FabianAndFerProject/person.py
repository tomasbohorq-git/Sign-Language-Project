from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

@dataclass
class PoseData:
    # 33 landmarks in pixel coords [(x,y,visibility), ...]
    landmarks: List[Tuple[float, float, float]]
    # useful points
    center: Tuple[float, float]
    left_wrist: Optional[Tuple[float, float]]
    right_wrist: Optional[Tuple[float, float]]
    head: Optional[Tuple[float, float]]       # nose/head proxy
    shoulder_dist: float
    coupling_thresh: float

@dataclass
class HandData:
    # 21 landmarks from cvzone: [[x,y,z], ...]
    lm: List[List[float]]
    bbox: Tuple[int, int, int, int]
    raw_gesture: int
    stable_gesture: int
    assigned_side: str  # "BODY_LEFT" or "BODY_RIGHT"
    dist_to_wrist: float
    changed: bool

@dataclass
class FaceData:
    bbox: Tuple[int, int, int, int]  # (x,y,w,h)
    center: Tuple[int, int]
    conf: float
    dist_to_head: float

class Person:
    def __init__(self, person_id: int):
        self.id = person_id
        self.pose: Optional[PoseData] = None
        self.face: Optional[FaceData] = None
        self.hands: Dict[str, Optional[HandData]] = {"BODY_LEFT": None, "BODY_RIGHT": None}
        self.pose_change = False

        self.t_ms: int = 0
        self.missed_frames: int = 0
        self.last_center: Optional[Tuple[float, float]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "t_ms": self.t_ms,
            "person_id": self.id,
            "pose": asdict(self.pose) if self.pose else None,
            "face": asdict(self.face) if self.face else None,
            "hands": {
                "BODY_LEFT": asdict(self.hands["BODY_LEFT"]) if self.hands["BODY_LEFT"] else None,
                "BODY_RIGHT": asdict(self.hands["BODY_RIGHT"]) if self.hands["BODY_RIGHT"] else None,
            }
        }
