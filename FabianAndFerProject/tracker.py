import math
from typing import List, Tuple, Dict, Optional
from person import Person, PoseData

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0]-b[0], a[1]-b[1]))

class PersonTracker:
    def __init__(self, max_people: int = 10, max_missed: int = 60, match_thresh_px: float = 160.0):
        """
        match_thresh_px: base distance threshold. We will scale it by shoulder distance when possible.
        max_missed: frames before removing a person if not seen
        """
        self.max_people = max_people
        self.max_missed = max_missed
        self.match_thresh_px = match_thresh_px
        self._next_id = 1
        self.people: Dict[int, Person] = {}

    def _new_person(self) -> Person:
        p = Person(self._next_id)
        self.people[p.id] = p
        self._next_id += 1
        return p

    def update(self, detected_poses: List[PoseData]) -> List[Person]:
        """
        Assign detected poses to existing persons (persistent IDs).
        Greedy matching by nearest center with threshold.
        Returns active persons updated this frame.
        """

        # Mark all as missed initially
        for p in self.people.values():
            p.missed_frames += 1

        if not detected_poses:
            # Remove dead persons
            self._cleanup()
            return list(self.people.values())

        # Build candidate matches: (cost, person_id, pose_idx)
        candidates = []
        pose_centers = [pose.center for pose in detected_poses]

        for pid, person in self.people.items():
            if person.last_center is None:
                continue
            for i, c in enumerate(pose_centers):
                candidates.append((_dist(person.last_center, c), pid, i))

        candidates.sort(key=lambda x: x[0])

        assigned_person = set()
        assigned_pose = set()

        # Greedy assignment
        for cost, pid, i in candidates:
            if pid in assigned_person or i in assigned_pose:
                continue

            pose = detected_poses[i]
            # Adaptive threshold: base + fraction of shoulder distance
            adaptive = max(self.match_thresh_px, 0.9 * pose.shoulder_dist)
            if cost <= adaptive:
                person = self.people[pid]
                person.pose = pose
                person.last_center = pose.center
                person.missed_frames = 0
                assigned_person.add(pid)
                assigned_pose.add(i)

        # Create new persons for unassigned poses (if room)
        for i, pose in enumerate(detected_poses):
            if i in assigned_pose:
                continue
            if len(self.people) >= self.max_people:
                continue
            person = self._new_person()
            person.pose = pose
            person.last_center = pose.center
            person.missed_frames = 0

        self._cleanup()
        return list(self.people.values())

    def _cleanup(self):
        dead = [pid for pid, p in self.people.items() if p.missed_frames > self.max_missed]
        for pid in dead:
            del self.people[pid]
