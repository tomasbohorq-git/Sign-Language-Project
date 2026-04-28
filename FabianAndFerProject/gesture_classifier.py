import abc
import numpy as np
from typing import List, Tuple, Any

# Assuming these exist in your imports
from gesture_logic import (
    shift_to_origin, rotate_hand_around_reference, expand_hand_area,
    check_all_points_position, compare_to_gesture_matrix
)

class GestureClassifier(abc.ABC):
    """Base class for all gesture classification strategies."""
    
    @abc.abstractmethod
    def classify(self, lm_list_xyz: List[Any]) -> int:
        """
        Processes hand landmarks and returns a gesture ID.
        Returns -1 if no gesture is detected.
        """
        pass

class HammingGestureClassifier(GestureClassifier):
    """Implementation using Hamming distance on a gesture matrix."""
    
    def __init__(self, gesture_matrix: np.ndarray, max_hamming: int = 1):
        self.gesture_matrix = gesture_matrix
        self.max_hamming = max_hamming

    def classify(self, lm_list_xyz: List[Any]) -> int:
        shifted = shift_to_origin(lm_list_xyz)
        rotated = rotate_hand_around_reference(shifted)
        palm_poly = expand_hand_area(rotated)
        palm_np = np.array(palm_poly, dtype=np.int32)
        features = check_all_points_position(rotated, palm_np)
        
        return compare_to_gesture_matrix(
            features, 
            self.gesture_matrix, 
            max_hamming=self.max_hamming
        )