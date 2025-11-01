import cv2
import numpy as np
import time
from typing import List, Tuple, Dict

def load_cascade(cascade_path: str) -> cv2.CascadeClassifier:
    """Load a Haar cascade classifier."""
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise ValueError(f"Failed to load cascade from {cascade_path}")
    return cascade

def non_max_suppression(boxes: List[Tuple[int, int, int, int]], 
                       scores: List[float], 
                       threshold: float = 0.3) -> List[int]:
    """Apply non-maximum suppression to remove overlapping boxes."""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    # Compute areas
    areas = boxes[:, 2] * boxes[:, 3]
    
    # Sort by scores
    idxs = np.argsort(scores)
    pick = []
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find IoU with rest of the boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # Compute intersection area
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[idxs[:last]]
        
        # Delete indices with overlap > threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > threshold)[0])))
    
    return pick

class FPSCounter:
    """Simple FPS counter."""
    def __init__(self, window_size: int = 30):
        self.times = []
        self.window_size = window_size
        
    def update(self) -> float:
        """Update FPS counter and return current FPS."""
        self.times.append(time.time())
        if len(self.times) > self.window_size:
            self.times.pop(0)
        
        if len(self.times) > 1:
            return (len(self.times) - 1) / (self.times[-1] - self.times[0])
        return 0.0