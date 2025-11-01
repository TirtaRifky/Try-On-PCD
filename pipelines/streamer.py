import cv2
import socket
import numpy as np
from typing import Tuple, List, Optional
from .utils import FPSCounter, non_max_suppression, load_cascade
from .features import ORBFeatureExtractor

class FaceDetectorStreamer:
    """Stream face detection results over UDP."""
    def __init__(self, cascade_path: str,
                 feature_extractor: ORBFeatureExtractor,
                 svm,
                 target_size: Tuple[int, int] = (64, 64)):
        self.cascade = load_cascade(cascade_path)
        self.feature_extractor = feature_extractor
        self.svm = svm
        self.target_size = target_size
        self.fps_counter = FPSCounter()
        
    def detect_face(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect faces in a frame using cascade + SVM validation."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect potential faces using Haar cascade
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return []
        
        # Prepare face candidates for SVM validation
        candidates = []
        boxes = []
        for (x, y, w, h) in faces:
            # Extract and resize face region
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, self.target_size)
            candidates.append(face_roi)
            boxes.append((x, y, w, h))
        
        # Extract BoVW features
        features = self.feature_extractor.extract_bovw_features(candidates)
        
        # Get SVM scores
        scores = self.svm.decision_function(features)
        
        # Filter detections using non-max suppression
        indices = non_max_suppression(boxes, scores)
        
        # Return filtered detections with scores
        return [(boxes[i], scores[i]) for i in indices]
    
    def start_streaming(self, camera_id: int,
                       udp_ip: str,
                       udp_port: int,
                       show_preview: bool = True):
        """Start the face detection and UDP streaming loop."""
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Initialize UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces
                detections = self.detect_face(frame)
                
                # Get primary (largest) face if any
                primary_face: Optional[Tuple[int, int, int, int]] = None
                if detections:
                    # Sort by area and take largest
                    detections.sort(key=lambda x: x[0][2] * x[0][3], reverse=True)
                    primary_face = detections[0][0]
                    
                    # Draw rectangle if preview is enabled
                    if show_preview:
                        x, y, w, h = primary_face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Send coordinates over UDP
                if primary_face:
                    x, y, w, h = primary_face
                    msg = f"{x},{y},{w},{h}"
                else:
                    msg = "0,0,0,0"  # No face detected
                
                sock.sendto(msg.encode(), (udp_ip, udp_port))
                
                # Show preview if enabled
                if show_preview:
                    # Update and show FPS
                    fps = self.fps_counter.update()
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('Face Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            sock.close()