import cv2
import numpy as np
import joblib
from pipelines.features import ORBFeatureExtractor

class HairOverlayApp:
    def __init__(self, models_dir: str, cascade_path: str, hair_assets_dir: str):
        # Load trained models
        self.svm = joblib.load(f"{models_dir}/svm.pkl")
        self.feature_extractor = ORBFeatureExtractor()
        self.feature_extractor.kmeans = joblib.load(f"{models_dir}/codebook.pkl")
        self.feature_extractor.scaler = joblib.load(f"{models_dir}/scaler.pkl")
        
        # Load Haar cascade for initial face detection
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load hair assets
        self.hair_images = []
        self.current_hair_idx = 0
        self.load_hair_assets(hair_assets_dir)
    
    def load_hair_assets(self, assets_dir: str):
        """Load and preprocess hair overlay images."""
        import glob
        import os
        
        hair_paths = glob.glob(os.path.join(assets_dir, "*.png"))
        for path in hair_paths:
            hair = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if hair is not None and hair.shape[-1] == 4:  # Ensure RGBA
                self.hair_images.append(hair)
        print(f"Loaded {len(self.hair_images)} hair assets")
    
    def verify_face(self, face_img: np.ndarray) -> bool:
        """Use SVM+ORB model to verify if region is a face."""
        # Preprocess
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        
        # Extract BoVW features
        features = self.feature_extractor.extract_bovw_features([face_resized])
        
        # Predict with SVM
        score = self.svm.decision_function(features)[0]
        return score > 0
    
    def overlay_hair(self, frame: np.ndarray, x: int, y: int, w: int, h: int):
        """Overlay hair asset on the frame at face position."""
        if not self.hair_images:
            return frame
        
        # Get current hair asset
        hair = self.hair_images[self.current_hair_idx]
        
        # Scale hair to match face width but maintain aspect ratio
        # Add scaling factor to make hair larger
        scale_factor = 2.0  # Increased scaling factor
        scale = (w * scale_factor) / hair.shape[1]
        new_h = int(hair.shape[0] * scale)
        new_w = int(hair.shape[1] * scale)
        hair_resized = cv2.resize(hair, (new_w, new_h))
        
        # Calculate overlay position (center horizontally, align with top of face)
        vertical_offset = int(new_h * 0.2)  # Shift hair down by 20% of its height instead of 25%
        hair_y = max(0, y - new_h//2 + vertical_offset)
        hair_x = max(0, x - (new_w - w)//2)  # Center the hair horizontally relative to face
        
        # Create mask from alpha channel
        alpha = hair_resized[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        
        # Get region of frame where hair will be overlaid
        roi_h = min(hair_resized.shape[0], frame.shape[0] - hair_y)
        roi_w = min(hair_resized.shape[1], frame.shape[1] - hair_x)
        
        if roi_h <= 0 or roi_w <= 0:
            return frame
            
        hair_roi = hair_resized[:roi_h, :roi_w, :3]
        alpha_roi = alpha[:roi_h, :roi_w]
        frame_roi = frame[hair_y:hair_y+roi_h, hair_x:hair_x+roi_w]
        
        # Blend hair with frame
        frame[hair_y:hair_y+roi_h, hair_x:hair_x+roi_w] = \
            frame_roi * (1 - alpha_roi) + hair_roi * alpha_roi
            
        return frame
    
    def run(self):
        """Run the hair overlay application."""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Mirror image
            frame = cv2.flip(frame, 1)
            
            # Detect faces using Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = frame[y:y+h, x:x+w]
                
                # Verify with our trained model
                if self.verify_face(face_img):
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Overlay hair
                    frame = self.overlay_hair(frame, x, y, w, h)
                else:
                    # Draw red rectangle for rejected faces
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Display controls
            cv2.putText(frame, "Press 'n' for next hair style", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Hair Overlay', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and self.hair_images:
                self.current_hair_idx = (self.current_hair_idx + 1) % len(self.hair_images)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hair overlay demo using trained face detector")
    parser.add_argument("--models_dir", type=str, default="models",
                       help="Directory containing trained models")
    parser.add_argument("--cascade_path", type=str,
                       default="assets/cascades/haarcascade_frontalface_default.xml",
                       help="Path to Haar cascade XML file")
    parser.add_argument("--hair_dir", type=str,
                       default="assets/hair-asset",
                       help="Directory containing hair overlay images")
    
    args = parser.parse_args()
    
    app = HairOverlayApp(
        models_dir=args.models_dir,
        cascade_path=args.cascade_path,
        hair_assets_dir=args.hair_dir
    )
    app.run()