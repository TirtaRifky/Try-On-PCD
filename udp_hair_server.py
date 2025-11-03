import cv2
import numpy as np
import joblib
import socket
from pipelines.features import ORBFeatureExtractor

class UDPHairServer:
    def __init__(self, models_dir: str, cascade_path: str, hair_assets_dir: str):
        # Initialize UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 9000))
        print("UDP server listening on 127.0.0.1:9000")
        
        # Load trained models
        self.svm = joblib.load(f"{models_dir}/svm.pkl")
        self.feature_extractor = ORBFeatureExtractor()
        self.feature_extractor.kmeans = joblib.load(f"{models_dir}/codebook.pkl")
        self.feature_extractor.scaler = joblib.load(f"{models_dir}/scaler.pkl")
        
        # Load Haar cascade
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load hair assets
        self.hair_images = []
        self.current_hair_idx = 0
        self.load_hair_assets(hair_assets_dir)
        
        # Store client address
        self.client_addr = None

    def load_hair_assets(self, assets_dir: str):
        """Load and preprocess hair overlay images."""
        import glob
        import os
        import re
        
        # Get all PNG files
        hair_paths = glob.glob(os.path.join(assets_dir, "*.png"))
        
        # Sort numerically using the number in the filename
        def get_number(path):
            match = re.search(r'hair(\d+)\.png$', path)
            return int(match.group(1)) if match else 0
            
        hair_paths = sorted(hair_paths, key=get_number)
        
        for path in hair_paths:
            hair = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if hair is not None and hair.shape[-1] == 4:  # Ensure RGBA
                self.hair_images.append(hair)
                
        print(f"Loaded {len(self.hair_images)} hair assets")

    def verify_face(self, face_img: np.ndarray) -> bool:
        """Use SVM+ORB model to verify if region is a face."""
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        features = self.feature_extractor.extract_bovw_features([face_resized])
        score = self.svm.decision_function(features)[0]
        return score > 0

    def overlay_hair(self, frame: np.ndarray, x: int, y: int, w: int, h: int):
        """Overlay hair asset on the frame at face position."""
        if not self.hair_images:
            return frame
            
        hair = self.hair_images[self.current_hair_idx]
        
        # Scale hair to match face width but maintain aspect ratio
        scale_factor = 2.0
        scale = (w * scale_factor) / hair.shape[1]
        new_h = int(hair.shape[0] * scale)
        new_w = int(hair.shape[1] * scale)
        hair_resized = cv2.resize(hair, (new_w, new_h))
        
        # Calculate overlay position
        vertical_offset = int(new_h * 0.2)
        hair_y = max(0, y - new_h//2 + vertical_offset)
        hair_x = max(0, x - (new_w - w)//2)
        
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

    def process_image(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        # Mirror image
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if self.verify_face(face_img):
                frame = self.overlay_hair(frame, x, y, w, h)
        
        return frame

    def handle_message(self, data: bytes, addr):
        """Handle incoming UDP messages."""
        message = data.decode('utf-8')
        
        if message == "ping":
            self.client_addr = addr
            self.sock.sendto(b"pong", addr)
            print(f"Connected to client at {addr}")
            
        elif message.startswith("hair_type:"):
            try:
                hair_num = int(message.split(":")[1]) - 1  # Convert to 0-based index
                if 0 <= hair_num < len(self.hair_images):
                    self.current_hair_idx = hair_num
                    print(f"Changed hair style to {hair_num + 1}")
                    self.sock.sendto(b"ok", addr)
                else:
                    print(f"Invalid hair index: {hair_num + 1}")
            except ValueError:
                print("Invalid hair_type message format")
                
        elif message == "disconnect":
            self.client_addr = None
            print(f"Client {addr} disconnected")

    def send_frame(self, frame):
        """Send a video frame to the connected client."""
        if self.client_addr:
            # Compress frame to JPEG
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            # Convert to bytes and send
            img_bytes = img_encoded.tobytes()
            try:
                self.sock.sendto(img_bytes, self.client_addr)
            except Exception as e:
                print(f"Error sending frame: {e}")

    def run(self):
        """Run the UDP hair server."""
        print("Hair overlay UDP server running...")
        try:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return

            while True:
                # Check for incoming messages
                if self.sock.fileno() != -1:  # Check if socket is still open
                    self.sock.setblocking(False)  # Non-blocking socket mode
                    try:
                        data, addr = self.sock.recvfrom(1024)
                        self.handle_message(data, addr)
                    except socket.error:
                        pass

                # Process camera frame if client is connected
                if self.client_addr:
                    ret, frame = cap.read()
                    if ret:
                        # Process frame with face detection and hair overlay
                        processed_frame = self.process_image(frame)
                        # Send processed frame to client
                        self.send_frame(processed_frame)

        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            if 'cap' in locals():
                cap.release()
            self.sock.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hair overlay UDP server")
    parser.add_argument("--models_dir", type=str, default="models",
                       help="Directory containing trained models")
    parser.add_argument("--cascade_path", type=str,
                       default="assets/cascades/haarcascade_frontalface_default.xml",
                       help="Path to Haar cascade XML file")
    parser.add_argument("--hair_dir", type=str,
                       default="assets/hair-asset",
                       help="Directory containing hair overlay images")
    
    args = parser.parse_args()
    
    server = UDPHairServer(
        models_dir=args.models_dir,
        cascade_path=args.cascade_path,
        hair_assets_dir=args.hair_dir
    )
    server.run()