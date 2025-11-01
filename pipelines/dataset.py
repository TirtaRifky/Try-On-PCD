import os
import cv2
import glob
import random
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

class FaceDataset:
    """Dataset loader for face detection with balanced sampling."""
    def __init__(self, face_dir: str, non_face_dir: str, n_faces: int = 3000, n_non_faces: int = 1000):
        """
        Initialize dataset loader.
        
        Args:
            face_dir: Directory containing face images (HELEN dataset)
            non_face_dir: Directory containing non-face images (ANIMALS)
            n_faces: Target number of face images to use (default: 3000)
            n_non_faces: Target number of non-face images to use (default: 1000)
        """
        self.face_dir = face_dir
        self.non_face_dir = non_face_dir
        self.n_faces = n_faces
        self.n_non_faces = n_non_faces
        
    def get_image_paths(self, directory: str, patterns: List[str] = None) -> List[str]:
        """Get all image paths matching the patterns in directory."""
        if patterns is None:
            patterns = ["*.jpg"]
        
        all_paths = []
        for pattern in patterns:
            paths = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
            all_paths.extend(paths)
        return all_paths
    
    def load_and_preprocess_image(self, img_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load and preprocess a single image."""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to target size
            resized = cv2.resize(gray, target_size)
            
            return resized
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    
    def prepare_dataset(self, target_size: Tuple[int, int] = (64, 64)) -> Tuple:
        """
        Load and prepare the dataset for training with balanced sampling.
        
        Args:
            target_size: Size to resize all images to
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Get all image paths
        face_paths = self.get_image_paths(self.face_dir, ["*.jpg"])  # Only use jpg files from HELEN
        non_face_paths = self.get_image_paths(self.non_face_dir, ["*.jpg", "*.jpeg"])  # Include both jpg and jpeg for non-faces
        
        print(f"Found {len(face_paths)} face images and {len(non_face_paths)} non-face images")
        
        if not face_paths or not non_face_paths:
            raise ValueError("No images found in data directories")
        
        # Sample faces
        if len(face_paths) >= self.n_faces:
            print(f"Sampling {self.n_faces} faces from {len(face_paths)} available...")
            face_paths = random.sample(face_paths, self.n_faces)
        else:
            print(f"Warning: Only {len(face_paths)} face images available, wanted {self.n_faces}")

        # Sample non-faces
        if len(non_face_paths) >= self.n_non_faces:
            print(f"Sampling {self.n_non_faces} non-faces from {len(non_face_paths)} available...")
            non_face_paths = random.sample(non_face_paths, self.n_non_faces)
        else:
            print(f"Warning: Only {len(non_face_paths)} non-face images available, wanted {self.n_non_faces}")
            
        # If we don't have enough non-faces, we might need to duplicate some
        if len(non_face_paths) < self.n_non_faces:
            while len(non_face_paths) < self.n_non_faces:
                non_face_paths.extend(random.sample(non_face_paths, min(len(non_face_paths), 
                                                                      self.n_non_faces - len(non_face_paths))))
        
        # Load and preprocess images
        faces = []
        non_faces = []
        
        print("Loading face images...")
        for path in face_paths:
            img = self.load_and_preprocess_image(path, target_size)
            if img is not None:
                faces.append(img)
        
        print("Loading non-face images...")
        for path in non_face_paths:
            img = self.load_and_preprocess_image(path, target_size)
            if img is not None:
                non_faces.append(img)
        
        # Create labels
        labels = np.concatenate([
            np.ones(len(faces)),
            np.zeros(len(non_faces))
        ])
        
        # Stack images
        X = np.array(faces + non_faces)
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Number of faces: {len(faces)}")
        print(f"Number of non-faces: {len(non_faces)}")
        print(f"Actual ratio: {len(faces)/len(non_faces):.2f}:1")
        
        # Split into train/val/test (70/15/15)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, labels, test_size=0.15, random_state=42, stratify=labels)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=(15/85), random_state=42, stratify=y_temp)
        
        # Print split sizes
        print(f"\nSplit sizes:")
        print(f"Training: {len(X_train)} images")
        print(f"Validation: {len(X_val)} images")
        print(f"Test: {len(X_test)} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test