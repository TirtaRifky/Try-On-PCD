import cv2
import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

class ORBFeatureExtractor:
    """Extract ORB features and create BoVW representations."""
    def __init__(self, n_keypoints: int = 1000, n_words: int = 64):
        # Increase default keypoints to get more features
        self.orb = cv2.ORB_create(
            nfeatures=n_keypoints,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )
        self.n_words = n_words
        self.kmeans: Optional[MiniBatchKMeans] = None
        self.scaler: Optional[StandardScaler] = None
        
    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Extract ORB keypoints and descriptors from an image."""
        keypoints = self.orb.detect(image, None)
        keypoints, descriptors = self.orb.compute(image, keypoints)
        
        if descriptors is None:
            # Return empty descriptor if no keypoints found
            return [], np.array([])
        return keypoints, descriptors
    
    def build_codebook(self, descriptors_list: List[np.ndarray]):
        """Build BoVW codebook using k-means clustering."""
        if not descriptors_list:
            raise ValueError("No descriptors provided to build codebook")
            
        # Concatenate all descriptors
        all_descriptors = np.vstack(descriptors_list)
        print(f"Building codebook with {len(descriptors_list)} descriptor sets, "
              f"total descriptors: {all_descriptors.shape[0]}")
        
        # Make sure we don't try to create more clusters than samples
        n_clusters = min(self.n_words, all_descriptors.shape[0] - 1)
        print(f"Using {n_clusters} clusters for k-means")
        
        # Train k-means
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(100, all_descriptors.shape[0]),
            random_state=42
        )
        self.kmeans.fit(all_descriptors)
        
    def compute_bovw(self, descriptors: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Compute Bag of Visual Words histogram for given descriptors."""
        if self.kmeans is None:
            raise ValueError("Codebook not built. Call build_codebook first.")
            
        if descriptors.size == 0:
            # Return zero vector if no descriptors
            return np.zeros(self.kmeans.n_clusters)
        
        # Assign each descriptor to a visual word
        visual_words = self.kmeans.predict(descriptors)
        
        # Create histogram
        hist = np.bincount(visual_words, minlength=self.kmeans.n_clusters)
        
        if normalize:
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # Add small epsilon to avoid division by zero
            
        return hist
    
    def extract_bovw_features(self, images: List[np.ndarray], fit: bool = False) -> np.ndarray:
        """Extract BoVW features from a list of images."""
        # Extract descriptors from all images
        descriptors_list = []
        for img in images:
            _, desc = self.extract_features(img)
            if desc.size > 0:
                descriptors_list.append(desc)
        
        if fit:
            # Build codebook if in training mode
            self.build_codebook(descriptors_list)
        
        # Compute BoVW histograms for all images
        bovw_features = []
        for img in images:
            _, desc = self.extract_features(img)
            if desc.size > 0:
                hist = self.compute_bovw(desc)
            else:
                hist = np.zeros(self.kmeans.n_clusters)
            bovw_features.append(hist)
        
        # Convert to numpy array
        bovw_features = np.array(bovw_features)
        
        if fit:
            # Fit scaler in training mode
            self.scaler = StandardScaler()
            bovw_features = self.scaler.fit_transform(bovw_features)
        elif self.scaler is not None:
            # Apply scaling in inference mode
            bovw_features = self.scaler.transform(bovw_features)
        
        return bovw_features