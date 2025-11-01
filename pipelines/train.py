import os
import joblib
from typing import Dict, Tuple
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_curve, average_precision_score
from .features import ORBFeatureExtractor

class FaceDetectorTrainer:
    """Train and evaluate the SVM face detector."""
    def __init__(self, feature_extractor: ORBFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.svm = LinearSVC(random_state=42)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train the SVM classifier and compute validation metrics."""
        # Extract BoVW features
        print("Extracting training features...")
        X_train_bovw = self.feature_extractor.extract_bovw_features(X_train, fit=True)
        
        print("Extracting validation features...")
        X_val_bovw = self.feature_extractor.extract_bovw_features(X_val)
        
        # Train SVM
        print("Training SVM...")
        self.svm.fit(X_train_bovw, y_train)
        
        # Compute validation metrics
        val_scores = self.svm.decision_function(X_val_bovw)
        precision, recall, _ = precision_recall_curve(y_val, val_scores)
        ap = average_precision_score(y_val, val_scores)
        
        metrics = {
            'average_precision': ap,
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
        
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the trained model on test data."""
        # Extract features
        X_test_bovw = self.feature_extractor.extract_bovw_features(X_test)
        
        # Compute test metrics
        test_scores = self.svm.decision_function(X_test_bovw)
        precision, recall, _ = precision_recall_curve(y_test, test_scores)
        ap = average_precision_score(y_test, test_scores)
        
        metrics = {
            'average_precision': ap,
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
        
        return metrics
    
    def save_models(self, models_dir: str):
        """Save trained models and feature extraction parameters."""
        os.makedirs(models_dir, exist_ok=True)
        
        # Save SVM model
        joblib.dump(self.svm, os.path.join(models_dir, 'svm.pkl'))
        
        # Save k-means codebook
        joblib.dump(self.feature_extractor.kmeans,
                   os.path.join(models_dir, 'codebook.pkl'))
        
        # Save scaler
        joblib.dump(self.feature_extractor.scaler,
                   os.path.join(models_dir, 'scaler.pkl'))
    
    @classmethod
    def load_models(cls, models_dir: str) -> Tuple['FaceDetectorTrainer', ORBFeatureExtractor]:
        """Load trained models and create a new trainer instance."""
        # Create feature extractor
        feature_extractor = ORBFeatureExtractor()
        
        # Load k-means codebook
        feature_extractor.kmeans = joblib.load(
            os.path.join(models_dir, 'codebook.pkl'))
        
        # Load scaler
        feature_extractor.scaler = joblib.load(
            os.path.join(models_dir, 'scaler.pkl'))
        
        # Create trainer instance
        trainer = cls(feature_extractor)
        
        # Load SVM model
        trainer.svm = joblib.load(os.path.join(models_dir, 'svm.pkl'))
        
        return trainer, feature_extractor