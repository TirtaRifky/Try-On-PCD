import os
import joblib
from typing import Dict, Tuple
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
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
        
        # Get predictions
        y_val_pred = self.svm.predict(X_val_bovw)
        val_scores = self.svm.decision_function(X_val_bovw)
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
        
        # Compute validation metrics
        precision, recall, _ = precision_recall_curve(y_val, val_scores)
        
        metrics = {
            'average_precision': float(average_precision_score(y_val, val_scores)),
            'accuracy': float(accuracy_score(y_val, y_val_pred)),
            'precision': float(precision_score(y_val, y_val_pred)),
            'recall': float(recall_score(y_val, y_val_pred)),
            'f1_score': float(f1_score(y_val, y_val_pred)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'roc_auc': float(roc_auc_score(y_val, val_scores)),
            'confusion_matrix': {
                'true_positive': int(tp),
                'false_positive': int(fp),
                'true_negative': int(tn),
                'false_negative': int(fn)
            },
            'precision_curve': precision.tolist(),
            'recall_curve': recall.tolist()
        }
        
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the trained model on test data."""
        # Extract features
        X_test_bovw = self.feature_extractor.extract_bovw_features(X_test)
        
        # Get predictions
        y_test_pred = self.svm.predict(X_test_bovw)
        test_scores = self.svm.decision_function(X_test_bovw)
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        
        # Compute test metrics
        precision, recall, _ = precision_recall_curve(y_test, test_scores)
        
        metrics = {
            'average_precision': float(average_precision_score(y_test, test_scores)),
            'accuracy': float(accuracy_score(y_test, y_test_pred)),
            'precision': float(precision_score(y_test, y_test_pred)),
            'recall': float(recall_score(y_test, y_test_pred)),
            'f1_score': float(f1_score(y_test, y_test_pred)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'roc_auc': float(roc_auc_score(y_test, test_scores)),
            'confusion_matrix': {
                'true_positive': int(tp),
                'false_positive': int(fp),
                'true_negative': int(tn),
                'false_negative': int(fn)
            },
            'precision_curve': precision.tolist(),
            'recall_curve': recall.tolist()
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