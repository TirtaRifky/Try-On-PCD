import os
import argparse
import json
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from pipelines.dataset import FaceDataset
from pipelines.features import ORBFeatureExtractor
from pipelines.train import FaceDetectorTrainer
from pipelines.streamer import FaceDetectorStreamer

def train(args):
    """Train the face detector model."""
    # Load dataset with specified counts
    dataset = FaceDataset(
        face_dir=args.pos_dir,
        non_face_dir=args.neg_dir,
        n_faces=args.n_faces,
        n_non_faces=args.n_non_faces
    )
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.prepare_dataset()
    
    # Initialize feature extractor with updated parameters
    feature_extractor = ORBFeatureExtractor(
        n_keypoints=args.n_keypoints,
        n_words=args.n_clusters
    )
    trainer = FaceDetectorTrainer(feature_extractor)
    
    # Train model
    print("Training model...")
    metrics = trainer.train(X_train, y_train, X_val, y_val)
    
    # Save models
    print("Saving models...")
    trainer.save_models(args.models_dir)
    
    # Save metrics
    metrics_path = os.path.join(args.models_dir, 'train_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training complete! Average Precision: {metrics['average_precision']:.3f}")

def evaluate(args):
    """Evaluate the trained model."""
    # Load dataset
    dataset = FaceDataset(args.pos_dir, args.neg_dir)
    _, _, X_test, _, _, y_test = dataset.prepare_dataset()
    
    # Load trained models
    trainer, _ = FaceDetectorTrainer.load_models(args.models_dir)
    
    # Evaluate
    print("Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save metrics
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n=== Evaluation Results ===")
    print(f"Accuracy:           {metrics['accuracy']:.3f}")
    print(f"Precision:          {metrics['precision']:.3f}")
    print(f"Recall:             {metrics['recall']:.3f}")
    print(f"F1-Score:           {metrics['f1_score']:.3f}")
    print(f"Specificity:        {metrics['specificity']:.3f}")
    print(f"Average Precision:  {metrics['average_precision']:.3f}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.3f}")
    print("\n=== Confusion Matrix ===")
    print(f"True Positives:     {metrics['confusion_matrix']['true_positive']}")
    print(f"False Positives:    {metrics['confusion_matrix']['false_positive']}")
    print(f"True Negatives:     {metrics['confusion_matrix']['true_negative']}")
    print(f"False Negatives:    {metrics['confusion_matrix']['false_negative']}")

def stream(args):
    """Run face detection on webcam and stream results over UDP."""
    # Load trained models
    trainer, feature_extractor = FaceDetectorTrainer.load_models(args.models_dir)
    
    # Initialize streamer
    cascade_path = os.path.join(args.cascade_dir, 'haarcascade_frontalface_default.xml')
    streamer = FaceDetectorStreamer(
        cascade_path=cascade_path,
        feature_extractor=feature_extractor,
        svm=trainer.svm
    )
    
    # Start streaming
    print(f"Starting face detection stream to {args.ip}:{args.port}")
    streamer.start_streaming(
        camera_id=args.camera,
        udp_ip=args.ip,
        udp_port=args.port,
        show_preview=not args.no_preview
    )

def main():
    parser = argparse.ArgumentParser(description='Face Detection with SVM and ORB features')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train the face detector')
    train_parser.add_argument('--pos_dir', type=str, required=True,
                            help='Directory containing face images')
    train_parser.add_argument('--neg_dir', type=str, required=True,
                            help='Directory containing non-face images')
    train_parser.add_argument('--models_dir', type=str, default='models',
                            help='Directory to save trained models')
    train_parser.add_argument('--n_keypoints', type=int, default=1000,
                            help='Number of ORB keypoints to detect')
    train_parser.add_argument('--n_clusters', type=int, default=64,
                            help='Number of visual words in BoVW codebook')
    train_parser.add_argument('--n_faces', type=int, default=3000,
                            help='Number of face images to use for training')
    train_parser.add_argument('--n_non_faces', type=int, default=1000,
                            help='Number of non-face images to use for training')
    
    # Evaluation parser
    eval_parser = subparsers.add_parser('eval', help='Evaluate the trained model')
    eval_parser.add_argument('--pos_dir', type=str, required=True,
                           help='Directory containing face images')
    eval_parser.add_argument('--neg_dir', type=str, required=True,
                           help='Directory containing non-face images')
    eval_parser.add_argument('--models_dir', type=str, default='models',
                           help='Directory containing trained models')
    eval_parser.add_argument('--report', type=str,
                           help='Path to save evaluation report')
    
    # Streaming parser
    stream_parser = subparsers.add_parser('stream',
                                        help='Run face detection on webcam')
    stream_parser.add_argument('--camera', type=int, default=0,
                             help='Camera device ID')
    stream_parser.add_argument('--ip', type=str, default='127.0.0.1',
                             help='UDP target IP')
    stream_parser.add_argument('--port', type=int, default=4242,
                             help='UDP target port')
    stream_parser.add_argument('--models_dir', type=str, default='models',
                             help='Directory containing trained models')
    stream_parser.add_argument('--cascade_dir', type=str, default='assets/cascades',
                             help='Directory containing Haar cascade XML files')
    stream_parser.add_argument('--no_preview', action='store_true',
                             help='Disable preview window')
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        evaluate(args)
    elif args.command == 'stream':
        stream(args)

if __name__ == '__main__':
    main()