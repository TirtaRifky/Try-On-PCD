# Virtual Hair Try-On with SVM+ORB Face Detection

A real-time virtual hair try-on system using classical computer vision techniques. The system combines Haar Cascade for initial face detection with ORB features and SVM classification for robust face verification, enabling accurate hair overlay placement without deep learning dependencies.

## 🌟 Features

- **Lightweight Face Detection**
  - Haar Cascade for initial face detection
  - ORB (Oriented FAST and Rotated BRIEF) features with Bag of Visual Words
  - SVM classification for face verification
  - No deep learning dependencies (no dlib, no cmake required)

- **Real-time Performance**
  - ≥15 FPS on 720p video
  - Efficient feature extraction and classification
  - Optimized for CPU processing

- **Virtual Hair Try-On**
  - Support for multiple hair styles
  - Real-time hair overlay with transparency
  - Interactive style switching
  - Automatic positioning and scaling

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd svm_orb_face_streamer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

The system requires a user-provided dataset organized as follows:

```
data/
  faces/         # Positive samples (face images)
  non_faces/     # Negative samples (non-face images)
```

Guidelines for dataset preparation:
- Face images should be roughly cropped around the face
- Non-face images can be any images without faces
- Recommended minimum: 1000 faces, 2000 non-faces
- Supported formats: jpg, jpeg, png
- Images will be automatically resized during training

## Usage

### Training

Train the face detector with your dataset:

```bash
python app.py train \
    --pos_dir data/faces \
    --neg_dir data/non_faces \
    --models_dir models \
    --n_keypoints 500
```

### Evaluation

Evaluate the trained model on the test split:

```bash
python app.py eval \
    --pos_dir data/faces \
    --neg_dir data/non_faces \
    --models_dir models \
    --report reports/test_metrics.json
```

### Live Streaming

Run face detection on webcam and stream coordinates via UDP:

```bash
python app.py stream \
    --camera 0 \
    --ip 127.0.0.1 \
    --port 4242 \
    --models_dir models
```

## Integration with Godot

Add this GDScript to receive face coordinates in your Godot project:

```gdscript
extends Node

var udp_server = PacketPeerUDP.new()
var port = 4242

func _ready():
    # Start UDP server
    if udp_server.listen(port) != OK:
        print("Failed to start UDP server")
        return
    print("Listening on port ", port)

func _process(_delta):
    # Check for new UDP packets
    if udp_server.get_available_packet_count() > 0:
        # Get packet as string
        var data = udp_server.get_packet().get_string_from_ascii()
        
        # Parse coordinates
        var coords = data.split(",")
        if coords.size() == 4:
            var x = int(coords[0])
            var y = int(coords[1])
            var w = int(coords[2])
            var h = int(coords[3])
            
            # Use coordinates to update game object
            if w > 0 and h > 0:  # Face detected
                print("Face detected at: ", x, ", ", y)
                # Update game object position/state
            else:  # No face detected
                print("No face detected")
```

## Technical Details

### Pipeline Components

1. **ROI Proposal**
   - Uses Haar Cascade for fast initial face detection
   - Provides candidate regions for detailed analysis

2. **Feature Extraction**
   - ORB (Oriented FAST and Rotated BRIEF) features
   - Fast, scale and rotation invariant
   - Bag of Visual Words (BoVW) representation

3. **Classification**
   - Linear SVM for efficient inference
   - Trained on BoVW features
   - Non-Maximum Suppression for multiple detections

### Performance

- Processing Speed: ≥15 FPS on 720p webcam feed
- Accuracy: Depends heavily on training data quality
- Typical AP scores: 0.85-0.95 with good dataset

## Limitations

- Performance depends heavily on training dataset quality
- Less robust than deep learning approaches
- Limited to frontal faces (due to Haar cascade)
- May struggle with extreme poses or lighting
- Single face tracking (streams only largest detected face)

## License

MIT License - feel free to use and modify for your projects.