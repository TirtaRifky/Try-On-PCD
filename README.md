# Virtual Hair Try-On System with Classical Computer Vision

A real-time virtual hair try-on system using classical computer vision techniques and Godot UI. The system combines traditional computer vision (Haar Cascade + SVM + ORB) with a modern Godot interface for an interactive hair try-on experience.

## 🌟 Features

### Computer Vision Engine
- **Two-Stage Face Detection**
  - Haar Cascade for fast initial detection
  - SVM + ORB for accurate face verification
  - No deep learning dependencies
  - CPU-optimized processing (≥15 FPS at 720p)

### Interactive UI (Godot)
- **User-Friendly Interface**
  - Real-time webcam feed
  - Easy hair style selection
  - Interactive controls
  - Style preview system

### Hair Try-On
- **Real-time Overlay**
  - Multiple hair styles support
  - Alpha blending for natural look
  - Automatic positioning and scaling
  - Smooth tracking and updates

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Godot 4.x
- Webcam

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Try-On-PCD
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Open the Godot project:
```
pcd_try-on-ui-godot/project.godot
```

## 💻 Usage

### 1. Train Face Detection Model (Optional)
Skip this if using pre-trained models.

```bash
# Training
python app.py train \
    --pos_dir data/faces \
    --neg_dir data/non_faces \
    --models_dir models \
    --n_keypoints 2000 \
    --n_faces 2500 \
    --n_non_faces 2500

# Evaluation
python app.py eval \
    --pos_dir data/faces \
    --neg_dir data/non_faces \
    --models_dir models \
    --report reports/eval_metrics.json
```

### 2. Start the Hair Try-On Server
```bash
python udp_hair_server.py \
    --models_dir models \
    --hair_dir assets/hair-asset
```

### 3. Launch Godot Interface
- Open project in Godot Editor
- Run the scene (F5)
- Or export and run the standalone application

## 🎮 Godot Interface Guide

### Main Features
- Webcam view with real-time hair overlay
- Hair style selection panel
- Information popup
- Settings control

### Controls
- Click hair styles to try different options
- Use info button for help
- Close button to exit application

## 🔧 System Architecture

### Python Backend
1. **Face Detection Pipeline**
   ```
   Camera Input → Haar Cascade → ORB Features → SVM Verification → Hair Overlay
   ```

2. **UDP Communication**
   - Server: Python backend (udp_hair_server.py)
   - Client: Godot interface (udp_client.gd)
   - Default port: 4242

### Godot Frontend
1. **Main Scene Components**
   - WebCamController: Handles video feed
   - SelectionListHair: Manages hair options
   - UDPClient: Handles communication

2. **Scripts**
   - `web_cam_controller.gd`: Webcam management
   - `selection_list_hair.gd`: Hair selection UI
   - `udp_client.gd`: Backend communication

## 📁 Project Structure
```
Try-On-PCD/
├── assets/                 # Resources
│   ├── cascades/          # Haar cascade files
│   └── hair-asset/        # Hair overlay images
├── data/                  # Training data
├── models/               # Trained models
├── pipelines/            # CV pipeline
├── pcd_try-on-ui-godot/  # Godot interface
│   ├── Scripts/          # GDScript files
│   ├── themes/           # UI assets
│   └── hair-asset/       # Hair images
├── app.py               # Training script
├── demo.py              # Basic demo
└── udp_hair_server.py   # Main server
```

## ⚙️ Configuration

### Python Backend
- Models directory: `models/`
- Hair assets: `assets/hair-asset/`
- Default port: 4242
- Cascade file: `assets/cascades/haarcascade_frontalface_default.xml`

### Godot Frontend
- Default UDP port: 4242
- Hair assets path: `hair-asset/`
- Video resolution: 720p

## 📝 Notes

### Performance Tips
- Ensure good lighting for face detection
- Keep face centered and frontal
- Maintain stable internet connection
- Close other webcam applications

### Known Limitations
- Works best with frontal faces
- Single face detection only
- Requires consistent lighting
- CPU-dependent performance


## 🙏 Acknowledgments
- OpenCV for computer vision tools
- Godot Engine for UI framework
- scikit-learn for machine learning components