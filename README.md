# Emotion Recognition with TensorFlow, OpenCV, and Real-Time Inference

This project implements a **real-time facial emotion recognition system** using TensorFlow/Keras, OpenCV’s DNN face detector, and custom training pipelines. It includes:

- A full training pipeline (CNN + MobileNetV2 backbone options)
- Model evaluation scripts
- Web-camera live emotion inference with smoothing + hysteresis logic
- Video inference
- Utility scripts, data loaders, and artifact management
- A clean project structure suitable for GitHub

---

## 🚀 Features

### **Modeling**
- Supports **two architectures**:
  - Custom CNN (`model_cnn.py`)
  - MobileNetV2 grayscale backbone (`model_backbone.py`)
- Data augmentation, class weighting, and checkpointing
- TensorFlow/Keras training loop with early stopping and LR scheduling
- Automatic artifact saving (`artifacts/`)

### **Inference**
- Real-time webcam detection (`infer_cam.py`)
- Video inference (`infer_video.py`)
- OpenCV DNN face detection (`deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel`)
- EMA smoothing, entropy checks, hysteresis thresholds
- Class-specific confidence rules
- Emoji overlay support

### **Project Structure**

```
project/
│
├── artifacts/                # Saved models, class names, confusion matrix (ignored in Git)
├── assets/                   # Emojis, sample videos, images
├── data/                     # Training/testing dataset (ignored in Git)
├── models/                   # Face detection models (deploy.prototxt, .caffemodel)
├── src/
│   ├── train.py              # Main training entrypoint
│   ├── eval.py               # Evaluation and confusion matrix
│   ├── infer_cam.py          # Webcam inference
│   ├── infer_video.py        # Video inference (arousal meter + optional FER)
│   ├── model_cnn.py
│   ├── model_backbone.py
│   ├── gradcam.py            # Grad-CAM helper (library use, no CLI)
│   ├── utils.py
│   └── data.py
│
├── run.py                    # Unified CLI (train / eval / cam / video)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📦 Installation

### 1. Create a virtual environment

> **Python 3.10 required** — TensorFlow 2.10 does not support 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
.\.venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Training a Model

Place your dataset in:

```
data/train/<class_name>/
data/test/<class_name>/
```

Then run:

```bash
python -m src.train --model cnn --epochs 120
# or
python -m src.train --model backbone --epochs 120
```

Artifacts will be saved to:

```
artifacts/emotion48.keras
artifacts/class_names.json
artifacts/model_checkpoint.keras
```

---

## 🎥 Webcam Emotion Recognition

```bash
python -m src.infer_cam
# or: python run.py cam
```

This opens a live window, detects the face, classifies emotions, and displays a corresponding emoji.
Requires a trained model in `artifacts/` (see Training above).

---

## 🎞 Video Analysis (arousal meter + optional FER)

```bash
python -m src.infer_video --video path/to/video.mp4 --display
```

To also run emotion recognition during cutscenes:

```bash
python -m src.infer_video --video path/to/video.mp4 --display --use_fer \
    --fer_model artifacts/emotion48.keras --fer_classes artifacts/class_names.json \
    --face_proto models/deploy.prototxt --face_weights models/res10_300x300_ssd_iter_140000.caffemodel
```

---

## 🧪 Evaluation

```bash
python -m src.eval
```

Outputs:

- Per-class precision/recall/F1  
- Normalized confusion matrix  
- Classification report  

---

## ⚙ Requirements

Python 3.10, plus:

```
tensorflow==2.10.0
numpy==1.23.5
protobuf==3.19.6
opencv-python
matplotlib
scikit-learn
```

---

## 📝 License

This project is released under the **MIT License**.  
See the LICENSE file for details.

---

## 🤝 Contributing

Pull requests are welcome. For major changes:

1. Open an issue to discuss your idea  
2. Submit a clean PR with detailed explanation  

---

## 📧 Contact

For questions or collaboration inquiries, feel free to reach out.

