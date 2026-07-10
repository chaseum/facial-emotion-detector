import json
import random
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from . import ARTIFACTS

def set_seed(s=42):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def preprocess_face(face_bgr, size=48):
    # Model expects RAW 0-255 pixels: Rescaling(1/255) lives inside the model.
    # Do NOT divide by 255 here — that double-scales and breaks predictions.
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype("float32")[None, ..., None]  # (1, size, size, 1)

def save_class_names(names, path: Path = ARTIFACTS / "class_names.json"):
    with open(path, "w") as f: json.dump(list(names), f)

def load_class_names(path: Path = ARTIFACTS / "class_names.json"):
    with open(path) as f: return json.load(f)