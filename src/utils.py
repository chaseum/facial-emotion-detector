import json
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from . import ARTIFACTS

def set_seed(s=42):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def save_class_names(names, path: Path = ARTIFACTS / "class_names.json"):
    with open(path, "w") as f: json.dump(list(names), f)

def load_class_names(path: Path = ARTIFACTS / "class_names.json"):
    with open(path) as f: return json.load(f)