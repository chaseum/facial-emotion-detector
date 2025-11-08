import cv2
import numpy as np
import tensorflow as tf
import json
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "emotion48.keras"
CLASS_PATH = ROOT / "artifacts" / "class_names.json"
EMOJI_DIR = ROOT / "emojis"

model = tf.keras.models.load_model(str(MODEL_PATH))
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)
name_to_idx = {n: i for i, n in enumerate(class_names)}

def load_emojis(emoji_dir, size = (256,256)):
    emojis = {}
    for cname in class_names:
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            p = (emoji_dir / (cname + ext))
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 2:
                    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[-1] == 4:
                    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                else:
                    bgr = img
                emojis[cname] = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                break
        if cname not in emojis:
            placeholder = np.full((size[1], size[0], 3), 240, dtype = np.uint8)
            cv2.putText(placeholder, cname, (10, size[1]//2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
            emojis[cname] = placeholder
    return emojis

emojis = load_emojis(EMOJI_DIR, size = (256, 256))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

ema_probs = None
ALPHA = 0.8
ENTER_THR = 0.60
EXIT_THR = 0.45
CONF_THR = 0.50

current_label_idx = None
window = deque(maxlen = 21)
picture_size = 48

def preprocess_face(gray_roi) :
    roi = cv2.resize(gray_roi, (picture_size,picture_size), interpolation=cv2.INTER_AREA)
    roi = roi.astype("float32")
    roi = np.expand_dims(roi, axis = -1)
    roi = np.expand_dims(roi, axis = 0)
    return roi

def majority_label(indices_window):
    if not indices_window:
        return None
    counts = np.bincount(list(indices_window), minlength = num_classes)
    return int(np.argmax(counts))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam :/")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (60,60)
    )

    shown_label = None
    shown_conf = 0.0

    if len(faces) > 0:
        areas = [(w*h, i) for i, (_,_,w,h) in enumerate(faces)]
        _, best_i = max(areas, key=lambda t: t[0])
        (x,y,w,h) = faces[best_i]

        roi = preprocess_face(gray[y:y+h, x:x+w])
        probs = model.predict(roi, verbose = 0)[0]

        if ema_probs is None:
            ema_probs = probs.copy()
        else:
            ema_probs = ALPHA * ema_probs + (1 - ALPHA) * probs

        idx = int(np.argmax(ema_probs))
        conf = float(ema_probs[idx])

        window.append(idx)
        maj_idx = int(np.bincount(list(window), minlength=num_classes).argmax())

        if current_label_idx is None:
            if conf >= ENTER_THR:
                current_label_idx = maj_idx
        else:
            if maj_idx != current_label_idx and conf >= ENTER_THR:
                current_label_idx = maj_idx
            elif conf < EXIT_THR:
                current_label_idx = None

        disp_idx = current_label_idx if current_label_idx is not None else idx
        disp_label = class_names[disp_idx]
        shown_label, shown_conf = disp_label, conf

        color = (0, 255, 0) if conf >= CONF_THR else (0,200,255)
        cv2.rectangle(display, (x,y), (x+w, y+h), color, 2)
        text = f"{disp_label} ({conf:.2f})"
        cv2.putText(display, text, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    panel_w = 300
    panel = np.full((display.shape[0], panel_w, 3), 245, dtype=np.uint8)

    if shown_label is None:
        emo = emojis.get("neutral", next(iter(emojis.values())))
        emo_h, emo_w = emo.shape[:2]
        y0 = max(0, (panel.shape[0] - emo_h)//2)
        x0 = max(0, (panel.shape[1] - emo_w)//2)
        panel[y0:y0+emo_h, x0:x0+emo_w] = emo
        cv2.putText(panel, "No face", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,50,50), 2, cv2.LINE_AA)
    else:
        emo = emojis[shown_label]
        emo_h, emo_w = emo.shape[:2]
        y0 = max(0,(panel.shape[0] - emo_h)//2)
        x0 = max(0, (panel.shape[1] - emo_w)//2)
        panel[y0:y0+emo_h, x0:x0+emo_w] = emo
        cv2.putText(panel, f"{shown_label}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow("Emotion detector", display)
    cv2.imshow("Emote", panel)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
