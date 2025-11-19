import cv2
import json
import time
import numpy as np
import tensorflow as tf
from collections import deque
from pathlib import Path

from . import PROJECT, ARTIFACTS, MODELS_DIR
from .utils import load_class_names

ASSETS = PROJECT / "assets"
EMOJI_DIR = ASSETS / "emojis"
MODEL_PATH = ARTIFACTS / "emotion48.keras"
CLASS_PATH = ARTIFACTS / "class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)
class_names = load_class_names(CLASS_PATH)

proto = MODELS_DIR / "deploy.prototxt"
weights = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))

IMG_SIZE = 48
EMA_ALPHA = 0.6
ENTER_THR = 0.35
EXIT_THR = 0.25
HOLD_SEC = 0.30
HIST_LEN = 8

def detect_faces(frame_bgr):
    (h, w) = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300,300)), 1.0, (300,300),
                                 (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0,0,i,2])
        if conf < 0.6:
            continue
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w-1,x2), min(h-1,y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1,y1,x2,y2))
    return boxes
	
def preprocess(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)   # <- BGR not BGRA
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = resized.astype("float32")
    return x[None, ..., None]

def soft_ema(prev, curr, alpha=EMA_ALPHA):
	return alpha * prev + (1-alpha) * curr

def load_emoji(label):
    p = EMOJI_DIR / f"{label}.png"
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        a = img[:, :, 3].astype(np.float32) / 255.0   # (H,W)
        return (bgr, a)
    return (img, None)

def paste_emoji(panel, emoji, max_w, max_h):
    if emoji is None:
        return panel
    bgr, a = emoji
    eh, ew = bgr.shape[:2]
    scale = min(max_w / ew, max_h / eh, 1.0)
    nw, nh = max(1, int(round(ew * scale))), max(1, int(round(eh * scale)))

    bgr_r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    y0 = (panel.shape[0] - nh) // 2
    x0 = (panel.shape[1] - nw) // 2
    roi = panel[y0:y0+nh, x0:x0+nw].astype(np.float32)

    if a is None:
        panel[y0:y0+nh, x0:x0+nw] = bgr_r.astype(np.uint8)
    else:
        a_r = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
        if a_r.ndim == 2:                 # ensure shape (nh, nw, 1)
            a_r = a_r[..., None]
        comp = bgr_r * a_r + roi * (1.0 - a_r)
        panel[y0:y0+nh, x0:x0+nw] = comp.astype(np.uint8)
    return panel

def main(cam_index = 0):
	cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
	if not cap.isOpened():
		raise RuntimeError("Cannot open camera")
	
	ema_prob = None
	hist = deque(maxlen = HIST_LEN)
	current_label = None
	last_switch_t = 0.0
	cached_emojis = {}

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		frame_draw = frame.copy()
		boxes = detect_faces(frame)
		face_roi = None
		if boxes:
			x1, y1, x2, y2 = max(boxes, key = lambda b: (b[2]-b[0])*(b[3]-b[1]))
			cv2.rectangle(frame_draw, (x1,y1), (x2,y2), (0,255,0), 2)
			face_roi = frame[y1:y2, x1:x2]

		if face_roi is not None:
			x = preprocess(face_roi)
			probs = model.predict(x, verbose = 0)[0]
		else:
			probs = np.zeros((len(class_names),), dtype = np.float32)
		
		if ema_prob is None:
			ema_prob = probs
		else:
			ema_prob = soft_ema(ema_prob, probs, EMA_ALPHA)

		hist.append(ema_prob.copy())
		probs_smoothed = np.mean(np.stack(hist, axis = 0), axis= 0)

		max_idx = int(np.argmax(probs_smoothed))
		max_prob = float(probs_smoothed[max_idx])
		top_label = class_names[max_idx]

		now = time.time()
		show_label = current_label
		if current_label is None:
			if max_prob >= ENTER_THR:
				current_label = top_label
				last_switch_t = now
		else:
			if top_label != current_label and (now - last_switch_t) >= HOLD_SEC:
				if max_prob >= ENTER_THR:
					current_label = top_label
					last_switch_t = now
			if max_prob < EXIT_THR and (now - last_switch_t) >= HOLD_SEC:
				current_label = None
		
		h, w = frame_draw.shape[:2]
		panel_w = int(0.35*w)
		panel = np.full((h,panel_w, 3), 245, dtype=np.uint8)

		title = "No face" if face_roi is None else f"{current_label or '...'} {max_prob:.2f}"
		cv2.putText(frame_draw, title, (10,30),
			  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
		
		if current_label:
			if current_label not in cached_emojis:
				cached_emojis[current_label] = load_emoji(current_label)
			panel = paste_emoji(panel, cached_emojis[current_label], max_w=panel_w - 20, max_h = h -20)

		out = np.hstack([frame_draw, panel])
		cv2.imshow("EmotionCamera", out)
		
		key = cv2.waitKey(1) & 0xFF
		if key == 27 or key == ord('q'):
			break
		elif key == ord('r'):
			ema_prob = None
			hist.clear()
			current_label = None
		
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
			


	