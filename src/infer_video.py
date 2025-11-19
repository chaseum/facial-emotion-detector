# infer_video.py  — Gameplay Affect Meter + selective FER in cutscenes
# ---------------------------------------------------------------
# Core idea:
# - Always compute arousal from video motion + luminance change
# - Detect cutscenes with a simple rule (low motion variance + low HUD/textiness)
# - Only during cutscenes (and if --use_fer), run your existing FER model on the biggest face
# - Draw a clean overlay with an arousal bar and optional emoji+label
#
# Minimal external deps: OpenCV, NumPy. OCR (pytesseract) and FER are optional via flags.
# ---------------------------------------------------------------

import argparse
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np

# Optional OCR for HUD/text density. Safe fallback if not installed.
try:
    import pytesseract
    OCR_OK = True
except Exception:
    OCR_OK = False

# Optional TensorFlow for FER. Only loaded if --use_fer is passed.
tf = None

# ---------- Utilities ----------
class EMA:
    def __init__(self, beta=0.8, init=0.0):
        self.b = beta
        self.y = init
        self.ready = False
    def update(self, x):
        if not self.ready:
            self.y = x
            self.ready = True
        else:
            self.y = self.b * self.y + (1 - self.b) * x
        return self.y

class SlidingNormalizer:
    def __init__(self, w=300, eps=1e-6):
        self.w = w
        self.buf = deque(maxlen=w)
        self.eps = eps
    def push(self, v):
        self.buf.append(float(v))
    def norm(self, v):
        if not self.buf:
            return 0.0
        arr = np.array(self.buf, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < self.eps:
            return 0.0
        return float((v - lo) / (hi - lo))

def safe_resize(img, max_w=640):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def draw_bar(canvas, x, y, w, h, v, label):
    # v in [0,1]
    v = float(np.clip(v, 0, 1))
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (50,50,50), 1)
    fill_h = int(h * v)
    cv2.rectangle(canvas, (x+1, y+h-fill_h), (x+w-1, y+h-1), (180,180,180), -1)
    cv2.putText(canvas, f"{label}: {v:.2f}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, cv2.LINE_AA)

def textiness_score(gray):
    # Cheap proxy for HUD/text density without OCR.
    # More white thin edges => higher “textiness”.
    e = cv2.Canny(gray, 80, 160, L2gradient=True)
    # Emphasize horizontal/vertical strokes
    kx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    ky = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    stroke = (np.abs(kx) + np.abs(ky)) / 2.0
    stroke = (stroke > np.percentile(stroke, 85)).astype(np.uint8)
    score = 0.6 * (e.mean()/255.0) + 0.4 * stroke.mean()
    return float(score)

def ocr_token_count(bgr):
    if not OCR_OK:
        return 0
    try:
        txt = pytesseract.image_to_string(bgr)
        # crude token count
        return len([t for t in txt.strip().split() if any(c.isalpha() for c in t)])
    except Exception:
        return 0

# ---------- Optical flow ----------
def make_flow():
    try:
        return cv2.optflow.DualTVL1OpticalFlow_create()
    except Exception:
        return None  # will fallback to Farneback

def flow_mag(prev_gray, gray, tvl1):
    if tvl1 is not None:
        f = tvl1.calc(prev_gray, gray, None)
    else:
        f = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    if f is None:
        return 0.0, None
    if f.ndim == 3 and f.shape[2] == 2:
        u, v = f[...,0], f[...,1]
    else:
        # Farneback returns 2-channel
        u, v = f[...,0], f[...,1]
    mag = np.sqrt(u*u + v*v).mean()
    return float(mag), f

# ---------- Cutscene detector ----------
class CutsceneDetector:
    def __init__(self, flow_win=90, var_factor=0.4, hud_thr=0.10, ocr_thr=8, needed=45):
        self.flow_norm = SlidingNormalizer(w=600)
        self.flow_win_vals = deque(maxlen=flow_win)
        self.var_factor = var_factor
        self.hud_thr = hud_thr
        self.ocr_thr = ocr_thr
        self.needed = needed
        self.counter = 0
        self.state = False  # False=gameplay, True=cutscene

    def update(self, flow_mag_value, hud_score, ocr_tokens):
        # Maintain normalization + a short window for variance
        self.flow_norm.push(flow_mag_value)
        self.flow_win_vals.append(flow_mag_value)
        if len(self.flow_win_vals) < self.flow_win_vals.maxlen:
            self.counter = 0
            self.state = False
            return self.state

        arr = np.array(self.flow_win_vals, dtype=np.float32)
        var = float(arr.var())
        med = float(np.median(arr))
        low_motion = var < (self.var_factor * max(med, 1e-6))
        low_hud = hud_score < self.hud_thr
        low_ocr = (ocr_tokens < self.ocr_thr) if OCR_OK else True

        if low_motion and low_hud and low_ocr:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.needed:
            self.state = True
        elif self.counter == 0:
            self.state = False
        return self.state

# ---------- Optional FER wiring ----------
class FERModule:
    def __init__(self, model_path, class_path,
                 proto_path=None, weights_path=None,
                 detect_every=5, ema_beta=0.7,
                 enter_thr=0.45, exit_thr=0.35):
        global tf
        import json, tensorflow as _tf
        tf = _tf
        self.model = tf.keras.models.load_model(model_path)
        with open(class_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)
        self.proto, self.weights = proto_path, weights_path
        self.detector = None
        if self.proto and self.weights:
            self.detector = cv2.dnn.readNetFromCaffe(str(self.proto), str(self.weights))
        self.detect_every = detect_every
        self.ema = EMA(ema_beta)
        self.enter_thr, self.exit_thr = enter_thr, exit_thr
        self.frame_idx = 0
        self.tracker = None
        self.box = None
        self.last_label = None
        self.last_prob = 0.0

    def _detect_faces(self, frame_bgr, thr=0.5):
        if self.detector is None:
            return []
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
        self.detector.setInput(blob)
        det = self.detector.forward()
        boxes = []
        for i in range(det.shape[2]):
            conf = float(det[0,0,i,2])
            if conf < thr: 
                continue
            x1 = int(det[0,0,i,3] * w); y1 = int(det[0,0,i,4] * h)
            x2 = int(det[0,0,i,5] * w); y2 = int(det[0,0,i,6] * h)
            boxes.append((max(0,x1), max(0,y1), min(w, x2-x1), min(h, y2-y1)))
        return boxes

    def _ensure_tracker(self):
		# Try non-legacy API first (opencv-python)
		for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"):
			ctor = getattr(cv2, name, None)
			if callable(ctor):
				self.tracker = ctor()
				return self.tracker
		# Try legacy API (opencv-contrib-python)
		legacy = getattr(cv2, "legacy", None)
		if legacy is not None:
			for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"):
				ctor = getattr(legacy, name, None)
				if callable(ctor):
					self.tracker = ctor()
					return self.tracker
		# No tracker available → run detection-only mode
		self.tracker = None
		return None
    def _prep_face(self, frame_bgr, box, size=48):
        x,y,w,h = box
        roi = frame_bgr[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        # normalize to [0,1]
        img = gray.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=(0, -1))  # (1,H,W,1)
        return img

    def step(self, frame_bgr):
        self.frame_idx += 1
        # Periodic detection or tracking
        run_det = (self.frame_idx % self.detect_every == 1) or (self.box is None)
        if run_det and self.detector is not None:
            boxes = self._detect_faces(frame_bgr, thr=0.3)
            if boxes:
                # choose largest
                self.box = max(boxes, key=lambda b: b[2]*b[3])
                tracker = self._ensure_tracker()
                if tracker is not None:
                    tracker.clear() if hasattr(tracker, "clear") else None
                    self.tracker = self._ensure_tracker()
                    self.tracker.init(frame_bgr, tuple(self.box))
            else:
                # keep using tracker if available
                pass
        elif self.tracker is not None and self.box is not None:
            ok, bb = self.tracker.update(frame_bgr)
            if ok:
                x, y, w, h = [int(v) for v in bb]
                self.box = (x, y, w, h)

        # Classify if we have a box
        label, prob = None, 0.0
        if self.box is not None:
            img = self._prep_face(frame_bgr, self.box)
            if img is not None:
                probs = self.model.predict(img, verbose=0)[0]
                prob = float(np.max(probs))
                label = self.class_names[int(np.argmax(probs))]

        # Hysteresis
        if label is not None:
            s = self.ema.update(prob)
            if self.last_label is None:
                if s >= self.enter_thr:
                    self.last_label, self.last_prob = label, s
            else:
                # if same class, update; else only switch if strong evidence
                if label == self.last_label:
                    self.last_prob = s
                else:
                    if s >= self.enter_thr:
                        self.last_label, self.last_prob = label, s
                    elif self.last_prob < self.exit_thr:
                        # drop state
                        self.last_label, self.last_prob = None, 0.0
        else:
            # No face; decay
            if self.last_prob < self.exit_thr:
                self.last_label, self.last_prob = None, 0.0
            else:
                self.last_prob *= 0.98

        return self.last_label, self.last_prob, self.box

# ---------- Main ----------
def main(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", default="", help="Optional path to save annotated MP4")
    ap.add_argument("--display", action="store_true", help="Show a live window")
    ap.add_argument("--downscale_width", type=int, default=960, help="Process width (keeps aspect)")
    ap.add_argument("--use_fer", action="store_true", help="Enable FER during cutscenes")
    ap.add_argument("--fer_model", default="", help="Path to emotion48.keras")
    ap.add_argument("--fer_classes", default="", help="Path to class_names.json")
    ap.add_argument("--face_proto", default="", help="deploy.prototxt for SSD face")
    ap.add_argument("--face_weights", default="", help="res10_*.caffemodel for SSD face")
    ap.add_argument("--csv", default="", help="Optional path to save arousal timeline CSV")
    args = ap.parse_args(argv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # Video writer if saving
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # We'll compose side-by-side: original + 280px overlay panel
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w + 280, h))

    # FER (optional)
    fer = None
    if args.use_fer:
        if not args.fer_model or not args.fer_classes:
            raise ValueError("--use_fer requires --fer_model and --fer_classes")
        proto = Path(args.face_proto) if args.face_proto else None
        weights = Path(args.face_weights) if args.face_weights else None
        fer = FERModule(args.fer_model, args.fer_classes, proto, weights)

    # Flow + arousal
    tvl1 = make_flow()
    prev_small_gray = None
    arousal_norm = SlidingNormalizer(w=600)
    arousal_ema = EMA(beta=0.8)
    cuts = CutsceneDetector(flow_win=90, var_factor=0.4, hud_thr=0.10, ocr_thr=8, needed=45)

    # Timeline storage (t, arousal, cutscene_flag, label, prob)
    timeline = []

    # Main loop
    t0 = time.time()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        # Optional processing size
        disp_frame = frame.copy()
        small = safe_resize(frame, max_w=args.downscale_width)
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Luminance change
        if prev_small_gray is None:
            lum = 0.0
        else:
            lum = float(np.mean(np.abs(small_gray.astype(np.int16) - prev_small_gray.astype(np.int16)))) / 255.0
        # Optical flow mag
        if prev_small_gray is None:
            flowm = 0.0
        else:
            flowm, _ = flow_mag(prev_small_gray, small_gray, tvl1)
        prev_small_gray = small_gray

        # Normalize and fuse video arousal
        arousal_norm.push(0.7*flowm + 0.3*lum)
        a_raw = 0.7 * arousal_norm.norm(flowm) + 0.3 * arousal_norm.norm(lum)
        a_sm = arousal_ema.update(a_raw)

        # HUD/text proxy + OCR tokens (optional)
        hud_score = textiness_score(small_gray)
        tokens = ocr_token_count(small) if OCR_OK else 0

        # Cutscene?
        is_cutscene = cuts.update(flowm, hud_score, tokens)

        # FER only during cutscenes
        fer_label, fer_prob, face_box = None, 0.0, None
        if is_cutscene and fer is not None:
            fer_label, fer_prob, face_box = fer.step(disp_frame)

        # Compose overlay panel
        h, w = disp_frame.shape[:2]
        panel_w = 280
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        # Arousal bar
        draw_bar(panel, 24, 40, 40, h - 80, a_sm, "Arousal")

        # Status text
        mode = "CUTSCENE" if is_cutscene else "GAMEPLAY"
        cv2.putText(panel, f"Mode: {mode}", (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Flow var gate: {cuts.var_factor:.2f}", (90, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)
        cv2.putText(panel, f"HUD score: {hud_score:.2f}", (90, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)

        # FER section
        if fer_label is not None:
            cv2.putText(panel, f"FER: {fer_label} ({fer_prob:.2f})", (24, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)
            if face_box is not None:
                x,y,wf,hf = face_box
                cv2.rectangle(disp_frame, (x,y), (x+wf,y+hf), (200,200,200), 2)
        else:
            cv2.putText(panel, "FER: (cutscenes only)", (24, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1, cv2.LINE_AA)

        # Side-by-side composite
        out = np.concatenate([disp_frame, panel], axis=1)

        # Timeline record
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        timeline.append((t, a_sm, int(is_cutscene), fer_label or "", float(fer_prob)))

        if args.display:
            cv2.imshow("Affect Analyzer", out)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if writer is not None:
            writer.write(out)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Optional CSV
    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_sec","arousal","cutscene","fer_label","fer_prob"])
            w.writerows(timeline)

if __name__ == "__main__":
    main()
