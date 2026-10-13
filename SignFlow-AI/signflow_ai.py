from __future__ import annotations
import argparse, csv, os, sys, time, tempfile, urllib.parse, urllib.request
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"; MODELS = ROOT / "models"; REPORTS = ROOT / "reports"; ASSETS = ROOT / "assets"
CSV_PATH = DATA / "gestures.csv"; MODEL_PATH = MODELS / "gesture_model.joblib"
TASK_PATH = ASSETS / "hand_landmarker.task"
TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FEATURE_VERSION = 1; FEATURE_COUNT = 126; THRESHOLD = .65
COLUMNS = ("label", "session_id", "captured_at_utc", *(f"f_{i:03d}" for i in range(FEATURE_COUNT)))
for folder in (DATA, MODELS, REPORTS, ASSETS): folder.mkdir(parents=True, exist_ok=True)

def ensure_task_model():
    if TASK_PATH.is_file() and 1_000_000 <= TASK_PATH.stat().st_size <= 30_000_000: return TASK_PATH
    u = urllib.parse.urlparse(TASK_URL)
    if u.scheme != "https" or u.hostname != "storage.googleapis.com": raise RuntimeError("Unapproved model URL")
    temp = None
    try:
        req = urllib.request.Request(TASK_URL, headers={"User-Agent": "SignFlow-AI/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            end = urllib.parse.urlparse(response.geturl())
            if end.scheme != "https" or end.hostname != "storage.googleapis.com": raise RuntimeError("Unexpected model download redirect")
            fd, temp = tempfile.mkstemp(dir=ASSETS, prefix="hand-model-", suffix=".tmp"); total = 0
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = response.read(131072)
                    if not chunk: break
                    total += len(chunk)
                    if total > 30_000_000: raise RuntimeError("Model asset exceeds size limit")
                    out.write(chunk)
                out.flush(); os.fsync(out.fileno())
        if total < 1_000_000: raise RuntimeError("Downloaded model appears incomplete")
        os.replace(temp, TASK_PATH); temp = None
        return TASK_PATH
    except Exception as exc: raise RuntimeError(f"Unable to obtain MediaPipe model: {exc}") from exc
    finally:
        if temp and os.path.exists(temp): os.unlink(temp)

class Tracker:
    def __init__(self):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_task_model())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=.6, min_hand_presence_confidence=.6, min_tracking_confidence=.6)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.started = time.monotonic(); self.last_ms = -1
    def process(self, frame):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ms = max(int((time.monotonic() - self.started) * 1000), self.last_ms + 1); self.last_ms = ms
        return self.detector.detect_for_video(image, ms)
    def close(self): self.detector.close()

def features(result):
    hands = getattr(result, "hand_world_landmarks", None) or []
    handed = getattr(result, "handedness", None) or []
    slots = {"left": np.zeros(63, np.float32), "right": np.zeros(63, np.float32)}; extras = []
    for i, points in enumerate(hands[:2]):
        arr = np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32)
        if arr.shape != (21, 3): raise ValueError("MediaPipe returned an unexpected landmark shape")
        arr -= arr[0]; scale = float(np.max(np.linalg.norm(arr[:, :2], axis=1)))
        if scale > 1e-6: arr /= scale
        vec = arr.reshape(-1); name = ""
        if i < len(handed) and handed[i]:
            c = handed[i][0]; name = str(getattr(c, "category_name", getattr(c, "categoryName", ""))).lower()
        if name in slots and not np.any(slots[name]): slots[name] = vec
        else: extras.append(vec)
    for side in ("left", "right"):
        if extras and not np.any(slots[side]): slots[side] = extras.pop(0)
    return np.concatenate((slots["left"], slots["right"])).astype(np.float32)

def has_hand(result): return bool(getattr(result, "hand_world_landmarks", None))
def draw(frame, result):
    for hand in (getattr(result, "hand_landmarks", None) or []):
        mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

def collect(label, samples=250, camera=0, interval=.15):
    label = label.strip()
    if not label or len(label) > 64: raise ValueError("Label must contain 1–64 characters")
    if not 10 <= samples <= 10000 or not .05 <= interval <= 5: raise ValueError("Samples/interval out of range")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened(): cap.release(); raise RuntimeError(f"Cannot open webcam index {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    tracker = None; count = 0; active = False; last = 0.; sid = str(uuid4())
    try:
        tracker = Tracker()
        while count < samples:
            ok, frame = cap.read()
            if not ok: raise RuntimeError("Webcam frame read failed")
            frame = cv2.flip(frame, 1); result = tracker.process(frame); draw(frame, result)
            if active and has_hand(result) and time.monotonic() - last >= interval:
                row = features(result)
                fresh = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
                with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if fresh: w.writerow(COLUMNS)
                    w.writerow([label, sid, datetime.now(timezone.utc).isoformat(), *row.tolist()])
                count += 1; last = time.monotonic()
            cv2.rectangle(frame, (10,10), (670,105), (22,28,36), -1)
            cv2.putText(frame, f"{label}: {count}/{samples}", (24,42), cv2.FONT_HERSHEY_SIMPLEX,.65,(240,245,250),2,cv2.LINE_AA)
            cv2.putText(frame, ("RECORDING" if active else "PAUSED - SPACE to record")+" | Q/ESC quit", (24,78),cv2.FONT_HERSHEY_SIMPLEX,.52,(70,220,180),1,cv2.LINE_AA)
            cv2.imshow("SignFlow AI - Data Collection", frame); key = cv2.waitKey(1) & 255
            if key in (ord("q"), 27): break
            if key == 32: active = not active
    finally:
        cap.release()
        if tracker: tracker.close()
        cv2.destroyAllWindows()
    print(f"Collected {count} samples for {label!r} at {CSV_PATH}")

def load_dataset():
    if not CSV_PATH.is_file(): raise FileNotFoundError("Dataset missing. Run the collect command for at least two labels.")
    X, y = [], []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS: raise ValueError("Dataset schema mismatch; move the old CSV and recollect")
        for n, row in enumerate(reader, 2):
            try:
                label = row["label"].strip(); vals = [float(row[f"f_{i:03d}"]) for i in range(FEATURE_COUNT)]
                if not label or not np.isfinite(vals).all(): raise ValueError("empty label or non-finite feature")
            except Exception as exc: raise ValueError(f"Invalid dataset row {n}: {exc}") from exc
            X.append(vals); y.append(label)
    if len(X) < 2: raise ValueError("Not enough labelled samples")
    return np.asarray(X, np.float32), np.asarray(y)

def train():
    X, y = load_dataset(); classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2: raise ValueError("At least two gesture labels are required")
    low = {str(c): int(n) for c,n in zip(classes,counts) if n < 25}
    if low: raise ValueError(f"Each label needs at least 25 samples: {low}")
    test_n = max(len(classes), int(round(len(y)*.2)))
    if test_n >= len(y): raise ValueError("Not enough data for held-out evaluation")
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=test_n,random_state=42,stratify=y)
    model = RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced_subsample",random_state=42,n_jobs=-1).fit(Xtr,ytr)
    pred = model.predict(Xte); labels = sorted(map(str,classes)); cm = confusion_matrix(yte,pred,labels=labels).tolist()
    metrics = {"created_at_utc":datetime.now(timezone.utc).isoformat(),"feature_version":FEATURE_VERSION,"feature_count":FEATURE_COUNT,"samples":int(len(y)),"classes":labels,"class_counts":{str(c):int(n) for c,n in zip(classes,counts)},"accuracy":float(accuracy_score(yte,pred)),"balanced_accuracy":float(balanced_accuracy_score(yte,pred)),"evaluation_note":"Random stratified holdout; correlated frames may inflate performance","classification_report":classification_report(yte,pred,labels=labels,output_dict=True,zero_division=0),"confusion_matrix_labels":labels,"confusion_matrix":cm}
    joblib.dump({"model":model,"feature_version":FEATURE_VERSION,"feature_count":FEATURE_COUNT,"threshold":THRESHOLD,"metrics":{k:metrics[k] for k in ("accuracy","balanced_accuracy","samples","classes")}},MODEL_PATH)
    (REPORTS/"evaluation.json").write_text(__import__("json").dumps(metrics,indent=2),encoding="utf-8")
    (REPORTS/"classification_report.txt").write_text(classification_report(yte,pred,labels=labels,zero_division=0),encoding="utf-8")
    with (REPORTS/"confusion_matrix.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["actual/predicted",*labels])
        for label,row in zip(labels,cm): w.writerow([label,*row])
    return metrics

class Smoother:
    def __init__(self, threshold=.65, size=7, votes=3):
        if not 0 <= threshold <= 1 or size < 1 or not 1 <= votes <= size: raise ValueError("Invalid smoothing settings")
        self.threshold=threshold; self.votes=votes; self.buffer=deque(maxlen=size)
    def reset(self): self.buffer.clear()
    def update(self,label,score):
        self.buffer.append((label,float(score)) if label and score>=self.threshold else None)
        valid=[x for x in self.buffer if x]
        if not valid: return "Uncertain",0.
        counts=Counter(x for x,_ in valid); best,n=counts.most_common(1)[0]
        if n < self.votes: return "Stabilizing...",0.
        scores=[s for x,s in valid if x==best]
        return best,sum(scores)/len(scores)

def run(camera=0):
    if not MODEL_PATH.is_file(): raise FileNotFoundError("No trained model. Collect samples and run train first.")
    bundle=joblib.load(MODEL_PATH)
    if not isinstance(bundle,dict) or bundle.get("feature_version")!=FEATURE_VERSION or bundle.get("feature_count")!=FEATURE_COUNT: raise ValueError("Model incompatible; retrain it")
    model=bundle["model"]
    if not hasattr(model,"predict_proba"): raise ValueError("Model lacks predict_proba; retrain it")
    smooth=Smoother(float(bundle.get("threshold",THRESHOLD))); cap=cv2.VideoCapture(camera)
    if not cap.isOpened(): cap.release(); raise RuntimeError(f"Cannot open webcam index {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)
    tracker=None; missing=0; prev=time.perf_counter(); fps=0.
    try:
        tracker=Tracker()
        while True:
            ok,frame=cap.read()
            if not ok: raise RuntimeError("Webcam frame read failed")
            frame=cv2.flip(frame,1); result=tracker.process(frame); label="No hand detected"; score=0.
            if has_hand(result):
                missing=0; draw(frame,result); probs=model.predict_proba(features(result).reshape(1,-1))[0]; i=int(np.argmax(probs)); label,score=smooth.update(str(model.classes_[i]),float(probs[i]))
            else:
                missing+=1
                if missing>=10: smooth.reset()
            now=time.perf_counter(); inst=1/max(now-prev,1e-6); prev=now; fps=inst if not fps else .85*fps+.15*inst
            cv2.rectangle(frame,(10,10),(650,120),(22,28,36),-1)
            cv2.putText(frame,"GESTURE: "+label[:40],(25,45),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,245,250),2,cv2.LINE_AA)
            cv2.putText(frame,f"MODEL SCORE: {score:.2f} | FPS: {fps:.1f}",(25,80),cv2.FONT_HERSHEY_SIMPLEX,.55,(70,220,180),1,cv2.LINE_AA)
            cv2.putText(frame,"R reset smoothing | Q/ESC quit",(25,105),cv2.FONT_HERSHEY_SIMPLEX,.45,(180,195,205),1,cv2.LINE_AA)
            cv2.imshow("SignFlow AI - Live Recognition",frame); key=cv2.waitKey(1)&255
            if key in (ord("q"),27): break
            if key==ord("r"): smooth.reset()
    finally:
        cap.release()
        if tracker: tracker.close()
        cv2.destroyAllWindows()

def doctor():
    import importlib.metadata
    print(f"SignFlow AI | Python {sys.version.split()[0]}"); failed=False
    for name in ("numpy","opencv-python","mediapipe","scikit-learn","joblib","pytest"):
        try: print(f"[OK] {name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError: print(f"[MISSING] {name}"); failed=True
    print(f"Dataset: {'present' if CSV_PATH.exists() else 'not collected'}")
    print(f"Model: {'present' if MODEL_PATH.exists() else 'not trained'}")
    return int(failed)

def main():
    p=argparse.ArgumentParser(description="SignFlow AI - real-time static hand gesture recognition")
    sub=p.add_subparsers(dest="cmd",required=True); sub.add_parser("doctor")
    c=sub.add_parser("collect"); c.add_argument("--label",required=True); c.add_argument("--samples",type=int,default=250); c.add_argument("--camera",type=int,default=0)
    sub.add_parser("train"); r=sub.add_parser("run"); r.add_argument("--camera",type=int,default=0); a=p.parse_args()
    try:
        if a.cmd=="doctor": return doctor()
        if a.cmd=="collect": collect(a.label,a.samples,a.camera); return 0
        if a.cmd=="train":
            m=train(); print(f"Accuracy {m['accuracy']:.3f} | balanced accuracy {m['balanced_accuracy']:.3f}"); print(f"Reports saved in {REPORTS}"); return 0
        if a.cmd=="run": run(a.camera); return 0
    except (ValueError,FileNotFoundError,RuntimeError,OSError) as exc:
        print(f"[ERROR] {exc}",file=sys.stderr); return 2
    return 2
if __name__=="__main__": raise SystemExit(main())
