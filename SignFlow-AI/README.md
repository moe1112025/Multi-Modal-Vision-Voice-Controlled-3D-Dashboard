# SignFlow AI

Local webcam-based **static hand-gesture recognition** using MediaPipe Hand Landmarker, OpenCV, NumPy, and a trained Random Forest. The app does not fabricate labels: collect real labelled samples and train a model before inference.

## Install (Windows PowerShell)

1. Extract the ZIP and open the SignFlow-AI folder in VS Code.
2. Run: py -3.11 -m venv .venv
3. Run: .\.venv\Scripts\Activate.ps1
4. Run: python -m pip install --upgrade pip
5. Run: python -m pip install -r requirements.txt

Python 3.11 is recommended. On macOS/Linux create a venv with python3.11 -m venv .venv, activate it, and install requirements. Package wheel availability depends on OS and architecture.

## Commands

- python signflow_ai.py doctor
- python -m pytest
- python signflow_ai.py collect --label open_palm --samples 250
- python signflow_ai.py collect --label closed_fist --samples 250
- python signflow_ai.py collect --label thumbs_up --samples 250
- python signflow_ai.py train
- python signflow_ai.py run
- python signflow_ai.py run --camera 1

During collection press SPACE to start/pause recording. Q or ESC exits. Collect at least 25 samples per class; 100–300 or more per class across multiple sessions is a better starting point. Vary lighting, background, orientation, and distance. The first run downloads the official MediaPipe task asset over HTTPS.

Training creates models/gesture_model.joblib and reports/evaluation.json, reports/classification_report.txt, and reports/confusion_matrix.csv. Live inference supports R to reset smoothing and Q/ESC to quit.

## Pipeline

Webcam -> MediaPipe hand landmarks -> normalized 126-value features -> labelled CSV -> Random Forest training/evaluation -> saved model -> confidence threshold and temporal smoothing -> OpenCV UI.

## Important limitation

This classifies static poses represented in your own dataset. It is **not a complete sign-language translator**. Dynamic signs need sequence modelling and temporal data; full sign-language recognition may also require facial expression, body pose, linguistic context, and representative datasets. Random Forest probability scores are not guaranteed calibrated real-world probabilities. Random frame-level train/test splits may overestimate performance when frames are correlated; validate on separate recording sessions/users before making performance claims.

## Privacy

Camera frames are not saved by this app. Landmark features are stored locally and may be biometric-related data; do not commit real datasets or trained models. The model download uses HTTPS and host checks but does not pin a cryptographic checksum.

## Troubleshooting

If the camera cannot open, close other camera apps or try --camera 1. If MediaPipe cannot install, verify Python 3.11 and OS/architecture wheel support. Poor predictions usually require more balanced, varied training samples; inspect the evaluation reports.
