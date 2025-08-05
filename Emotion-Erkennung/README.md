# Real-Time Emotion Detection with YOLO

This project uses a YOLO classification model to detect emotions from a live webcam feed in real time.  
It loads a trained model (`emotion_erkennung.pt`), analyzes each video frame, and overlays the detected emotions and their confidence scores directly onto the video stream.


This trained model can be downloaded on this link https://github.com/prathmesh444/Emotion-Detection-using-Face-Recognition/tree/main/Model%203/run/classify/train2/weights


Its original author is Prathmesh soni, https://github.com/prathmesh444
---

## Features
- **Webcam live feed** using OpenCV
- **YOLO-based classification** for emotion recognition
- Displays:
  - **Top predicted emotion** (large green text)
  - **Other possible emotions** with confidence scores (smaller white text)
- **Quit** the application by pressing `q`

---

## Requirements
- Python 3.8+
- [Ultralytics YOLO](https://docs.ultralytics.com/) (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)

---

## Usage
Since this is a Jupyter Notebook, make sure all required dependencies are installed.  
You can then open and run it in your preferred IDE (e.g., VS Code, PyCharm) or directly in a terminal