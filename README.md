# Driver Drowsiness Detection System

A real-time driver drowsiness detection system using a CNN trained on facial images, with MediaPipe face landmark detection for live webcam inference. Detects three states: **Alert**, **Drowsy – Eyes Closed**, and **Drowsy – Yawning**.

---

## Project Structure

```
Drowsiness_detection/
├─ Drowsiness_model/
│  ├─ assets/
│  └─ variables/
└─ split_dataset/
   ├─ test/
   │  ├─ Alert/
   │  ├─ Drowsy_Eyes_Closed/
   │  └─ Drowsy_Yawning/
   ├─ train/
   │  ├─ Alert/
   │  ├─ Drowsy_Eyes_Closed/
   │  └─ Drowsy_Yawning/
   └─ validation/
      ├─ Alert/
      ├─ Drowsy_Eyes_Closed/
      └─ Drowsy_Yawning/
```

---

## Requirements

- Python 3.8+
- Webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `opencv-python`, `numpy`, `mediapipe`, `tensorflow`

---

## Usage

### Run the webcam script

```bash
python webcam_script.py
```

Press **Q** to quit the webcam window.

### Run the notebook

Open `model.ipynb` in Jupyter or VS Code. The notebook covers:
1. Dataset preprocessing (face detection + cropping via dlib HOG)
2. CNN model definition and training
3. Evaluation — Classification Report and Confusion Matrix
4. Live webcam inference

> **Note:** Dataset preparation and model training cells were originally run on Google Colab. To run locally, update the dataset paths in the notebook to match your local directory structure.

---

## Model

| Component         | Detail                              |
|-------------------|-------------------------------------|
| Architecture      | CNN (3 Conv blocks + Dense head)    |
| Input             | 64 × 64 grayscale face crop         |
| Output classes    | Alert, Drowsy_Eyes_Closed, Drowsy_Yawning |
| Face detection    | MediaPipe Face Landmarker (478 pts) |
| Smoothing         | 15-frame majority vote + 1s lock    |
| Confidence cutoff | 0.75                                |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.