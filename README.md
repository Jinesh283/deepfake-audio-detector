# 🎙️ Deepfake Audio Detector

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-90.74%25-green)

## 📌 Project Description
A deep learning system that classifies speech recordings as **Real (Human)** or **Fake (AI Generated)** using CNN on MFCC features.

---

## 🏆 Results

| Metric | Score | Required | Status |
|--------|-------|----------|--------|
| Accuracy | 90.74% | 80% | ✅ PASS |
| F1 Score | 90.10% | 80% | ✅ PASS |
| EER | 3.17% | 12% | ✅ PASS |
| Real Accuracy | 99.5% | 75% | ✅ PASS |
| Fake Accuracy | 82.4% | 75% | ✅ PASS |

---

## 📂 Dataset
- Dataset: Fake-or-Real (Kaggle)
- Folder: for-norm
- Training: 53,868 files
- Validation: 10,798 files
- Testing: 4,634 files

---

## 🧠 Methodology

### 1. Preprocessing
- Audio loaded at 16kHz, clipped to 4 seconds
- Converted to MFCC features (40 coefficients x 200 frames)

### 2. Feature Extraction
- MFCC (Mel Frequency Cepstral Coefficients)
- Each audio file converted to 40x200 grid

### 3. Model Architecture (CNN)
- Input (40, 200, 1)
- Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
- Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
- Conv2D(128) + BatchNorm + MaxPool + Dropout(0.30)
- Flatten
- Dense(256) + Dropout(0.40)
- Dense(1, sigmoid)
- Total Parameters: 4,190,081
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 20

---

## 📁 Project Structure
- deepfake_audio.ipynb — Full training notebook
- app.py — Streamlit web app
- predict.py — CLI prediction script
- best_model.h5 — Trained model
- report/ — Confusion matrix and training plots

---

## 🚀 How to Run

Install dependencies:
pip install tensorflow librosa streamlit scikit-learn

Run app:
streamlit run app.py

Test single file:
python predict.py your_audio.wav

---

## 🌐 Web App Features
- Upload Audio: WAV/MP3/FLAC supported
- Live Record: Record from browser mic
- Shows Real/Fake result with confidence score

---

## 📊 Confusion Matrix

| | Predicted Real | Predicted Fake |
|---|---|---|
| Actual Real | 2253 ✅ | 11 ❌ |
| Actual Fake | 418 ❌ | 1952 ✅ |

---

## 👨‍💻 Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow/Keras | Model training |
| Librosa | Audio processing |
| Scikit-learn | Metrics |
| Streamlit | Web app |
| Google Colab | Training |
| ngrok | Deployment |
