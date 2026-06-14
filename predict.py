
import numpy as np
import librosa
import tensorflow as tf
import sys

# Model load karo
model = tf.keras.models.load_model('best_model.h5')

def predict_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000, duration=4.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 200:
        mfcc = np.pad(mfcc, ((0,0),(0, 200 - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :200]
    
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    prob = model.predict(mfcc)[0][0]
    
    label = "FAKE (AI Generated)" if prob > 0.5 else "REAL (Human)"
    confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
    
    print(f"Result     : {label}")
    print(f"Confidence : {confidence:.2f}%")

if __name__ == "__main__":
    file_path = sys.argv[1]
    predict_audio(file_path)
