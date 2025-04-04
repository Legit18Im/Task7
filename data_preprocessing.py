# data_preprocessing.py

import os
import librosa
import numpy as np
import pickle
import config

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def get_emotion_from_filename(filename):
    filename = filename.lower()
    if "angry" in filename:
        return "angry"
    elif "disgust" in filename:
        return "disgust"
    elif "fear" in filename:
        return "fear"
    elif "happy" in filename:
        return "happy"
    elif "sad" in filename:
        return "sad"
    elif "neutral" in filename:
        return "neutral"
    elif "pleasant" in filename or "surprise" in filename or "ps" in filename:
        return "surprise"
    else:
        return None

def load_data():
    X, y = [], []
    for root, dirs, files in os.walk(config.DATASET_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            emotion = get_emotion_from_filename(file)
            if emotion is None:
                print(f"Skipping unknown emotion: {file}")
                continue
            features = extract_features(file_path)
            X.append(features)
            y.append(emotion)
    return np.array(X), np.array(y)

X, y = load_data()
os.makedirs('/content/drive/MyDrive/Task_7/processed_data', exist_ok=True)
with open('/content/drive/MyDrive/Task_7/processed_data/processed_data.pkl', 'wb') as f:
    pickle.dump((X, y), f)
print(f"Saved processed data with {len(X)} samples.")