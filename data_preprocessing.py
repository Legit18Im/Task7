import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET_PATH = "/kaggle/input/TESS_Toronto_emotional_speech_set"
EMOTIONS = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "pleasant": 5, "sad": 6}


def extract_features(file_path, max_pad_len=200):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)  # Increased MFCCs
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs


def load_dataset():
    features, labels = [], []
    for emotion, label in EMOTIONS.items():
        emotion_path = os.path.join(DATASET_PATH, emotion)
        for file in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, file)
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)


X, y = load_dataset()
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  # Reshape for CNN input
y = tf.keras.utils.to_categorical(y, num_classes=len(EMOTIONS))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.savez("/kaggle/working/processed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("Data preprocessing completed. Processed data saved.")
