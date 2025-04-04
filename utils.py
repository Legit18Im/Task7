import os
import numpy as np
import librosa
import tensorflow as tf
import joblib
import config

# Load model and label encoder once to avoid reloading on every request
model = tf.keras.models.load_model(os.path.join(config.MODEL_SAVE_PATH, 'emotion_model.keras'))
label_encoder = joblib.load(os.path.join(config.MODEL_SAVE_PATH, 'label_encoder2.pkl'))

def extract_features(file_path):
    """Extract MFCC + delta features and pad/crop to consistent length."""
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC)
    delta = librosa.feature.delta(mfcc)
    features = np.vstack((mfcc, delta))

    # Pad or crop
    pad_width = config.MAX_PAD_LEN - features.shape[1]
    if pad_width > 0:
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :config.MAX_PAD_LEN]

    return features.T


  


def predict_emotion_with_probs(audio_path):
    features = extract_features(audio_path)
    
    # Reshape to (1, time_steps, 1) instead of (1, time_steps, 80)
    features = np.expand_dims(features, axis=-1)  # Add a single channel
    
    prediction_probs = model.predict(features)
    top_emotion_index = np.argmax(prediction_probs[0])
    top_emotion = label_encoder.inverse_transform([top_emotion_index])[0]
    return prediction_probs, top_emotion
