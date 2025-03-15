import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from data_preprocessing import extract_features

# Load trained model
model = tf.keras.models.load_model("/kaggle/working/emotion_model.keras")
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant", "sad"]

def predict_emotion(file_path):
    feature = extract_features(file_path)
    feature = feature.reshape(1, feature.shape[0], feature.shape[1], 1)
    prediction = model.predict(feature)
    return EMOTIONS[np.argmax(prediction)]

st.title("Emotion Detection through Voice")
uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])

if uploaded_file is not None:
    file_path = f"/kaggle/working/temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    emotion = predict_emotion(file_path)
    st.write(f"Predicted Emotion: {emotion}")
