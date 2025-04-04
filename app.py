import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import librosa
import numpy as np
import os
import joblib
import tensorflow as tf
import config
import wave
from scipy.io.wavfile import write
from female_voice_check import is_female_voice
from utils import predict_emotion_with_probs
from visualization import plot_waveform, plot_spectrogram

class VoiceEmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Emotion Detection")
        self.root.geometry("400x350")
        
        self.label = tk.Label(root, text="Select or Record an audio file for analysis:")
        self.label.pack(pady=10)
        
        self.upload_button = tk.Button(root, text="Upload Audio", command=self.process_audio)
        self.upload_button.pack(pady=5)
        
        self.record_button = tk.Button(root, text="Record Voice", command=self.record_audio)
        self.record_button.pack(pady=5)
        
        self.result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=10)
    
    def process_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if not file_path:
            return
        
        self.analyze_audio(file_path)
    
    def record_audio(self, duration=5, sample_rate=22050):
        messagebox.showinfo("Recording", "Recording will start now. Speak for 5 seconds.")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()
        file_path = "recorded_voice.wav"
        write(file_path, sample_rate, audio_data)
        
        self.analyze_audio(file_path)
    
    def analyze_audio(self, file_path):
        if not is_female_voice(file_path):
            messagebox.showerror("Error", "The voice is not detected as female.")
            return
        
        probs, emotion = predict_emotion_with_probs(file_path)
        waveform_path = plot_waveform(file_path)
        spectrogram_path = plot_spectrogram(file_path)
        
        self.result_label.config(text=f"Predicted Emotion: {emotion}")
        messagebox.showinfo("Success", f"Emotion detected: {emotion}\nWaveform saved: {waveform_path}\nSpectrogram saved: {spectrogram_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceEmotionApp(root)
    root.mainloop()
