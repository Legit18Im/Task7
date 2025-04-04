import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import os
import joblib
import tensorflow as tf
import config
from female_voice_check import is_female_voice
from utils import predict_emotion_with_probs
from visualization import plot_waveform, plot_spectrogram

class VoiceEmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Emotion Detection")
        self.root.geometry("400x300")
        
        self.label = tk.Label(root, text="Select an audio file for analysis:")
        self.label.pack(pady=10)
        
        self.upload_button = tk.Button(root, text="Upload Audio", command=self.process_audio)
        self.upload_button.pack(pady=10)
        
        self.result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)
    
    def process_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if not file_path:
            return
        
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
