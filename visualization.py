import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import uuid

def plot_waveform(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    filename = f"waveform_{uuid.uuid4().hex}.png"
    save_path = os.path.join("static", filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    stft = librosa.stft(y)
    stft_db = librosa.amplitude_to_db(abs(stft))
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    filename = f"spectrogram_{uuid.uuid4().hex}.png"
    save_path = os.path.join("static", filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path
