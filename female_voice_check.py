import librosa
import numpy as np



def is_female_voice(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    
    if len(y) < sr:  # Ignore clips shorter than 1 second
        return False
    
    f0, _, _ = librosa.pyin(y, fmin=140, fmax=300)  # Adjusted pitch range
    pitches = f0[~np.isnan(f0)]
    
    if len(pitches) == 0:
        return False

    avg_pitch = np.mean(pitches)
    return 140 <= avg_pitch <= 300  # Updated threshold
