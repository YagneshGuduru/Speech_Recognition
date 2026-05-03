import librosa
import numpy as np

def extract_mfccs(audio_path, sr = 16000, n_mfcc = 13, n_fft = 2048, hop_length = 512):
    
    audio, _ = librosa.load(audio_path, sr = sr)

    mfccs = librosa.feature.mfcc(
        y = audio,
        sr = sr,
        n_mfcc = n_mfcc,
        n_fft = n_fft,
        hop_length = hop_length
    )

    return mfccs
