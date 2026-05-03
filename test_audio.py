import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

audio_file_path = "data/speech_commands_v0.02/happy/0a2b400e_nohash_0.wav"

# Load audio file.
audio_data, sample_rate = librosa.load(audio_file_path, sr = None)

# audio_data, sample_rate = sf.read(audio_file_path) 

print(f"Sampe rate: {sample_rate} Hz")
print(f"Number of samples: {len(audio_data)}")
print(f"Duration: {len(audio_data) / sample_rate: .2f} seconds")

# Visualize the wave form
plt.figure(figsize=(10,5))
librosa.display.waveshow(y = audio_data, sr = sample_rate)
plt.title("Waveform of the audio file.")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.show()