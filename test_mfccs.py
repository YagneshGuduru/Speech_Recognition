from src.feature_extraction import extract_mfccs
import matplotlib.pyplot as plt
import librosa.display

file_path = "data/speech_commands_v0.02/happy/0a2b400e_nohash_0.wav"
mfccs = extract_mfccs(file_path)
print("MFCC Shape: ", mfccs.shape)

plt.figure(figsize=(10,5))
librosa.display.specshow(mfccs, x_axis = 'time')
plt.colorbar()
plt.title('MFCCS')
plt.tight_layout()
plt.show()
