# Audio / spectrogram utilities
import numpy as np
import librosa
import random
import matplotlib.pyplot as plt
import io
import base64

SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 1024
HOP_LENGTH = 512
SEG_SECONDS = 15
MAX_SEG_LEN = 240

def extract_mel_spectrogram(audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann')
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T

def z_normalize(feature):
    mean = np.mean(feature, axis=0)
    std = np.std(feature, axis=0) + 1e-8
    return (feature - mean) / std

def pad_to_length(mel, target_frames=MAX_SEG_LEN):
    t, f = mel.shape
    if t >= target_frames:
        return mel[:target_frames, :]
    else:
        pad = np.zeros((target_frames - t, f), dtype=mel.dtype)
        return np.vstack([mel, pad])

def segment_audio_by_seconds(audio, sr=SAMPLE_RATE, seg_seconds=SEG_SECONDS):
    seg_len = seg_seconds * sr
    segments = []
    for start in range(0, len(audio), seg_len):
        end = min(len(audio), start + seg_len)
        seg = audio[start:end]
        segments.append(seg)
    return segments

# Spectrogram plotting
def plot_spectrogram_base64(audio_np, sr=SAMPLE_RATE, title=None):
    import librosa.display

    mel = librosa.feature.melspectrogram(
        y=audio_np,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(6, 3))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        cmap='magma'
    )
    plt.colorbar(format="%+2.0f dB")
    
    # 🔹 Remove title text (to avoid transcript overlap)
    # Instead, transcripts will be shown separately in the report, not on the plot
    
    plt.tight_layout(pad=1.5)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

