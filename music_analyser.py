# music_analyser.py

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import shutil
import json
import tkinter as tk
from tkinter import filedialog, ttk
import gc
import tensorflow as tf

# Monkey patch for NumPy compatibility
if not hasattr(np, 'complex'):
    np.complex = np.complex128

# Safe GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU memory growth enabled")
    except Exception as e:
        print("⚠️ Could not set GPU memory growth:", str(e))
else:
    print("🧠 No GPU detected, using CPU mode")

# Get next output folder like 001, 002, etc.
def get_next_run_folder(base_dir="output"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.isdigit()]
    numbers = [int(n) for n in existing] if existing else [0]
    next_id = max(numbers) + 1 if numbers else 1
    new_folder = os.path.join(base_dir, f"{next_id:03d}")
    os.makedirs(new_folder)
    return new_folder

# Convert MP3 to WAV + apply Spleeter
def convert_and_separate(mp3_path, out_base):
    from spleeter.separator import Separator
    base = os.path.splitext(os.path.basename(mp3_path))[0]
    wav_path = os.path.join(out_base, "wav_files", base + ".wav")
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    y, sr = librosa.load(mp3_path, sr=16000, mono=True)
    sf.write(wav_path, y, sr)
    separator = Separator("spleeter:4stems")
    separator.separate_to_file(wav_path, os.path.join(out_base, "separated_audio"))
    stems_path = os.path.join(out_base, "separated_audio", base)
    return base, stems_path

# Extract audio features
def extract_features(path):
    y, sr = librosa.load(path, sr=None)
    features = {}
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["Tempo (BPM)"] = tempo
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["Estimated Key"] = f"Pitch class {np.argmax(np.mean(chroma, axis=1))}"
    rms = librosa.feature.rms(y=y)
    features["Loudness (RMS)"] = float(np.mean(rms))
    features["Dynamic Range"] = float(np.max(rms) - np.min(rms))
    features["Spectral Centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features["Spectral Bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features["Spectral Rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = librosa.feature.zero_crossing_rate(y)
    features["Zero Crossing Rate"] = float(np.mean(zcr))
    features["Clipping Detected"] = bool(np.any(np.abs(y) >= 0.99))
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    features["Sibilance Ratio"] = float(np.sum(S[freqs > 4000]) / np.sum(S))
    features["Stereo Correlation"] = "Mono" if y.ndim == 1 else float(np.corrcoef(y[0], y[1])[0, 1])
    features["Reverb Estimate"] = "High" if float(np.mean(zcr)) > 0.1 else "Low"
    return features

# Save waveform and spectrogram plots
def plot_visuals(path, out_dir):
    y, sr = librosa.load(path, sr=None)
    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_waveform.png"))
    plt.close()

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_spectrum.png"))
    plt.close()

# Display features in a popup
def show_popup(all_features):
    win = tk.Toplevel()
    win.title("🎧 Audio Diagnostics Summary")
    win.geometry("800x500")
    tree = ttk.Treeview(win, columns=("File", "Feature", "Value"), show="headings")
    tree.heading("File", text="File")
    tree.heading("Feature", text="Feature")
    tree.heading("Value", text="Value")
    tree.pack(fill="both", expand=True)
    for file, features in all_features.items():
        for k, v in features.items():
            tree.insert("", "end", values=(file, k, v))

# GUI Main
def run_gui():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
    if not file_path:
        return

    output_dir = get_next_run_folder("output")
    base, stems = convert_and_separate(file_path, output_dir)
    all_features = {}

    for f in os.listdir(stems):
        if f.endswith(".wav"):
            path = os.path.join(stems, f)
            plot_visuals(path, os.path.join(output_dir, "plots"))
            feats = extract_features(path)
            all_features[f] = feats

    with open(os.path.join(output_dir, "advanced_features.json"), "w") as f:
        json.dump(all_features, f, indent=2)

    show_popup(all_features)
    tf.keras.backend.clear_session()
    gc.collect()

if __name__ == "__main__":
    run_gui()
