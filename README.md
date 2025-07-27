# 🎧 Music Analyzer (AI + Audio Visualizer)

A complete audio analysis tool powered by **Spleeter**, **YAMNet**, and **Librosa**, built with Python and a simple GUI.

---

## 🚀 Features

✅ Convert MP3 to WAV  
✅ Separate stems (vocals, drums, bass, others) using Spleeter  
✅ Classify each stem using Google's YAMNet model  
✅ Extract deep audio insights:
- Tempo, Pitch, Loudness, Dynamic Range
- Spectral Bandwidth, Rolloff, Zero-Crossing Rate
- Sibilance Ratio, Reverb Estimate, Stereo Detection
- Clipping Detection

✅ Automatically generate:
- 📈 Waveform & Spectrogram Plots (per stem)
- 📊 Classification Summary CSV
- 📑 Advanced Audio Features (JSON)
✅ Show results in a scrollable GUI table  
✅ Save all results into a new folder (`output/001`, `002`, etc.)

---

## 📁 Folder Structure (Auto-Generated)

Every time you run the tool, a new folder is created:


✅ All files are saved uniquely – no overwriting  
✅ You can input multiple MP3s one by one

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt


to run GUI:
python music_analyser.py


to edit the json file:
python edit_audio_features.py


🤖 Tech Stack
Python 3.10+

TensorFlow + TensorFlow Hub
Librosa
Matplotlib
Spleeter
Tkinter (GUI)

🧠 Credits
YAMNet (Google TensorFlow Hub)
Spleeter (by Deezer)
Librosa

🙌 Contribution
PRs welcome! Report bugs or drop feature requests via issues.