# app_v2.py
# Flask app that accepts an audio file and returns analysis JSON (and can render results.html if present).
# Includes upgraded engineering metrics: Reverb (late/early + RT60 proxy + pre-delay), Phasing/Mono, Time Signature.
# Safe to run without heavy ML models; genre/emotion are stubbed unless available locally.

import os
import io
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import soundfile as sf
import librosa

from flask import Flask, request, jsonify, render_template, send_from_directory

# ------------------------------
# Config
# ------------------------------
TARGET_SR = 44100
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
STEMS_DIR = Path(os.getenv("STEMS_DIR", "./stems"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STEMS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

# ------------------------------
# Optional heavy models (safe imports)
# ------------------------------

class ASTClassifier:
    def __init__(self, device: str = "cpu"):
        self.device = device
    def predict_proba_over(self, path: str, labels: List[str]) -> Dict[str, float]:
        # stub uniform prediction
        if not labels:
            return {}
        p = 1.0 / len(labels)
        return {lab: p for lab in labels}

class MERTGenre:
    def __init__(self, device: str = "cpu", label_list: Optional[List[str]] = None):
        self.device = device
        self.labels = label_list or ["Pop","Rock","HipHop","EDM","Classical","Jazz","Folk","Metal"]
    def predict_top(self, path: str) -> Tuple[str, Dict[str,float]]:
        # stub: random-ish by spectral centroid
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
        c = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        idx = int((c / (sr/2)) * (len(self.labels)-1))
        idx = max(0, min(idx, len(self.labels)-1))
        top = self.labels[idx]
        probs = {lab: (0.6 if lab==top else 0.4/(len(self.labels)-1)) for lab in self.labels}
        return top, probs

GENRE_LABELS = ["Pop","Rock","HipHop","EDM","Classical","Jazz","Folk","Metal"]

def fuse_genre_probs(ast_probs: Dict[str,float], mert_probs: Dict[str,float], labels: List[str], w_ast=0.4, w_mert=0.6):
    labels = labels or list(set(list(ast_probs.keys()) + list(mert_probs.keys())))
    items = []
    for lab in labels:
        pa = ast_probs.get(lab, 0.0)
        pm = mert_probs.get(lab, 0.0)
        items.append((lab, w_ast*pa + w_mert*pm))
    items.sort(key=lambda x: x[1], reverse=True)
    top = items[0][0] if items else "Unknown"
    return top, items

# Music2Emotion (stub)
class _M2EStub:
    def predict_full(self, path: str) -> Dict[str, Any]:
        # very naive: tempo & centroid to moods
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        bright = float(cent / (sr/2))
        mood_tags = [("energetic" if tempo>120 else "calm", 0.6),
                     ("bright" if bright>0.45 else "warm", 0.4)]
        return {"mood_tags": mood_tags, "valence": round(4 + 4*(bright-0.5),2), "arousal": round(4 + 4*((tempo-90)/90),2)}

m2e_ensemble = _M2EStub()

# ------------------------------
# Utils
# ------------------------------
def load_audio_stereo(path: str, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    y, _ = librosa.load(path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    y = y.astype(np.float32, copy=False)
    return y, sr

def to_mono(y_stereo: np.ndarray) -> np.ndarray:
    if y_stereo.ndim == 1: return y_stereo
    return librosa.to_mono(y_stereo)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(float)**2) + 1e-12))

# ------------------------------
# Engineering metrics
# ------------------------------
def _reverb_room(y, sr):
    """
    Estimate reverb/room characteristics from a full mix.
    Returns:
      - late_early_db: late-to-early energy ratio in dB (higher -> wetter)
      - grade: human label (dry/moderate/wet)
      - decay_rt60_s: RT60 proxy (Schroeder EDC T30 * 2), seconds
      - pre_delay_ms: first early-reflection gap estimate in ms (onset env autocorr)
    """
    y_mono = to_mono(y)
    # Spectral energy split early/late for wet/dry feel
    S = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512))**2
    early = S[:, :S.shape[1]//3].mean()
    late = S[:, S.shape[1]//3:].mean() + 1e-12
    ldr = float(10*np.log10(late/early + 1e-12))
    grade = "dry" if ldr < -3 else "moderate" if ldr < 1 else "wet"

    # RT60 proxy via Schroeder integration (T30 -> RT60 ~ 2*T30)
    e = y_mono.astype(float)**2
    if e.size == 0:
        decay_rt60_s = None
    else:
        edc = np.cumsum(e[::-1])[::-1]
        edc = edc / (edc[0] + 1e-12)
        edc_db = 10.0*np.log10(edc + 1e-12)
        def _first_idx_below(th_db):
            idx = np.where(edc_db <= th_db)[0]
            return int(idx[0]) if idx.size else None
        i5 = _first_idx_below(-5.0)
        i35 = _first_idx_below(-35.0)
        if i5 is not None and i35 is not None and i35 > i5:
            t30 = (i35 - i5) / sr
            decay_rt60_s = round(float(2.0 * t30), 2)
        else:
            decay_rt60_s = None

    # Pre-delay proxy: autocorrelation of onset strength envelope
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr, hop_length=hop)
    pre_delay_ms = None
    if onset_env.size > 8:
        ac = librosa.autocorrelate(onset_env, max_size=int(0.25 * sr / hop) + 2)  # up to ~250ms
        min_lag = int(0.010 * sr / hop)
        max_lag = int(0.080 * sr / hop)
        if max_lag > min_lag and max_lag < ac.shape[0]:
            window = ac[min_lag:max_lag]
            if window.size > 0:
                peak_rel = int(np.argmax(window))
                lag = min_lag + peak_rel
                pre_delay_ms = round(float(lag * hop / sr * 1000.0), 1)

    return {
        "late_early_db": round(ldr, 2),
        "grade": grade,
        "decay_rt60_s": decay_rt60_s,
        "pre_delay_ms": pre_delay_ms
    }

def _phasing_check(y_stereo):
    """
    Detect potential phasing/mono-compatibility issues.
    Returns {"issue": bool, "corr": r, "mono_drop_db": dB, "note": str}
    """
    if y_stereo.shape[0] == 1:
        y_stereo = np.vstack([y_stereo, y_stereo])
    L, R = y_stereo[0], y_stereo[1]
    if L.size == 0 or R.size == 0:
        return {"issue": False, "corr": 1.0, "mono_drop_db": 0.0, "note": "n/a"}
    corr = float(np.corrcoef(L, R)[0,1])
    mid = 0.5*(L + R)
    side = 0.5*(L - R)
    stereo_rms = np.sqrt((rms(L)**2 + rms(R)**2)/2.0)
    mono_rms = rms(mid)
    drop_db = 20*np.log10((mono_rms + 1e-12)/(stereo_rms + 1e-12))
    issue = (corr < -0.10) or (drop_db < -3.0) or (rms(side) > rms(mid)*1.2)
    note = "Possible phase cancellation" if issue else "No major issue"
    return {"issue": bool(issue), "corr": round(corr,3), "mono_drop_db": round(float(drop_db),2), "note": note}

def _time_signature(y, sr):
    """
    Guess 3/4 vs 4/4 using beat-strength periodicity.
    Returns {"signature": "4/4"|"3/4"|"unknown", "confidence": 0..1}
    """
    y_mono = to_mono(y)
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr, hop_length=hop)
    if onset_env.size < 16:
        return {"signature":"unknown", "confidence": 0.0}
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop, trim=False)
    if beats.size < 8:
        return {"signature":"unknown", "confidence": 0.0}
    beat_env = onset_env[beats]
    z = (beat_env - beat_env.mean()) / (beat_env.std() + 1e-9) if beat_env.std() > 1e-6 else beat_env*0.0
    def pattern_score(period):
        mask = np.zeros_like(z)
        mask[::period] = 1.0
        best = -1e9
        for s in range(period):
            score = float(np.dot(z, np.roll(mask, s))) / (len(z) + 1e-9)
            best = max(best, score)
        return best
    s3 = pattern_score(3); s4 = pattern_score(4)
    if max(s3, s4) < 0.05:
        return {"signature":"unknown", "confidence": 0.0}
    if s4 >= s3:
        conf = (s4 - min(0.0, s3)) / (abs(s4) + abs(s3) + 1e-9)
        return {"signature":"4/4", "confidence": round(float(max(0.0, min(1.0, conf))),2)}
    else:
        conf = (s3 - min(0.0, s4)) / (abs(s4) + abs(s3) + 1e-9)
        return {"signature":"3/4", "confidence": round(float(max(0.0, min(1.0, conf))),2)}

def _tempo(y, sr):
    t = float(librosa.beat.tempo(y=to_mono(y), sr=sr, aggregate=None).mean())
    return {"bpm": round(t, 2)}

def _frequency_masking(y, sr):
    """
    Very simple band energy percentages as a proxy for masking risk.
    """
    y_mono = to_mono(y)
    S = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bands = {
        "Sub (20-60Hz)": (20, 60),
        "Bass (60-150Hz)": (60, 150),
        "Low-Mid (150-500Hz)": (150, 500),
        "Mid (500-2kHz)": (500, 2000),
        "High-Mid (2-6kHz)": (2000, 6000),
        "Air (6-16kHz)": (6000, 16000),
    }
    total = S.sum() + 1e-12
    out = []
    for name, (lo, hi) in bands.items():
        idx = np.where((freqs>=lo) & (freqs<hi))[0]
        share = float(S[idx,:].sum() / total) if idx.size else 0.0
        out.append((name, round(share,4)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def _structure(y, sr, k=6):
    """
    Rough musical structure via spectral clustering of MFCCs.
    """
    y_mono = to_mono(y)
    mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=20)
    # Recurrence matrix + novelty score
    R = librosa.segment.recurrence_matrix(mfcc, width=3, mode='affinity', sym=True)
    # Laplacian segmentation
    seg = librosa.segment.agglomerative(R, k=k)
    # Convert to times
    hop = 512
    seg_times = librosa.frames_to_time(np.r_[0, np.flatnonzero(np.diff(seg)) + 1, mfcc.shape[1]], sr=sr, hop_length=hop)
    labels = []
    for i in range(len(seg_times)-1):
        start = float(seg_times[i]); end = float(seg_times[i+1])
        label = f"Section {i+1}"
        labels.append({"label": label, "start": round(start,1), "end": round(end,1)})
    return labels

# ------------------------------
# Orchestrator
# ------------------------------
def run_engineering_analysis(src_path: str, saved_stems: Optional[Dict[str,str]] = None) -> Dict[str, Any]:
    y, sr = load_audio_stereo(src_path, sr=TARGET_SR)
    y_mono = to_mono(y)

    # Core analyses
    tempo = _tempo(y, sr)
    reverb = _reverb_room(y_mono, sr)
    phasing = _phasing_check(y)
    time_sig = _time_signature(y_mono, sr)
    masking = _frequency_masking(y_mono, sr)
    structure = _structure(y_mono, sr)

    return {
        "tempo": tempo,
        "reverb": reverb,
        "phasing": phasing,
        "time_signature": time_sig,
        "masking": masking,
        "structure": structure,
    }

def curate_analysis(filename: str, genre: str, mood_tags: List[Tuple[str,float]], val: Optional[float], aro: Optional[float],
                    saved_stems: Dict[str,str], ast_top: str="", mert_top: str="", fused_probs: List[Tuple[str,float]] = None) -> Dict[str,Any]:
    moods = ", ".join([m for m,_ in (mood_tags or [])][:4])
    text = f"{filename}: {genre or 'Unknown genre'} with moods {moods or 'n/a'}. "
    if val is not None and aro is not None:
        text += f"Valence {val}, Arousal {aro}. "
    if saved_stems:
        text += f\"Stems: {', '.join(saved_stems.keys())}. \"
    return {"summary_text": text.strip(), "dominant_family": genre or "Unknown"}

# High-level composition
def run_full_analysis(src_path: str) -> Dict[str, Any]:
    # 1) Stems (skipped unless you wire Demucs)
    saved_stems = {}

    # 2) Genre (AST+MERT stubs)
    ast_probs = ASTClassifier().predict_proba_over(src_path, GENRE_LABELS)
    ast_top = max(ast_probs.items(), key=lambda x: x[1])[0] if ast_probs else "Unknown"
    mert_top, mert_probs = MERTGenre(label_list=GENRE_LABELS).predict_top(src_path)
    fused_top, fused_items = fuse_genre_probs(ast_probs, mert_probs, GENRE_LABELS)

    # 3) Emotion
    emo = m2e_ensemble.predict_full(src_path)
    mood_tags = emo.get("mood_tags", [])
    val = emo.get("valence"); aro = emo.get("arousal")

    # 4) Engineering
    eng = run_engineering_analysis(src_path, saved_stems)

    # 5) Curated
    curated = curate_analysis(Path(src_path).name, fused_top, mood_tags, val, aro, saved_stems,
                              ast_top=ast_top, mert_top=mert_top, fused_probs=fused_items)

    return {
        "file": Path(src_path).name,
        "genre": {
            "fused_top": fused_top,
            "ast_top": ast_top,
            "mert_top": mert_top,
            "fused_probs": fused_items,
        },
        "moods": mood_tags,
        "valence": val,
        "arousal": aro,
        "stems": saved_stems,
        "curated": curated,
        "engineering": eng,
    }

# ------------------------------
# Flask routes
# ------------------------------
@app.route("/", methods=["GET"])
def home():
    # If you still have templates/index.html, render it; else show a basic info page.
    try:
        return render_template("index.html")
    except Exception:
        return "<h3>AI Music Analyzer</h3><p>POST an audio file to /analyze-json or use the React UI.</p>"

@app.route("/analyze-json", methods=["POST"])
def analyze_json():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "file missing"}), 400
    suf = Path(f.filename or "audio").suffix or ".wav"
    tmp = UPLOAD_DIR / f"up_{os.getpid()}_{np.random.randint(1e9)}{suf}"
    with open(tmp, "wb") as o:
        o.write(f.read())
    try:
        result = run_full_analysis(str(tmp))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
