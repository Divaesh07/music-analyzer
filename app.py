import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
from io import BytesIO
from tempfile import NamedTemporaryFile

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort

import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
import torch.nn as nn
from scipy.signal import correlate, butter, filtfilt

# Extra dependency for LUFS:
# pip install pyloudnorm
import pyloudnorm as pyln

# ============== CONFIG ==============

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "audio_analysis_results"
STEMS_DIR = RESULTS_DIR / "stem"  # only directory we keep

# Point this to your trimmed repo
M2E_REPO = BASE_DIR / "Music2Emotion"
SAVED_MODELS_DIR = M2E_REPO / "saved_models"

USE_GPU = True
DEVICE = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
AMP = (DEVICE == "cuda")

# Ensure output dirs
os.makedirs(STEMS_DIR, exist_ok=True)

# ============== IMPORT PIPELINE PARTS ==============

# 1) Demucs
from demucs.pretrained import get_model as get_demucs_model
from demucs.apply import apply_model

# 2) AST (AudioSet) for genre/instruments
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# 3) Music2Emotion
if str(M2E_REPO) not in sys.path:
    sys.path.insert(0, str(M2E_REPO))
try:
    import music2emo as _m2e_mod
    Music2emo = getattr(_m2e_mod, "Music2emo")
except Exception as e:
    raise RuntimeError(f"Failed to import Music2Emotion: {e}")

# 4) MERT encoder
from transformers import AutoProcessor, AutoModel


def log(*a):
    print("[app]", *a)


# ============== AUDIO HELPERS ==============

def load_audio_any(path, sr=None, mono=False):
    y, fs = librosa.load(path, sr=sr, mono=mono)
    return y, fs

def to_mono(y):
    return librosa.to_mono(y) if y.ndim > 1 else y

def resample(y, sr, target_sr):
    if sr == target_sr:
        return y, sr
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr), target_sr

def normalize_peak(y, peak=0.98):
    m = np.max(np.abs(y)) + 1e-9
    return y * (peak / m)

def bytes_to_tempfile(file_storage) -> str:
    data = file_storage.read()
    file_storage.stream.seek(0)
    tmp = NamedTemporaryFile(delete=False, suffix=Path(file_storage.filename).suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name

def write_wav_16k_mono(src_path: str) -> str:
    y, sr = librosa.load(src_path, sr=None, mono=False)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32)
    tmp = NamedTemporaryFile(delete=False, suffix="_16kmono.wav")
    sf.write(tmp.name, y16, 16000)
    tmp.flush()
    tmp.close()
    return tmp.name


def separate_with_demucs(audio_path, device=DEVICE):
    try:
        model = get_demucs_model('htdemucs').to(device).eval()
        wav, sr = torchaudio.load(audio_path)  # [C, T]
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        if sr != 44100:
            wav = torchaudio.transforms.Resample(sr, 44100)(wav)
            sr = 44100
        wav = wav.to(device)
        with torch.inference_mode():
            if AMP:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    sources = apply_model(model, wav.unsqueeze(0))[0]
            else:
                sources = apply_model(model, wav.unsqueeze(0))[0]
        names = ['drums', 'bass', 'other', 'vocals']
        out = {n: sources[i].detach().to("cpu").numpy() for i, n in enumerate(names)}
        return out, sr
    except Exception as e:
        log("Demucs error:", e)
        return {}, None


def save_audio_stem(inst_label, stem_audio, sr, audio_path):
    out_dir = STEMS_DIR / inst_label
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(audio_path).stem
    out_path = out_dir / f"{base}_{inst_label}.wav"
    if stem_audio.ndim == 2:  # [C, T] -> [T, C]
        stem_audio = stem_audio.T
    sf.write(str(out_path), stem_audio, sr)
    return str(out_path)


# ============== AST CLASSIFIER ==============

class ASTClassifier:
    GENRE_WHITELIST = {
        "Pop music","Rock music","Hip hop music","Jazz","Classical music",
        "Electronic music","Metal","Blues","Reggae","Country",
        "Funk","Soul music","Dance music","Techno","House music",
        "Trance music","Disco","R&B","Gospel music","Folk music","Latin music",
        "Indie rock","Alternative rock","Punk rock","Grunge","Ambient music"
    }

    def __init__(self, device=DEVICE):
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = AutoModelForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        ).to(self.device).eval()
        self.id2label = self.model.config.id2label

    def _chunks(self, x, sr, seconds=10.0):
        win = int(seconds * sr)
        if len(x) <= win:
            yield x
            return
        for s in range(0, len(x), win):
            e = min(s + win, len(x))
            yield x[s:e]

    def predict_proba_over(self, path, target_labels):
        try:
            audio, _ = librosa.load(path, sr=16000, mono=True)
            logits_sum, n = None, 0
            with torch.inference_mode():
                for ch in self._chunks(audio, 16000, 10.0):
                    inputs = self.fe(ch, sampling_rate=16000, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    if AMP:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = self.model(**inputs).logits
                    else:
                        logits = self.model(**inputs).logits
                    v = logits[0].detach().to("cpu").numpy()
                    logits_sum = v if logits_sum is None else (logits_sum + v)
                    n += 1
            if not n:
                return {lab: 0.0 for lab in target_labels}
            avg = logits_sum / n
            probs = np.exp(avg - np.max(avg)); probs = probs / probs.sum()
            ast_map = {self.id2label[int(i)]: float(probs[i]) for i in range(len(probs))}
            out = {lab: ast_map.get(lab, 0.0) for lab in target_labels}
            s = sum(out.values())
            if s > 0:
                for k in out:
                    out[k] /= s
            return out
        except Exception as e:
            log("AST proba mapping error:", e)
            return {lab: 0.0 for lab in target_labels}


# ============== MERT GENRE ==============

class MERTGenre:
    def __init__(self, device="cpu", label_list=None, chunk_sec=10.0, model_name="m-a-p/MERT-v1-95M"):
        self.device = device
        self.chunk_sec = float(chunk_sec)
        self.target_sr = 24000
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=False).to(self.device).eval()
        self.labels = label_list or [
            "Pop music","Rock music","Hip hop music","Jazz","Classical music",
            "Electronic music","Metal","Blues","Reggae","Country",
            "Funk","Soul music","Dance music","Techno","House music",
            "Trance music","Disco","R&B","Gospel music","Folk music","Latin music",
            "Indie rock","Alternative rock","Punk rock","Grunge","Ambient music"
        ]
        hidden = 768
        self.head = nn.Linear(hidden, len(self.labels)).to(self.device).eval()

    def _chunks(self, x: np.ndarray, sr: int):
        win = int(self.chunk_sec * sr)
        if win <= 0 or len(x) <= win:
            yield x
            return
        for s in range(0, len(x), win):
            e = min(s + win, len(x))
            yield x[s:e]

    @torch.inference_mode()
    def predict_proba(self, path: str) -> dict:
        x, _ = librosa.load(path, sr=self.target_sr, mono=True)
        logits_sum = None
        n = 0
        for ch in self._chunks(x, self.target_sr):
            proc = self.processor(raw_speech=[ch], sampling_rate=self.target_sr, return_tensors="pt", padding=True)
            proc = {k: v.to(self.device) for k, v in proc.items()}
            if AMP and self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self.backbone(**proc)
            else:
                feats = self.backbone(**proc)
            hs = feats.last_hidden_state
            pooled = hs.mean(dim=1)
            logits = self.head(pooled)
            v = logits.squeeze(0).detach().to("cpu").numpy()
            logits_sum = v.astype(np.float32) if logits_sum is None else (logits_sum + v)
            n += 1
        if not n:
            return {lab: 0.0 for lab in self.labels}
        avg = logits_sum / float(n)
        m = float(np.max(avg))
        probs = np.exp(avg - m)
        s = float(probs.sum())
        probs = probs / s if s > 0 else np.zeros_like(probs, dtype=np.float32)
        return {lab: float(p) for lab, p in zip(self.labels, probs.astype(float).tolist())}

    @torch.inference_mode()
    def predict_top(self, path: str):
        proba = self.predict_proba(path)
        if not proba:
            return "Unknown", proba
        top_label = max(proba.items(), key=lambda x: x[1])[0]
        return top_label, proba


def fuse_genre_probs(ast_label_probs, mert_label_probs, labels, w_ast=0.4, w_mert=0.6):
    merged = {}
    for lab in labels:
        pa = float(ast_label_probs.get(lab, 0.0))
        pm = float(mert_label_probs.get(lab, 0.0))
        merged[lab] = w_ast * pa + w_mert * pm
    s = sum(merged.values())
    if s > 0:
        for k in merged:
            merged[k] /= s
    items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    top = items[0][0] if items else "Unknown"
    return top, items


# ============== CURATION HELPERS (unchanged core) ==============

def _topk_tags(mood_tags, k=5, min_prob=0.05):
    if not mood_tags:
        return []
    filt = [(lab, float(p)) for lab, p in mood_tags if p >= min_prob]
    filt.sort(key=lambda x: x[1], reverse=True)
    return filt[:k]

def _mood_family(tag):
    t = tag.lower()
    if any(k in t for k in ["happy","joy","cheer","positive","bright","excite","energetic","party","uplift"]):
        return "Positive/Energetic"
    if any(k in t for k in ["sad","melancholy","blue","down","gloom","heartbreak"]):
        return "Negative/Sad"
    if any(k in t for k in ["calm","chill","relax","mellow","soothing","ambient","soft"]):
        return "Calm/Relaxed"
    if any(k in t for k in ["angry","aggressive","intense","dark","tense","anxious"]):
        return "Tense/Aggressive"
    if any(k in t for k in ["romance","love","tender","warm"]):
        return "Warm/Tender"
    return "Mixed/Other"

def _dominant_family(mood_tags, k=5):
    top = _topk_tags(mood_tags, k=k)
    counts = {}
    for lab, _ in top:
        fam = _mood_family(lab)
        counts[fam] = counts.get(fam, 0) + 1
    if not counts:
        return "Unknown"
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

def _va_quadrant(val, aro):
    if val is None or aro is None:
        return "Unknown"
    return (
        "High Valence / High Arousal" if val >= 4.5 and aro >= 4.5 else
        "High Valence / Low Arousal" if val >= 4.5 and aro < 4.5 else
        "Low Valence / High Arousal" if val < 4.5 and aro >= 4.5 else
        "Low Valence / Low Arousal"
    )

def _confidence_band(mood_tags):
    if not mood_tags:
        return "low"
    top = _topk_tags(mood_tags, k=5, min_prob=0.0)
    if not top:
        return "low"
    best = top[0][1]
    tail = top[-1][1] if len(top) > 1 else 0.0
    spread = best - tail
    if best >= 0.5 and spread >= 0.25:
        return "high"
    if best >= 0.3 and spread >= 0.15:
        return "medium"
    return "low"

def _stem_notes(saved_stems):
    notes = []
    try:
        sizes = {}
        for k, p in saved_stems.items():
            if p and os.path.exists(p):
                sizes[k] = os.path.getsize(p)
        if not sizes:
            return notes
        total = sum(sizes.values())
        rel = {k: (v/total if total > 0 else 0.0) for k, v in sizes.items()}
        if rel.get("vocals", 0) > 0.35:
            notes.append("Vocals likely prominent.")
        if rel.get("bass", 0) > 0.30:
            notes.append("Bass-forward mix.")
        if rel.get("drums", 0) > 0.30:
            notes.append("Percussion is dominant.")
        if rel.get("other", 0) > 0.40:
            notes.append("Strong accompaniment/background elements.")
    except Exception:
        pass
    return notes

def curate_analysis(filename, genre, mood_tags, val, aro, saved_stems, ast_top=None, mert_top=None, fused_probs=None):
    top5 = _topk_tags(mood_tags, k=5, min_prob=0.05)
    fam = _dominant_family(mood_tags, k=5)
    quad = _va_quadrant(val, aro)
    conf = _confidence_band(mood_tags)
    stems_msg = _stem_notes(saved_stems)

    va_txt = f"valence {val:.2f}, arousal {aro:.2f}" if (val is not None and aro is not None) else "valence/arousal unavailable"
    summary = (
        f"{filename}: genre '{genre}' (AST: {ast_top} | MERT: {mert_top}). "
        f"Top moods: {', '.join([f'{lab} ({p:.2f})' for lab, p in top5]) if top5 else 'no strong mood tags'}. "
        f"Affect: {quad} ({va_txt}). "
        f"Dominant mood family: {fam}. "
        f"Confidence: {conf}."
    )
    if stems_msg:
        summary += " " + " ".join(stems_msg)

    return {
        "summary_text": summary,
        "dominant_family": fam,
        "va_quadrant": quad,
        "confidence": conf,
        "stem_notes": stems_msg,
        "top_moods": [{"tag": lab, "prob": float(p)} for lab, p in top5],
        "genre": {
            "fused_top": genre,
            "ast_top": ast_top,
            "mert_top": mert_top,
            "fused_probs": fused_probs or []
        },
        "valence": val,
        "arousal": aro,
        "models_used": {
            "stems": "Demucs htdemucs",
            "genre_ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "genre_mert": "m-a-p/MERT-v1-95M + linear head",
            "emotion": "Music2Emotion ensemble (.ckpt files in saved_models)"
        }
    }

# Initialize heavy models once
ast = ASTClassifier(DEVICE)
GENRE_LABELS = list(ASTClassifier.GENRE_WHITELIST)
mert_genre = MERTGenre(device=DEVICE, label_list=GENRE_LABELS, model_name="m-a-p/MERT-v1-95M")

# ============== M2E ENSEMBLE ==============

class M2ERunnerSingle:
    def __init__(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        self.repo_root = str(M2E_REPO)
        self.ckpt = ckpt_path
        def _init():
            return Music2emo(model_weights=ckpt_path)
        import torch as _torch
        orig = _torch.load
        def _load_dev(*args, **kw):
            if "map_location" not in kw:
                kw["map_location"] = torch.device(DEVICE)
            return orig(*args, **kw)
        _torch.load = _load_dev
        try:
            self.model = _init()
        finally:
            _torch.load = orig

    def predict_full(self, audio_path, wav16_cache=None):
        if wav16_cache is None:
            wav16_cache = write_wav_16k_mono(audio_path)
        cwd = os.getcwd()
        try:
            os.chdir(self.repo_root)
            out = self.model.predict(wav16_cache)
            if isinstance(out, dict):
                out.setdefault("mood_probs", {})
                out.setdefault("predicted_moods", [])
                return out
            return {"mood_probs": {}, "predicted_moods": [], "valence": None, "arousal": None}
        except Exception as e:
            log("M2E predict error:", e)
            return {"mood_probs": {}, "predicted_moods": [], "valence": None, "arousal": None}
        finally:
            os.chdir(cwd)

class M2EAllCkptEnsemble:
    def __init__(self, ckpt_dir=SAVED_MODELS_DIR):
        files = [str((Path(ckpt_dir)/f)) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        files.sort()
        if not files:
            raise RuntimeError(f"No .ckpt files found in {ckpt_dir}")
        self.runners = []
        for p in files:
            try:
                self.runners.append(M2ERunnerSingle(p))
            except Exception as e:
                log("Skip ckpt:", p, "|", e)
        if not self.runners:
            raise RuntimeError("No valid Music2Emotion checkpoints loaded.")

    def predict_full(self, audio_path):
        wav16 = write_wav_16k_mono(audio_path)
        all_probs = {}
        val_list, aro_list = [], []
        for r in self.runners:
            out = r.predict_full(audio_path, wav16_cache=wav16)
            for lab, prob in out.get("mood_probs", {}).items():
                all_probs.setdefault(lab, []).append(float(prob))
            v, a = out.get("valence"), out.get("arousal")
            if v is not None: val_list.append(float(v))
            if a is not None: aro_list.append(float(a))
        try:
            os.unlink(wav16)
        except Exception:
            pass
        merged = [(lab, float(np.mean(vs))) for lab, vs in all_probs.items()]
        merged.sort(key=lambda x: x[1], reverse=True)
        val = float(np.mean(val_list)) if val_list else None
        aro = float(np.mean(aro_list)) if aro_list else None
        return {"mood_tags": merged, "valence": val, "arousal": aro}

m2e_ensemble = M2EAllCkptEnsemble(SAVED_MODELS_DIR)


# ============== ENGINEERING ANALYSIS ==============

def _meter_lufs(y, sr):
    y_mono = to_mono(y)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y_mono)
    return loudness

def _rms_peak(y):
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    return float(rms), float(peak)

def _crest_factor(y):
    rms, peak = _rms_peak(y)
    return float(20*np.log10((peak+1e-9)/(rms+1e-12)))

def _band_energy(y, sr, f_lo, f_hi):
    S = np.abs(librosa.stft(to_mono(y), n_fft=2048, hop_length=512))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
    if len(idx)==0:
        return 0.0
    return float(np.mean(S[idx, :]))

def _ratio_band(y, sr, band, full=(20, 20000)):
    b = _band_energy(y, sr, band[0], band[1])
    f = _band_energy(y, sr, full[0], full[1]) + 1e-9
    return float(b/f)

def _spectral_features(y, sr):
    y_mono = to_mono(y)
    S = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512))
    spec_centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    spec_bw = librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).mean()
    zcr = librosa.feature.zero_crossing_rate(y_mono).mean()
    brightness = float(spec_centroid/ (sr/2))
    boominess = _ratio_band(y_mono, sr, (60, 150))
    noisiness = float(zcr)
    tonality = 1.0 - noisiness
    return {
        "brightness": round(brightness, 3),
        "boominess": round(boominess, 3),
        "noisiness": round(noisiness, 3),
        "tonal_index": round(float(tonality), 3),
        "rolloff_hz": float(rolloff)
    }

def _sibilance(y, sr):
    ratio = _ratio_band(y, sr, (5000, 10000))
    # detect frames where 6–8kHz spikes vs neighbors
    S = np.abs(librosa.stft(to_mono(y), n_fft=2048, hop_length=512))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    sib_idx = np.where((freqs>=6000)&(freqs<=8000))[0]
    broad_idx = np.where((freqs>=2000)&(freqs<=12000))[0]
    sib_curve = S[sib_idx, :].mean(axis=0)
    broad_curve = S[broad_idx, :].mean(axis=0) + 1e-12
    spikes = np.mean((sib_curve/broad_curve) > 1.5)
    severity = float(0.5*ratio + 0.5*spikes)
    grade = "low" if severity < 0.15 else "moderate" if severity < 0.35 else "high"
    return {"severity": round(severity,3), "grade": grade}

def _masking_overlap(stems, sr):
    # Compute 8 band energies per stem, then pairwise overlap
    bands = [(20,60),(60,120),(120,250),(250,500),(500,1000),(1_000,2_000),(2_000,5_000),(5_000,10_000)]
    names = list(stems.keys())
    E = {}
    for n in names:
        y, fs = librosa.load(stems[n], sr=sr, mono=True)
        E[n] = np.array([_ratio_band(y, fs, b) for b in bands])
    overlaps = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            o = float(np.mean(np.minimum(E[names[i]], E[names[j]])))
            overlaps.append({"pair": f"{names[i]}–{names[j]}", "overlap": round(o,3)})
    overlaps.sort(key=lambda x: -x["overlap"])
    return overlaps[:5]

def _stereo_metrics(y_stereo):
    if y_stereo.ndim == 1:
        return {"width": 0.0, "corr": 1.0, "balance_db": 0.0, "note": "mono"}
    L = y_stereo[0]; R = y_stereo[1]
    mid = (L+R)/2.0; side = (L-R)/2.0
    width = float(np.sqrt(np.mean(side**2)/(np.mean(mid**2)+1e-12)))
    # Inter-channel correlation
    corr = float(np.corrcoef(L, R)[0,1])
    # Channel balance in dB
    l_rms = np.sqrt(np.mean(L**2))+1e-12
    r_rms = np.sqrt(np.mean(R**2))+1e-12
    balance_db = float(20*np.log10(l_rms/r_rms))
    note = "wide" if width>0.5 else "narrow" if width<0.2 else "moderate"
    return {"width": round(width,3), "corr": round(corr,3), "balance_db": round(balance_db,2), "note": note}

def _tempo_and_beats(y, sr):
    y_mono = to_mono(y)
    tempo, beats = librosa.beat.beat_track(y=y_mono, sr=sr, trim=False)
    # Safely coerce tempo to a scalar float even if it comes as a NumPy array
    tempo_scalar = float(np.asarray(tempo).reshape(-1)[0])
    times = librosa.frames_to_time(beats, sr=sr)
    beat_times = [float(np.round(t, 3)) for t in times[:200]]
    return {"bpm": float(np.round(tempo_scalar, 2)), "beats": beat_times}

def _key_detection(y, sr):
    y_mono = to_mono(y)
    chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
    mean_chroma = chroma.mean(axis=1)
    # simple template major/minor from librosa defaults
    maj_template = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    min_template = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    maj_scores = [np.roll(maj_template, i).dot(mean_chroma) for i in range(12)]
    min_scores = [np.roll(min_template, i).dot(mean_chroma) for i in range(12)]
    maj_best = int(np.argmax(maj_scores)); min_best = int(np.argmax(min_scores))
    maj_val = maj_scores[maj_best]; min_val = min_scores[min_best]
    pitch_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    if maj_val >= min_val:
        return {"key": f"{pitch_names[maj_best]} major", "confidence": float(round(maj_val/(maj_val+min_val+1e-9),3))}
    else:
        return {"key": f"{pitch_names[min_best]} minor", "confidence": float(round(min_val/(maj_val+min_val+1e-9),3))}

def _timbre_formant_proxy(y, sr):
    y_mono = to_mono(y)
    S = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512)) + 1e-9
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    # spectral slope estimate
    logS = np.log(S)
    slope = np.polyfit(freqs[:len(freqs)//2], logS[:len(freqs)//2,:].mean(axis=1), 1)[0]
    warmth = _ratio_band(y_mono, sr, (150, 400))
    air = _ratio_band(y_mono, sr, (10000, 14000))
    # Provide qualitative tag
    tag = "warm" if warmth>0.06 else "neutral" if air<0.02 else "airy"
    return {"warmth": round(warmth,3), "air": round(air,3), "slope": round(float(slope),6), "tag": tag}

def _clipping_check(y):
    y_mono = to_mono(y)
    hard = np.sum(np.abs(y_mono) >= 0.999)
    cf = _crest_factor(y_mono)
    clipped = hard > 0
    return {"clipped_samples": int(hard), "crest_db": float(round(cf,2)), "flag": bool(clipped)}

def _presence_air(y, sr):
    presence = _ratio_band(y, sr, (4000, 6000))
    air = _ratio_band(y, sr, (10000, 14000))
    return {"presence": round(presence,3), "air": round(air,3)}

def _silence_noise(y, sr):
    y_mono = to_mono(y)
    frame = librosa.util.frame(y_mono, frame_length=2048, hop_length=512, axis=0)
    rms = np.sqrt(np.mean(frame**2, axis=1))
    thr = np.percentile(rms, 20)  # bottom quintile as noise floor region
    noise_floor = float(20*np.log10(thr + 1e-12))
    silent_ratio = float(np.mean(rms < 0.01*np.max(rms)))
    return {"noise_floor_db": round(noise_floor,2), "silence_ratio": round(silent_ratio,3)}

def _transient_energy(y, sr):
    y_mono = to_mono(y)
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
    punch = float(np.percentile(onset_env, 95) / (np.mean(onset_env)+1e-9))
    sharpness = float(np.mean(np.diff(onset_env.clip(min=0))) )
    return {"punch_score": round(punch,3), "sharpness": round(sharpness,3)}

def _reverb_room(y, sr):
    y_mono = to_mono(y)
    S = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512))**2
    early = S[:, :S.shape[1]//3].mean()
    late = S[:, S.shape[1]//3:].mean() + 1e-12
    ldr = float(10*np.log10(late/early + 1e-12))
    grade = "dry" if ldr < -3 else "moderate" if ldr < 1 else "wet"
    return {"late_early_db": round(ldr,2), "grade": grade}

def _track_structure(y, sr):
    """
    Segment track into coarse sections using a novelty-based method that avoids
    librosa.segment.agglomerative API differences. Works across librosa versions.
    """
    y_mono = to_mono(y)
    hop = 512
    # Log-mel spectrogram
    S = librosa.feature.melspectrogram(y=y_mono, sr=sr, n_mels=96, hop_length=hop, fmax=min(16000, sr//2))
    logS = librosa.power_to_db(S, ref=np.max)

    # Build recurrence (self-similarity) matrix
    # mode='affinity' yields higher values for similar frames
    R = librosa.segment.recurrence_matrix(logS, k=7, width=3, sym=True, mode='affinity')

    # Compute a novelty curve from the affinity matrix by applying a checkerboard kernel
    # This approximates boundary strength.
    # Create a simple checkerboard kernel
    kernel_size = 16
    kb = np.add.outer(np.arange(kernel_size), -np.arange(kernel_size))
    kb = np.sign(kb)  # checkerboard-like
    # Convolve along the main diagonal band
    novelty = []
    diag = np.diag_indices_from(R)
    # Project along diagonals by summing local neighborhoods
    for i in range(R.shape[0]):
        i0 = max(0, i - kernel_size//2)
        i1 = min(R.shape[0], i + kernel_size//2)
        window = R[i0:i1, i0:i1]
        if window.size == 0:
            novelty.append(0.0)
        else:
            # Pad/crop kernel to window
            ks0 = min(window.shape[0], kernel_size)
            ks1 = min(window.shape[1], kernel_size)
            k = kb[:ks0, :ks1]
            # Normalize
            wn = window - window.mean()
            val = float(np.sum(wn[:ks0, :ks1] * k))
            novelty.append(val)
    novelty = np.array(novelty)
    # Normalize novelty
    novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-9)

    # Peak-pick boundaries
    # Aim for up to ~6 sections -> 5 internal boundaries plus start(0)
    peaks = librosa.util.peak_pick(novelty, pre_max=8, post_max=8, pre_avg=8, post_avg=8, delta=0.1, wait=16)
    # Always include start, optionally end
    frames = [0] + sorted(list(map(int, peaks)))
    # Enforce minimum spacing to avoid clutter
    min_gap = int(2.0 * sr / hop)  # ~2s in frames
    pruned = []
    last = -10**9
    for f in frames:
        if f - last >= min_gap:
            pruned.append(f)
            last = f
    # Limit number of sections to ~6 (Intro..Outro). Keep first N-1 boundaries evenly if too many.
    max_sections = 6
    if len(pruned) > max_sections:
        # Keep start and select evenly spaced boundaries
        idxs = np.linspace(1, len(pruned)-1, max_sections-1, endpoint=False).astype(int)
        pruned = [pruned[0]] + [pruned[i] for i in idxs]

    # Convert to times
    times = librosa.frames_to_time(np.array(pruned), sr=sr, hop_length=hop).tolist()
    labels = ["Intro", "Verse", "Chorus", "Bridge", "Break", "Outro"]
    segs = []
    for i, t in enumerate(times):
        name = labels[i] if i < len(labels) else f"Section {i+1}"
        segs.append({"label": name, "time": float(round(t, 2))})
    return segs[:10]


def _instrument_detection_from_stems(stem_paths, sr):
    # Use Demucs stems presence and rough tone labels
    detected = []
    for name, p in stem_paths.items():
        if not p or not os.path.exists(p):
            continue
        y, fs = librosa.load(p, sr=sr, mono=True)
        energy = float(np.sqrt(np.mean(y**2)))
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=fs).mean())
        tone = "low" if centroid < 1000 else "mid" if centroid < 3000 else "high"
        detected.append({"stem": name, "present": energy > 0.005, "rms": round(energy,4), "tone": tone})
    return detected

def run_engineering_analysis(full_mix_path, saved_stems):
    # Load full mix stereo for width analyses
    y_st, sr = torchaudio.load(full_mix_path)
    y_np = y_st.numpy()
    # Convert to numpy [C, T]
    if y_np.shape[0] == 1:
        y_np = np.vstack([y_np, y_np])
    y_mono = librosa.to_mono(y_np)

    # 1 Genre classification is already computed separately; we won't duplicate here.

    # 2 Instrument detection via stems
    instruments = _instrument_detection_from_stems(saved_stems, sr)

    # 3 Loudness
    lufs = _meter_lufs(y_mono, sr)
    rms, peak = _rms_peak(y_mono)

    # 4 Spectral features
    spectral = _spectral_features(y_mono, sr)

    # 5 Dynamic range / crest factor
    crest = _crest_factor(y_mono)

    # 6 Sibilance
    sibilance = _sibilance(y_mono, sr)

    # 7 Masking between stems
    masking = _masking_overlap(saved_stems, sr) if saved_stems else []

    # 8 Stereo width
    stereo = _stereo_metrics(y_np)

    # 9 Tempo & beat
    tempo_beats = _tempo_and_beats(y_mono, sr)

    # 10 Key detection
    key = _key_detection(y_mono, sr)

    # 11 Formant/Timbre proxy
    timbre = _timbre_formant_proxy(y_mono, sr)

    # 12 Clipping/Distortion
    clip = _clipping_check(y_mono)

    # 13 Presence/Air
    pa = _presence_air(y_mono, sr)

    # 14 Silence & Noise floor
    sn = _silence_noise(y_mono, sr)

    # 15 Transient Energy
    trans = _transient_energy(y_mono, sr)

    # 16 Reverb/Room
    reverb = _reverb_room(y_mono, sr)

    # 17 Track structure
    structure = _track_structure(y_mono, sr)

    return {
        "instrument_detection": instruments,
        "loudness": {"lufs": round(float(lufs),2), "rms": round(float(rms),4), "peak": round(float(peak),4)},
        "spectral": spectral,
        "dynamic_range": {"crest_db": round(float(crest),2)},
        "sibilance": sibilance,
        "masking": masking,
        "stereo": stereo,
        "tempo": tempo_beats,
        "key": key,
        "timbre": timbre,
        "clipping": clip,
        "presence_air": pa,
        "silence_noise": sn,
        "transient": trans,
        "reverb": reverb,
        "structure": structure
    }


# ============== FLASK APP ==============

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", result=None, show_only_curated=False)

    if "audio" not in request.files:
        return redirect(url_for("index"))
    f = request.files["audio"]
    if not f or f.filename == "":
        return redirect(url_for("index"))

    show_only_curated = request.form.get("show_curated") == "1"

    up_path = bytes_to_tempfile(f)

    try:
        # 1) Separate stems (only persisted outputs)
        stems, sr = separate_with_demucs(str(up_path), DEVICE)
        saved_stems = {}
        if stems:
            for lab, arr in stems.items():
                saved = save_audio_stem(lab.capitalize(), arr, sr, str(up_path))
                saved_stems[lab] = saved

        # 2) Genre (AST + MERT + Fusion)
        ast_proba = ast.predict_proba_over(str(up_path), GENRE_LABELS)
        ast_top = max(ast_proba.items(), key=lambda x: x[1])[0] if ast_proba else "Unknown"
        mert_top, mert_proba = mert_genre.predict_top(str(up_path))
        genre_fused, fused_items = fuse_genre_probs(ast_proba, mert_proba, GENRE_LABELS, w_ast=0.4, w_mert=0.6)

        # 3) Emotions (Music2Emotion ensemble)
        emo = m2e_ensemble.predict_full(str(up_path))
        mood_tags = emo.get("mood_tags", [])
        val = emo.get("valence")
        aro = emo.get("arousal")

        # 4) Curated analysis
        curated = curate_analysis(
            filename=Path(f.filename).name,
            genre=genre_fused,
            mood_tags=mood_tags,
            val=val,
            aro=aro,
            saved_stems=saved_stems,
            ast_top=ast_top,
            mert_top=mert_top,
            fused_probs=fused_items
        )

        # 5) Engineering analysis (17 items ordered in UI)
        eng = run_engineering_analysis(str(up_path), saved_stems)

        # 6) Save JSON snapshot
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{Path(f.filename).stem}_{uuid.uuid4().hex[:8]}_{ts}"
        json_path = RESULTS_DIR / f"{base}_analysis.json"
        result_payload = {
            "file": Path(f.filename).name,
            "genre": {
                "fused_top": genre_fused,
                "ast_top": ast_top,
                "mert_top": mert_top,
                "fused_probs": fused_items
            },
            "mood_tags": mood_tags,
            "valence": val,
            "arousal": aro,
            "stems": saved_stems,
            "curated_analysis": curated,
            "engineering_analysis": eng
        }
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(result_payload, jf, indent=2)

        # 7) Render
        return render_template(
            "index.html",
            result={
                "filename": Path(f.filename).name,
                "genre": result_payload["genre"],
                "mood_tags": mood_tags[:10],
                "valence": val,
                "arousal": aro,
                "json_name": json_path.name,
                "curated": curated,
                "engineering": eng
            },
            show_only_curated=show_only_curated
        )
    finally:
        try:
            os.unlink(up_path)
        except Exception:
            pass


@app.route("/results/<path:name>")
def results_file(name):
    path = RESULTS_DIR / name
    if path.exists():
        return send_from_directory(RESULTS_DIR, name)
    abort(404)


if __name__ == "__main__":
    log(f"Device: {DEVICE} | AMP: {AMP}")
    app.run(host="0.0.0.0", port=5000, debug=False)
