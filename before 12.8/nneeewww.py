# new4.py
# 🎵 Music Analyzer — Ensemble over all Music2Emotion checkpoints
# Demucs (stems→instruments) + AST (genre) + Music2Emotion (all .ckpt under saved_models)
# - Status strip shows per-checkpoint progress: “Running i/N: <ckpt>”
# - Saves both raw (all tags) and curated (optional) in JSON

import os, sys, threading, json, datetime, shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from demucs.pretrained import get_model as get_demucs_model
from demucs.apply import apply_model
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ================== PATHS & PRESETS ==================
MUSIC2EMO_REPO_ROOT = r"C:\Users\disng\Documents\music-analyzer\Music2Emotion"
SAVED_MODELS_DIR    = os.path.join(MUSIC2EMO_REPO_ROOT, "saved_models")

# Curated display list (optional filter in UI)
CURATED_MOODS = [
    "sad", "happy", "angry", "calm", "energetic",
    "melancholic", "hopeful", "romantic"
]

# ================== APP CONFIG ==================
CONFIDENCE_THRESHOLD   = 0.15  # AST confidence cut for labels
DEFAULT_MOOD_THRESHOLD = 0.50  # UI filter default for mood tags display (not model logic)
TOPK                   = 10    # top K mood tags to show in UI

RESULTS_DIR = "audio_analysis_results"
STEMS_DIR   = os.path.join(RESULTS_DIR, "stems")

USE_GPU = True
DEVICE  = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
AMP     = (DEVICE == "cuda")
DEBUG   = True
def log(*a):
    if DEBUG: print("[DEBUG]", *a)

# ================== Music2Emotion import ==================
MUSIC2EMO_AVAILABLE = False
if os.path.isdir(MUSIC2EMO_REPO_ROOT) and MUSIC2EMO_REPO_ROOT not in sys.path:
    sys.path.insert(0, MUSIC2EMO_REPO_ROOT)
try:
    import music2emo as _m2e_mod  # must be your edited music2emo.py that returns mood_probs, valence, arousal
    Music2emo = getattr(_m2e_mod, "Music2emo")
    MUSIC2EMO_AVAILABLE = True
    log("Music2Emotion import: OK")
except Exception as e:
    log("Music2Emotion import FAILED:", e)

# ================== HELPERS ==================
def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEMS_DIR, exist_ok=True)

def save_audio_stem(inst_label, stem_audio, sr, audio_path):
    out_dir = os.path.join(STEMS_DIR, inst_label)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(out_dir, f"{base}_{inst_label}.wav")
    if stem_audio.ndim == 2:  # Demucs outputs [C, T]
        stem_audio = stem_audio.T
    sf.write(out_path, stem_audio, sr)
    return os.path.abspath(out_path)

def to_16k_mono_wav(src_path):
    y, _ = librosa.load(src_path, sr=16000, mono=True)
    tmp_dir = os.path.join(RESULTS_DIR, "tmp"); os.makedirs(tmp_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.abspath(os.path.join(tmp_dir, f"{base}_16kmono.wav"))
    sf.write(out_path, y, 16000)
    return out_path

def _with_load_mapped_to_device(fn):
    import torch as _torch
    orig = _torch.load
    def _load_dev(*args, **kw):
        if "map_location" not in kw:
            kw["map_location"] = torch.device(DEVICE)
        return orig(*args, **kw)
    _torch.load = _load_dev
    try:
        return fn()
    finally:
        _torch.load = orig

# ================== DEMUCS ==================
def separate_with_demucs(audio_path, device=DEVICE):
    try:
        model = get_demucs_model('htdemucs').to(device).eval()
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        if sr != 44100:
            wav = torchaudio.transforms.Resample(sr, 44100)(wav); sr = 44100
        wav = wav.to(device)

        with torch.inference_mode():
            if AMP:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    sources = apply_model(model, wav.unsqueeze(0))[0]
            else:
                sources = apply_model(model, wav.unsqueeze(0))[0]
        names = ['drums', 'bass', 'other', 'vocals']
        out = {n: sources[i].detach().to("cpu").numpy() for i, n in enumerate(names)}
        log("Demucs separation OK:", list(out.keys()))
        return out, sr
    except Exception as e:
        log("Demucs separation FAILED:", e)
        return {}, None

# ================== AST (AudioSet) ==================
# ================== AST (AudioSet) ==================
class ASTClassifier:
    GENRE_WHITELIST = {
        "Pop music","Rock music","Hip hop music","Jazz","Classical music",
        "Electronic music","Metal","Blues","Reggae","Country",
        "Funk","Soul music","Dance music","Techno","House music",
        "Trance music","Disco","R&B","Gospel music","Folk music","Latin music",
        "Indie rock","Alternative rock","Punk rock","Grunge","Ambient music"
    }
    GENERIC_BADGE = {"Music", "Sound", "Silence", "Noise", "Speech"}

    def __init__(self):
        self.fe = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = AutoModelForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        ).to(DEVICE).eval()
        self.id2label = self.model.config.id2label
        log("AST model loaded on", DEVICE)

    def _chunks(self, x, sr, seconds=10.0):
        win = int(seconds * sr)
        if len(x) <= win:
            yield x
            return
        for s in range(0, len(x), win):
            e = min(s + win, len(x))
            yield x[s:e]

    def predict_top_genre(self, path):
        """Average logits over 10s chunks, then pick the best label from a genre whitelist."""
        try:
            audio, sr = librosa.load(path, sr=16000, mono=True)
            logits_sum, n = None, 0
            with torch.inference_mode():
                for ch in self._chunks(audio, 16000, 10.0):
                    inputs = self.fe(ch, sampling_rate=16000, return_tensors="pt", padding=True)
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    if AMP:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = self.model(**inputs).logits
                    else:
                        logits = self.model(**inputs).logits
                    v = logits[0].detach().to("cpu").numpy()
                    logits_sum = v if logits_sum is None else (logits_sum + v)
                    n += 1

            if not n:
                return "Unknown"

            avg = logits_sum / n
            probs = np.exp(avg - np.max(avg))
            probs = probs / probs.sum()
            order = np.argsort(-probs)

            # 1) prefer a real genre from whitelist
            for i in order:
                lab = self.id2label[int(i)]
                if lab in self.GENRE_WHITELIST:
                    return lab

            # 2) otherwise pick the top non-generic class
            for i in order:
                lab = self.id2label[int(i)]
                if lab not in self.GENERIC_BADGE:
                    return lab

            # 3) absolute fallback = top-1
            return self.id2label[int(order[0])]
        except Exception as e:
            log("AST genre FAILED:", e)
            return "Unknown"

    # (kept for instruments code that uses thresholded multi-labels)
    def predict(self, path, threshold=CONFIDENCE_THRESHOLD):
        try:
            audio, _ = librosa.load(path, sr=16000, mono=True)
            inputs = self.fe(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.inference_mode():
                if AMP:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(**inputs).logits
                else:
                    logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0].detach().to("cpu").numpy()
            idxs = np.where(probs >= threshold)[0]
            # sort by confidence (desc)
            idxs = idxs[np.argsort(-probs[idxs])]
            return [(self.id2label[int(i)], float(probs[int(i)])) for i in idxs]
        except Exception as e:
            log("AST predict FAILED:", e)
            return []


# ================== Music2Emotion runners & ensemble ==================
class M2ERunnerSingle:
    """Single ckpt runner; uses safe temp folders via your edited music2emo.py."""
    def __init__(self, ckpt_path):
        if not MUSIC2EMO_AVAILABLE:
            raise RuntimeError("Music2Emotion not importable.")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        self.repo_root = MUSIC2EMO_REPO_ROOT
        self.ckpt = ckpt_path
        self.name = os.path.basename(ckpt_path)

        def _init():
            return Music2emo(model_weights=ckpt_path)
        self.model = _with_load_mapped_to_device(_init)
        log("M2E ready on", DEVICE, ":", ckpt_path)

    def predict_full(self, audio_path, wav16_cache=None):
        # music2emo.py handles its own temp_out and returns dict with:
        # {"valence": float, "arousal": float, "predicted_moods": [...], "mood_probs": {...}}
        # We pass 16k mono WAV path if precomputed
        if wav16_cache is None:
            wav16_cache = to_16k_mono_wav(audio_path)
        cwd = os.getcwd()
        try:
            os.chdir(self.repo_root)
            out = self.model.predict(wav16_cache)  # your edited code reads and computes features
            if isinstance(out, dict):
                # ensure shape
                out.setdefault("mood_probs", {})
                return out
            return {"mood_probs": {}, "valence": None, "arousal": None, "predicted_moods": []}
        except Exception as e:
            log("Music2Emotion predict FAILED:", e)
            return {"mood_probs": {}, "valence": None, "arousal": None, "predicted_moods": []}
        finally:
            os.chdir(cwd)

def _infer_dataset_from_name(basename: str):
    b = basename.lower()
    if "jamendo" in b or b.startswith("j_"): return "jamendo"
    if "deam" in b or b.startswith("d_"):    return "deam"
    if "emomusic" in b or b.startswith("e_"):return "emomusic"
    if "pmemo" in b or b.startswith("p_"):   return "pmemo"
    return "unknown"

class M2EAllCkptEnsemble:
    """
    Loads ALL .ckpt under saved_models and averages:
      - mood_probs: mean across ckpts (by tag)
      - valence/arousal: mean across ckpts that output them
    """
    def __init__(self, ckpt_dir=SAVED_MODELS_DIR):
        if not os.path.isdir(ckpt_dir):
            raise RuntimeError(f"saved_models dir not found: {ckpt_dir}")
        files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        files.sort()
        if not files:
            raise RuntimeError(f"No .ckpt files found in {ckpt_dir}")
        self.runners = []
        for p in files:
            try:
                self.runners.append(M2ERunnerSingle(p))
            except Exception as e:
                log("Skip ckpt:", p, "| reason:", e)
        if not self.runners:
            raise RuntimeError("No valid Music2Emotion checkpoints could be loaded.")

    def predict_full(self, audio_path, status_cb=None):
        wav16 = to_16k_mono_wav(audio_path)
        all_mood_probs = {}  # tag -> list[prob]
        val_list, aro_list = [], []
        total = len(self.runners)

        for i, r in enumerate(self.runners, 1):
            if status_cb:
                status_cb(i, total, r.name)
            out = r.predict_full(audio_path, wav16_cache=wav16)
            mp = out.get("mood_probs", {})
            for lab, p in mp.items():
                all_mood_probs.setdefault(lab, []).append(float(p))
            v = out.get("valence"); a = out.get("arousal")
            if v is not None: val_list.append(float(v))
            if a is not None: aro_list.append(float(a))

        # average probs per tag
        merged = [(lab, float(np.mean(vals))) for lab, vals in all_mood_probs.items()]
        merged.sort(key=lambda x: x[1], reverse=True)

        val = float(np.mean(val_list)) if val_list else None
        aro = float(np.mean(aro_list)) if aro_list else None

        primary = merged[0][0].capitalize() if merged else "Unknown"
        # Also keep a label list in descending order (like predicted_moods)
        predicted = [lab for lab, p in merged if p >= 0.0]  # no threshold here; UI slider will filter view
        return {
            "mood_tags": merged,             # list of (label, prob)
            "predicted_moods": predicted,    # labels only, sorted
            "valence": val,
            "arousal": aro,
        }

# ================== instruments (AST-driven, canonical mapping) ==================
INSTR_ALIAS = {
    "Vocals": ["vocals","singing","male singing","female singing","choir","choral"],
    "Bass":   ["bass","bass guitar","double bass"],
    "Drums":  ["drums","drum kit","snare drum","bass drum","cymbal","hi-hat","tom-tom"],
    "Guitar": ["guitar","electric guitar","acoustic guitar"],
    "Piano":  ["piano"], "Violin": ["violin"], "Cello": ["cello"],
    "Trumpet":["trumpet"], "Synthesizer": ["synthesizer","synth"],
    "Electronic": ["electronic","electronica"], "Clapping": ["clapping"],
}
def _canon_instr(lbl: str):
    l = lbl.lower()
    for canon, keys in INSTR_ALIAS.items():
        if any(k in l for k in keys):
            return canon
    return None

# ================== PIPELINE ==================
class UnifiedAudioPipeline:
    def __init__(self):
        self.ast = ASTClassifier()
        if not MUSIC2EMO_AVAILABLE:
            raise RuntimeError("music2emo.py not importable.")
        self.m2e_ensemble = M2EAllCkptEnsemble(SAVED_MODELS_DIR)
        log("device:", DEVICE, "| AMP:", AMP)

    def _detect_genre(self, path):
        return self.ast.predict_top_genre(path)


    def _instruments(self, saved_paths, mix_path=None):
        inst = set()
        paths = list(saved_paths.values())
        if mix_path: paths.append(mix_path)
        for sf_path in paths:
            try:
                preds = self.ast.predict(sf_path, threshold=0.10)
                for lbl, _ in preds:
                    canon = _canon_instr(lbl)
                    if canon: inst.add(canon)
            except Exception as e:
                log("Instrument detect error:", e)
        return sorted(inst)

    def analyze_file(self, audio_path, apply_curated=False, progress_cb=None, status_cb=None):
        ensure_dirs()
        results = {}

        if progress_cb: progress_cb(0.10, "Separating stems…")
        stems, sr = separate_with_demucs(audio_path, DEVICE)
        if not stems:
            results["Error"] = "Separation failed"
            return results

        if progress_cb: progress_cb(0.30, "Saving stems…")
        saved_paths = {lab: save_audio_stem(lab.capitalize(), arr, sr, audio_path)
                       for lab, arr in stems.items()}
        results["Saved_Stems"] = saved_paths

        if progress_cb: progress_cb(0.40, "Detecting genre…")
        results["Genre"] = self._detect_genre(audio_path)

        # Emotions — ensemble across ALL ckpts
        if progress_cb: progress_cb(0.70, "Predicting emotions (all checkpoints)…")

        def _ckpt_status(i, total, name):
            if progress_cb:
                frac = 0.70 + 0.15 * (i / max(total, 1))
                progress_cb(frac, f"Checkpoint {i}/{total}: {name}")
            if status_cb:
                status_cb(f"Running {i}/{total}: {name}")

        emo_raw = self.m2e_ensemble.predict_full(audio_path, status_cb=_ckpt_status)

        # RAW (always written)
        raw_emo = {
            "mood_tags": emo_raw.get("mood_tags", []),
            "valence":   emo_raw.get("valence"),
            "arousal":   emo_raw.get("arousal"),
        }
        raw_emo["primary"] = raw_emo["mood_tags"][0][0].capitalize() if raw_emo["mood_tags"] else "Unknown"

        # DISPLAY (maybe curated)
        disp_emo = dict(raw_emo)
        if apply_curated and disp_emo["mood_tags"]:
            keep = set([m.lower() for m in CURATED_MOODS])
            disp_emo["mood_tags"] = [(lab, p) for (lab, p) in disp_emo["mood_tags"] if lab.lower() in keep]
            disp_emo["primary"] = disp_emo["mood_tags"][0][0].capitalize() if disp_emo["mood_tags"] else raw_emo["primary"]

        results["EmotionFullRaw"] = raw_emo
        results["EmotionFull"]    = disp_emo
        results["Emotion"]        = disp_emo["primary"]
        results["curated_applied"]= bool(apply_curated)
        if apply_curated:
            results["curated_list"] = CURATED_MOODS

        if progress_cb: progress_cb(0.88, "Extracting instruments…")
        results["Instruments"] = self._instruments(saved_paths, mix_path=os.path.abspath(audio_path))

        if progress_cb: progress_cb(0.96, "Writing JSON…")
        base = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(RESULTS_DIR, f"{base}_analysis.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        results["JSON"] = json_path

        if progress_cb: progress_cb(1.0, "Done.")
        if status_cb:   status_cb("Idle.")
        return results

# ================== UI ==================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎵 Music Analyzer (Dark)")
        self.geometry("1220x820")
        self.configure(bg="#121212")
        self.columnconfigure(0, weight=1)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=8, background="#1DB954", foreground="black", borderwidth=0)
        style.map("TButton", background=[("active", "#18a84a")])
        style.configure("TScale", troughcolor="#2A2A2A", background="#121212")
        style.configure("TProgressbar", background="#F97316")

        # Top bar
        top = ttk.Frame(self); top.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        top.columnconfigure(0, weight=1)
        ttk.Label(top, text="Upload Audio File", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")
        self.btn_choose = ttk.Button(top, text="📂 Choose Files", command=self.choose_files)
        self.btn_choose.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.lbl_files = ttk.Label(top, text="No file selected")
        self.lbl_files.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        # Threshold + curated filter
        row = ttk.Frame(self); row.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text="Mood Detection Threshold", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.thr_value = tk.DoubleVar(value=DEFAULT_MOOD_THRESHOLD)
        self.scale = ttk.Scale(row, from_=0, to=1, orient="horizontal", variable=self.thr_value)
        self.scale.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.lbl_thr = ttk.Label(row, text=f"{self.thr_value.get():.2f}")
        self.lbl_thr.grid(row=1, column=2, sticky="e", padx=(8,0))
        self.thr_value.trace_add("write", lambda *_: self.lbl_thr.config(text=f"{self.thr_value.get():.2f}"))

        self.curated_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Only show curated mood list", variable=self.curated_on).grid(row=2, column=0, sticky="w", pady=(6,0))

        # Actions
        action = ttk.Frame(self); action.grid(row=2, column=0, sticky="ew", padx=12)
        action.columnconfigure(0, weight=1)
        self.btn_analyze = ttk.Button(action, text="🎧 Analyze Emotions (Ensemble)", command=self.analyze_clicked)
        self.btn_analyze.grid(row=0, column=0, sticky="w")
        self.pbar = ttk.Progressbar(action, mode="determinate", length=260)
        self.pbar.grid(row=0, column=1, sticky="e")

        # Middle: logs & results
        mid = ttk.Frame(self); mid.grid(row=3, column=0, sticky="nsew", padx=12, pady=(8, 8))
        self.rowconfigure(3, weight=1)
        mid.columnconfigure(0, weight=3)
        mid.columnconfigure(1, weight=2)
        mid.rowconfigure(0, weight=1)

        log_frame = ttk.Frame(mid); log_frame.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        ttk.Label(log_frame, text="Log", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.txt_log = tk.Text(log_frame, height=12, bg="#121212", fg="#9ae6b4", insertbackground="white",
                               relief="flat", font=("Consolas", 9))
        self.txt_log.pack(fill="both", expand=True, pady=(4,0))

        res_frame = ttk.Frame(mid); res_frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(res_frame, text="Analysis Results", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.txt_res = tk.Text(res_frame, height=12, bg="#121212", fg="white", insertbackground="white",
                               relief="flat", font=("Consolas", 10))
        self.txt_res.pack(fill="both", expand=True, pady=(4,0))

        # Bottom plots
        bot = ttk.Frame(self); bot.grid(row=4, column=0, sticky="nsew", padx=12, pady=(0,12))
        self.rowconfigure(4, weight=1)
        bot.columnconfigure(0, weight=3)
        bot.columnconfigure(1, weight=2)

        left_plot = ttk.Frame(bot); left_plot.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        ttk.Label(left_plot, text="Mood Probabilities", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.tk_bar = None
        self.left_plot_host = ttk.Frame(left_plot); self.left_plot_host.pack(fill="both", expand=True)

        right_plot = ttk.Frame(bot); right_plot.grid(row=0, column=1, sticky="nsew")
        ttk.Label(right_plot, text="Valence–Arousal Space", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.tk_va = None
        self.right_plot_host = ttk.Frame(right_plot); self.right_plot_host.pack(fill="both", expand=True)

        # Status strip
        status = ttk.Frame(self); status.grid(row=5, column=0, sticky="ew", padx=12, pady=(0,10))
        self.status_var = tk.StringVar(value="Ready.")
        self.lbl_status = ttk.Label(status, textvariable=self.status_var, anchor="w")
        self.lbl_status.pack(fill="x")

        # Pipeline + state
        try:
            self.pipeline = UnifiedAudioPipeline()
        except Exception as e:
            messagebox.showerror("Model error", f"Could not initialize Music2Emotion.\n{e}")
            raise
        self.files = []
        self.log_line(f"Device: {DEVICE} | AMP: {AMP}")

    # UI helpers
    def log_line(self, s): self.txt_log.insert("end", s + "\n"); self.txt_log.see("end")
    def set_status(self, text: str):
        self.status_var.set(text)
        self.lbl_status.update_idletasks()

    def choose_files(self):
        fs = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.wav *.mp3")])
        if not fs: return
        self.files = list(fs)
        self.lbl_files.config(text=f"{len(self.files)} file(s) selected")

    def analyze_clicked(self):
        if not self.files:
            messagebox.showinfo("No files", "Please choose at least one audio file.")
            return
        thr = float(self.thr_value.get())
        curated = bool(self.curated_on.get())
        self.btn_analyze.state(["disabled"]); self.pbar["value"] = 0
        threading.Thread(target=self._process_files, args=(self.files, thr, curated), daemon=True).start()

    def _process_files(self, files, thr, curated):
        for f in files:
            self.log_line(f"Processing: {f}")
            def progress_cb(frac, msg):
                self.pbar["value"] = max(0, min(100, int(frac*100)))
                self.pbar.update_idletasks()
                if msg: self.log_line(msg)
            try:
                results = self.pipeline.analyze_file(
                    f, apply_curated=curated,
                    progress_cb=progress_cb,
                    status_cb=self.set_status
                )
                self.render_results(f, results, thr)
            except Exception as e:
                self.log_line(f"Error: {e}")
        self.set_status("Idle.")
        self.btn_analyze.state(["!disabled"])
        messagebox.showinfo("Done", f"Processed {len(files)} file(s).")

    # plotting
    def draw_bar(self, items):
        if self.tk_bar: self.tk_bar.get_tk_widget().destroy()
        fig = Figure(figsize=(8, 3), dpi=100); ax = fig.add_subplot(111)
        labels = [a for a,_ in items][:TOPK]; vals = [b for _,b in items][:TOPK]
        y = np.arange(len(labels)); ax.barh(y, vals)
        ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
        ax.set_xlabel("Probability"); ax.set_title("Top Predicted Mood Tags"); fig.tight_layout()
        self.tk_bar = FigureCanvasTkAgg(fig, master=self.left_plot_host); self.tk_bar.draw()
        self.tk_bar.get_tk_widget().pack(fill="both", expand=True)

    def draw_va(self, valence, arousal):
        if self.tk_va: self.tk_va.get_tk_widget().destroy()
        fig = Figure(figsize=(3.2, 2.8), dpi=100); ax = fig.add_subplot(111)
        ax.set_xlim(0, 9); ax.set_ylim(0, 9)
        ax.set_xlabel("Valence (Positivity)"); ax.set_ylabel("Arousal (Intensity)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        if (valence is not None) and (arousal is not None): ax.scatter([valence], [arousal], s=40)
        fig.tight_layout()
        self.tk_va = FigureCanvasTkAgg(fig, master=self.right_plot_host); self.tk_va.draw()
        self.tk_va.get_tk_widget().pack(fill="both", expand=True)

    # render
    def render_results(self, path, results, thr):
        emo = results.get("EmotionFull", {"mood_tags": [], "valence": None, "arousal": None, "primary": "Unknown"})
        tags_all = sorted(emo.get("mood_tags", []), key=lambda x: x[1], reverse=True)
        tags_thr = [(l,p) for (l,p) in tags_all if p >= thr]
        val, aro = emo.get("valence"), emo.get("arousal")

        self.txt_res.delete("1.0", "end")
        disp = tags_thr if tags_thr else tags_all
        s = ", ".join([f"{lab} ({prob:.2f})" for lab, prob in disp[:TOPK]]) or "(no tags)"
        self.txt_res.insert("end", f"🎭 Predicted Mood Tags: {s}\n\n")
        self.txt_res.insert("end", f"💖 Valence: {val:.2f} (1–9)\n" if val is not None else "💖 Valence: n/a\n")
        self.txt_res.insert("end", f"⚡ Arousal: {aro:.2f} (1–9)\n" if aro is not None else "⚡ Arousal: n/a\n")
        self.txt_res.insert("end", f"\n🎼 Genre (AST): {results.get('Genre')}\n")
        self.txt_res.insert("end", "🥁 Instruments: " + (", ".join(results.get("Instruments", [])) or "(none)") + "\n")
        self.txt_res.insert("end", f"📝 JSON: {results.get('JSON')}\n")

        self.draw_bar(tags_all)
        self.draw_va(val, aro)

# ---- bootstrap
if __name__ == "__main__":
    app = App()
    app.mainloop()
