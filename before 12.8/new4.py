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

# ================== PATHS & MODELS ==================
MUSIC2EMO_REPO_ROOT = r"C:\Users\disng\OneDrive\Documents\music-analyzer\Music2Emotion"

# Candidate checkpoints (we'll load all that exist)
CANDIDATE_CKPTS = [
    os.path.join(MUSIC2EMO_REPO_ROOT, "saved_models", "D_all.ckpt"),
    os.path.join(MUSIC2EMO_REPO_ROOT, "saved_models", "E_all.ckpt"),
    os.path.join(MUSIC2EMO_REPO_ROOT, "saved_models", "J_all.ckpt"),
    os.path.join(MUSIC2EMO_REPO_ROOT, "saved_models", "P_all.ckpt"),
    os.path.join(MUSIC2EMO_REPO_ROOT, "saved_models", "deam_best.ckpt"),
    os.path.join(MUSIC2EMO_REPO_ROOT, "saved_models", "emomusic_best.ckpt"),
    os.path.join(MUSIC2EMO_REPO_ROOT, "saved_models", "jamendo_best.ckpt"),
    os.path.join(MUSIC2EMO_REPO_ROOT, "saved_models", "pmemo_best.ckpt"),
]

# ================== APP CONFIG ==================
CONFIDENCE_THRESHOLD   = 0.15    # PANN per-label display threshold (for instruments)
DEFAULT_MOOD_THRESHOLD = 0.50    # UI slider to hide tiny mood scores
TOPK                   = 10

RESULTS_DIR = "audio_analysis_results"
STEMS_DIR   = os.path.join(RESULTS_DIR, "stems")

USE_GPU = True
DEVICE  = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
AMP     = (DEVICE == "cuda")
DEBUG   = True
def log(*a):
    if DEBUG: print("[DEBUG]", *a)

class M2EWrapperEnsemble:
    def __init__(self, ckpt_paths):
        if not MUSIC2EMO_AVAILABLE:
            raise RuntimeError("Music2Emotion not importable.")
        if isinstance(ckpt_paths, (str, os.PathLike)):
            ckpt_paths = [ckpt_paths]
        # unique & existing
        uniq = []
        for p in ckpt_paths:
            if p and os.path.isfile(p) and p not in uniq:
                uniq.append(p)

        self.repo_root = MUSIC2EMO_REPO_ROOT
        self.safe_root = os.path.abspath(os.getenv("M2E_SAFE_ROOT", r"C:\m2e_work"))
        os.makedirs(self.safe_root, exist_ok=True)
        self.base_work = os.path.join(self.safe_root, "temp_out")
        self.models = []
        for p in uniq:
            try:
                def _init():
                    return Music2emo(model_weights=p)
                model = _with_load_mapped_to_device(_init)
                self.models.append((os.path.basename(p), model))
                log("M2E ready on", DEVICE, ":", p)
            except Exception as e:
                log("Skip M2E ckpt:", p, "| reason:", e)
        if not self.models:
            raise RuntimeError("No valid Music2Emotion checkpoints could be loaded.")

    def _prepare_clean_workdir(self):
        try:
            shutil.rmtree(self.base_work, ignore_errors=True)
        except Exception as e:
            log("WARN temp_out remove:", e)
        os.makedirs(self.base_work, exist_ok=True)
        for sub in ("mert", "features", "embeddings"):
            os.makedirs(os.path.join(self.base_work, sub), exist_ok=True)
        run_dir = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.workdir = os.path.join(self.base_work, run_dir)
        for sub in ("mert", "features", "embeddings"):
            os.makedirs(os.path.join(self.workdir, sub), exist_ok=True)
        os.environ["M2E_TEMP_OUT"] = self.workdir
        os.environ["TEMP_OUT"] = self.workdir

    def _normalize(self, out):
        tags = []
        valence = out.get("valence") if isinstance(out, dict) else None
        arousal = out.get("arousal") if isinstance(out, dict) else None

        probs_dict = None
        if isinstance(out, dict):
            for k in ("mood_probs", "probs", "mood_probabilities", "scores"):
                if k in out and isinstance(out[k], dict):
                    probs_dict = out[k]; break
        if probs_dict:
            tags = sorted([(k, float(v)) for k, v in probs_dict.items()],
                          key=lambda x: x[1], reverse=True)

        if not tags and isinstance(out, dict):
            labs = out.get("predicted_moods") or out.get("moods")
            scs  = out.get("predicted_scores") or out.get("scores_list")
            if isinstance(labs, (list,tuple)) and isinstance(scs, (list,tuple)) and len(labs)==len(scs):
                tags = sorted([(str(labs[i]), float(scs[i])) for i in range(len(labs))],
                              key=lambda x: x[1], reverse=True)

        if not tags and isinstance(out, dict):
            labs = out.get("predicted_moods") or out.get("moods")
            if isinstance(labs, (list, tuple)) and len(labs) > 0:
                n = len(labs)
                weights = np.linspace(0.95, 0.55, n).tolist()
                tags = [(str(labs[i]), float(weights[i])) for i in range(n)]

        primary = (tags[0][0].capitalize() if tags else "Unknown")
        return {"mood_tags": tags, "valence": valence, "arousal": arousal, "primary": primary}

    def predict_full(self, audio_path, status_cb=None):
        wav16 = to_16k_mono_wav(audio_path)
        outs = []
        cwd = os.getcwd()
        try:
            n = len(self.models)
            for i, (name, model) in enumerate(self.models, start=1):
                if status_cb:
                    status_cb(f"Running checkpoints… {i}/{n} • {name}")
                self._prepare_clean_workdir()
                os.chdir(self.repo_root)
                try:
                    out = model.predict(wav16)
                except Exception as e:
                    log("Music2Emotion predict FAILED:", e)
                    out = {}
                outs.append(self._normalize(out if isinstance(out, dict) else {}))

            # Merge the outputs of all models in the ensemble
            return outs[0] if len(outs) == 1 else _merge_mood_outputs(outs)
        finally:
            os.chdir(cwd)

# ================== Music2Emotion import ==================
MUSIC2EMO_AVAILABLE = False
if os.path.isdir(MUSIC2EMO_REPO_ROOT) and MUSIC2EMO_REPO_ROOT not in sys.path:
    sys.path.insert(0, MUSIC2EMO_REPO_ROOT)
try:
    import music2emo as _m2e_mod
    Music2emo = getattr(_m2e_mod, "Music2emo")
    MUSIC2EMO_AVAILABLE = True
    log("Music2Emotion import: OK")
except Exception as e:
    log("Music2Emotion import FAILED:", e)

# ================== HELPERS ==================
from collections import defaultdict

def _merge_mood_outputs(outputs):
    acc = defaultdict(list)
    vals, aros = [], []
    for out in outputs:
        for lab, p in out.get("mood_tags", []):
            acc[lab.lower()].append(float(p))
        if out.get("valence") is not None: vals.append(float(out["valence"]))
        if out.get("arousal") is not None: aros.append(float(out["arousal"]))
    tags = [(lab, float(np.mean(ps))) for lab, ps in acc.items()]
    tags.sort(key=lambda x: x[1], reverse=True)
    tags = [(lab.capitalize(), p) for lab, p in tags]
    val = float(np.mean(vals)) if vals else None
    aro = float(np.mean(aros)) if aros else None
    primary = tags[0][0] if tags else "Unknown"
    return {"mood_tags": tags, "valence": val, "arousal": aro, "primary": primary}

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEMS_DIR, exist_ok=True)

def save_audio_stem(inst_label, stem_audio, sr, audio_path):
    out_dir = os.path.join(STEMS_DIR, inst_label)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(out_dir, f"{base}_{inst_label}.wav")
    if stem_audio.ndim == 2:
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

def _rescale_1_9_if_raw(x):
    """If x looks like [-1,1], return scaled 1..9; else return x."""
    if x is None: return None
    if -1.2 <= x <= 1.2:
        return 1.0 + (x + 1.0) * 4.0
    return float(x)

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

# ================== GenreClassifier (AST) ==================
class GenreClassifier:
    def __init__(self):
        self.fe = AutoFeatureExtractor.from_pretrained("xpariz10/ast-finetuned-audioset-10-10-0.4593-finetuning-ESC-50")
        self.model = AutoModelForAudioClassification.from_pretrained("xpariz10/ast-finetuned-audioset-10-10-0.4593-finetuning-ESC-50").to(DEVICE).eval()
        self.id2label = self.model.config.id2label
        log("AST model loaded on", DEVICE)

    def predict_genre(self, path):
        try:
            audio, sr = librosa.load(path, sr=16000, mono=True)
            inputs = self.fe(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.inference_mode():
                if AMP:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(**inputs).logits
                else:
                    logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0].detach().to("cpu").numpy()
            idx = np.argmax(probs)
            genre = self.id2label[idx]
            return genre
        except Exception as e:
            log("AST genre classification FAILED:", e)
            return "Unknown"

# ================== UnifiedAudioPipeline ==================
class UnifiedAudioPipeline:
    def __init__(self):
        self.genre_classifier = GenreClassifier()  # Use AST for genre classification
        self.m2e = M2EWrapperEnsemble(CANDIDATE_CKPTS)
        log("device:", DEVICE, "| AMP:", AMP)

    def _detect_genre(self, path):
        genre = self.genre_classifier.predict_genre(path)
        log(f"Genre Prediction (AST): {genre}")
        return genre

    # Extract instruments using audio features
    def _extract_instruments(self, path):
        inst = set()
        try:
            audio, sr = librosa.load(path, sr=16000, mono=True)  # Load audio file
            # Extract features using the feature extractor
            inputs = self.genre_classifier.fe(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.inference_mode():
                logits = self.genre_classifier.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0].detach().to("cpu").numpy()

            # Mapping the output labels to instruments
            predicted_labels = np.argsort(-probs)[:TOPK]
            for idx in predicted_labels:
                label = self.genre_classifier.id2label[idx]
                inst_canon = _canon_instr(label)
                if inst_canon:
                    inst.add(inst_canon)

            log(f"Detected Instruments: {inst}")
        except Exception as e:
            log("Instrument detection failed:", e)
        return sorted(inst)

    def analyze_with_separation(self, audio_path, status_cb=None, progress_cb=None):
        ensure_dirs()
        results = {}

        if progress_cb: progress_cb(0.10, "Separating stems…")
        stems, sr = separate_with_demucs(audio_path, DEVICE)
        if not stems:
            results["Error"] = "Separation failed"; return results

        if progress_cb: progress_cb(0.30, "Saving stems…")
        saved_paths = {lab: save_audio_stem(lab.capitalize(), arr, sr, audio_path)
                       for lab, arr in stems.items()}
        results["Saved_Stems"] = saved_paths

        if progress_cb: progress_cb(0.40, "Detecting genre…")
        results["Genre"] = self._detect_genre(audio_path)

        if progress_cb: progress_cb(0.65, "Predicting emotions (ensemble)…")
        emo = self.m2e.predict_full(audio_path, status_cb=status_cb)
        results["EmotionFull"] = emo
        results["Emotion"] = emo.get("primary", "Unknown")

        if progress_cb: progress_cb(0.85, "Extracting instruments…")
        results["Instruments"] = self._extract_instruments(audio_path)

        if progress_cb: progress_cb(0.95, "Writing JSON…")
        base = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(RESULTS_DIR, f"{base}_analysis.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        results["JSON"] = json_path
        return results

    # analyze a single file directly (no separation)
    def analyze_raw_file(self, audio_path, status_cb=None, progress_cb=None):
        ensure_dirs()
        results = {}
        if progress_cb: progress_cb(0.25, "Detecting genre…")
        results["Genre"] = self._detect_genre(audio_path)

        if progress_cb: progress_cb(0.55, "Predicting emotions (ensemble)…")
        emo = self.m2e.predict_full(audio_path, status_cb=status_cb)
        results["EmotionFull"] = emo
        results["Emotion"] = emo.get("primary", "Unknown")

        if progress_cb: progress_cb(0.75, "Detecting instruments…")
        results["Instruments"] = self._extract_instruments(audio_path)

        if progress_cb: progress_cb(0.90, "Writing JSON…")
        base = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(RESULTS_DIR, f"{base}_analysis.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        results["JSON"] = json_path
        return results


# ================== UI ==================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎵 Music Analyzer (Ensemble)")
        self.geometry("1280x860")
        self.configure(bg="#0b132b")  # deep navy
        self.columnconfigure(0, weight=1)

        # THEME
        style = ttk.Style(self)
        style.theme_use("clam")
        # panels / controls
        style.configure("TFrame", background="#0b132b")
        style.configure("Card.TFrame", background="#1c2541")
        style.configure("TLabel", background="#0b132b", foreground="#e0e6f6")
        style.configure("Card.TLabel", background="#1c2541", foreground="#e0e6f6")
        style.configure("TButton", padding=10, background="#5bc0be", foreground="#061a40", borderwidth=0)
        style.map("TButton", background=[("active", "#4aaead")])
        style.configure("TScale", troughcolor="#243b55", background="#0b132b")
        style.configure("Horizontal.TProgressbar", troughcolor="#243b55", background="#5bc0be")

        # Top bar
        top = ttk.Frame(self, style="TFrame"); top.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Upload Audio Files", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        self.btn_choose = ttk.Button(top, text="📂 Choose Files", command=self.choose_files)
        self.btn_choose.grid(row=0, column=2, sticky="e", padx=(10, 0))
        self.lbl_files = ttk.Label(top, text="No file selected")
        self.lbl_files.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        # Options card
        card = ttk.Frame(self, style="Card.TFrame"); card.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        card.columnconfigure(1, weight=1)
        ttk.Label(card, text="Mood Detection Threshold", style="Card.TLabel", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(10,0))
        self.thr_value = tk.DoubleVar(value=DEFAULT_MOOD_THRESHOLD)
        self.scale = ttk.Scale(card, from_=0, to=1, orient="horizontal", variable=self.thr_value)
        self.scale.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10)
        self.lbl_thr = ttk.Label(card, text=f"{self.thr_value.get():.2f}", style="Card.TLabel")
        self.lbl_thr.grid(row=1, column=2, sticky="e", padx=(8,10))
        self.thr_value.trace_add("write", lambda *_: self.lbl_thr.config(text=f"{self.thr_value.get():.2f}"))

        self.skip_sep = tk.BooleanVar(value=False)
        ttk.Checkbutton(card, text="Skip Demucs (I selected stems / raw tracks)", variable=self.skip_sep,
                        style="Card.TLabel").grid(row=2, column=0, sticky="w", padx=10, pady=(8,10))

        # Actions row
        action = ttk.Frame(self, style="TFrame")
        action.grid(row=2, column=0, sticky="ew", padx=14)
        action.columnconfigure(0, weight=1)

        # Curated tags checkbox ABOVE the Analyze button
        self.curated_checkbox = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            action,
            text="Use Curated Tags for Accuracy",
            variable=self.curated_checkbox,
            style="Card.TLabel"
        ).grid(row=0, column=0, sticky="w", padx=(0, 14), pady=(0, 12))

        # Analyze button (below curated checkbox)
        self.btn_analyze = ttk.Button(action, text="🎧 Analyze (Ensemble)", command=self.analyze_clicked)
        self.btn_analyze.grid(row=1, column=0, sticky="w")
        self.pbar = ttk.Progressbar(action, mode="determinate", length=320, style="Horizontal.TProgressbar")
        self.pbar.grid(row=1, column=1, sticky="e")


        # Middle: logs & results
        mid = ttk.Frame(self, style="TFrame"); mid.grid(row=3, column=0, sticky="nsew", padx=14, pady=(10, 8))
        self.rowconfigure(3, weight=1)
        mid.columnconfigure(0, weight=3)
        mid.columnconfigure(1, weight=2)
        mid.rowconfigure(0, weight=1)

        # Log panel
        log_frame = ttk.Frame(mid, style="Card.TFrame"); log_frame.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        ttk.Label(log_frame, text="Log", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(8,0))
        self.txt_log = tk.Text(log_frame, height=12, bg="#1c2541", fg="#a9e9e7", insertbackground="white",
                               relief="flat", font=("Consolas", 10))
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=(6,10))

        # Results panel
        res_frame = ttk.Frame(mid, style="Card.TFrame"); res_frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(res_frame, text="Analysis Results", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(8,0))
        self.txt_res = tk.Text(res_frame, height=12, bg="#1c2541", fg="#e0e6f6", insertbackground="white",
                               relief="flat", font=("Consolas", 10))
        self.txt_res.pack(fill="both", expand=True, padx=10, pady=(6,10))

        # Bottom: plots
        bot = ttk.Frame(self, style="TFrame"); bot.grid(row=4, column=0, sticky="nsew", padx=14, pady=(0,12))
        self.rowconfigure(4, weight=1)
        bot.columnconfigure(0, weight=3)
        bot.columnconfigure(1, weight=2)

        left_plot = ttk.Frame(bot, style="Card.TFrame"); left_plot.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        ttk.Label(left_plot, text="Mood Probabilities", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(8,0))
        self.tk_bar = None
        self.left_plot_host = ttk.Frame(left_plot, style="Card.TFrame"); self.left_plot_host.pack(fill="both", expand=True, padx=10, pady=(6,10))

        right_plot = ttk.Frame(bot, style="Card.TFrame"); right_plot.grid(row=0, column=1, sticky="nsew")
        ttk.Label(right_plot, text="Valence–Arousal Space", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(8,0))
        self.tk_va = None
        self.right_plot_host = ttk.Frame(right_plot, style="Card.TFrame"); self.right_plot_host.pack(fill="both", expand=True, padx=10, pady=(6,10))

        # Status strip
        status = ttk.Frame(self, style="TFrame"); status.grid(row=5, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Ready.")
        self.status_lbl = ttk.Label(status, textvariable=self.status_var, anchor="w")
        self.status_lbl.grid(row=0, column=0, sticky="ew", padx=14, pady=(0,8))

        # Pipeline + state
        try:
            self.pipeline = UnifiedAudioPipeline()
        except Exception as e:
            messagebox.showerror("Model error", f"Could not initialize models.\n{e}")
            raise
        self.files = []
        self.log_line(f"Device: {DEVICE} | AMP: {AMP}")

    # ===== UI helpers =====
    def log_line(self, s): self.txt_log.insert("end", s + "\n"); self.txt_log.see("end")

    def choose_files(self):
        fs = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
        if not fs: return
        self.files = list(fs)
        self.lbl_files.config(text=f"{len(self.files)} file(s) selected")

    def analyze_clicked(self):
        if not self.files:
            messagebox.showinfo("No files", "Please choose at least one audio file.")
            return
        thr = float(self.thr_value.get())
        skip = bool(self.skip_sep.get())
        self.btn_analyze.state(["disabled"]); self.pbar["value"] = 0
        threading.Thread(target=self._process_files, args=(self.files, thr, skip), daemon=True).start()

    def _process_files(self, files, thr, skip_sep):
        for f in files:
            self.log_line(f"Processing: {f}")
            def progress_cb(frac, msg):
                self.pbar["value"] = max(0, min(100, int(frac*100)))
                self.pbar.update_idletasks()
                if msg: self.log_line(msg)
            def status_cb(msg):
                self.status_var.set(msg)

            try:
                if skip_sep:
                    results = self.pipeline.analyze_raw_file(f, status_cb=status_cb, progress_cb=progress_cb)
                else:
                    results = self.pipeline.analyze_with_separation(f, status_cb=status_cb, progress_cb=progress_cb)
                self.render_results(f, results, thr)
            except Exception as e:
                self.log_line(f"Error: {e}")
        self.status_var.set("Ready.")
        self.btn_analyze.state(["!disabled"])
        messagebox.showinfo("Done", f"Processed {len(files)} file(s).")

    # ===== plotting =====
    def draw_bar(self, items):
        if self.tk_bar: self.tk_bar.get_tk_widget().destroy()
        fig = Figure(figsize=(8, 3), dpi=100); ax = fig.add_subplot(111)
        labels = [a for a,_ in items][:TOPK]; vals = [b for _,b in items][:TOPK]
        y = np.arange(len(labels)); ax.barh(y, vals)
        ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
        ax.set_xlabel("Probability"); ax.set_title("Top Predicted Mood Tags"); fig.tight_layout()
        self.tk_bar = FigureCanvasTkAgg(fig, master=self.left_plot_host); self.tk_bar.draw()
        self.tk_bar.get_tk_widget().pack(fill="both", expand=True)

    def draw_va(self, valence_raw, arousal_raw):
        if self.tk_va: self.tk_va.get_tk_widget().destroy()
        fig = Figure(figsize=(3.2, 2.8), dpi=100); ax = fig.add_subplot(111)
        ax.set_xlim(1, 9); ax.set_ylim(1, 9)
        ax.set_xlabel("Valence (1–9)"); ax.set_ylabel("Arousal (1–9)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        v9 = _rescale_1_9_if_raw(valence_raw) if valence_raw is not None else None
        a9 = _rescale_1_9_if_raw(arousal_raw) if arousal_raw is not None else None
        if (v9 is not None) and (a9 is not None):
            ax.scatter([v9], [a9], s=40)
        fig.tight_layout()
        self.tk_va = FigureCanvasTkAgg(fig, master=self.right_plot_host); self.tk_va.draw()
        self.tk_va.get_tk_widget().pack(fill="both", expand=True)

    # ===== render =====
    def render_results(self, path, results, thr):
        base = os.path.basename(path)
        emo = results.get("EmotionFull", {"mood_tags": [], "valence": None, "arousal": None, "primary": "Unknown"})
        tags_all = sorted(emo.get("mood_tags", []), key=lambda x: x[1], reverse=True)
        tags_thr = [(l,p) for (l,p) in tags_all if p >= thr]
        v_raw, a_raw = emo.get("valence"), emo.get("arousal")
        v9 = _rescale_1_9_if_raw(v_raw) if v_raw is not None else None
        a9 = _rescale_1_9_if_raw(a_raw) if a_raw is not None else None

        self.txt_res.delete("1.0", "end")
        self.txt_res.insert("end", f"📄 File: {base}\n\n")
        disp = tags_thr if tags_thr else tags_all
        s = ", ".join([f"{lab} ({prob:.2f})" for lab, prob in disp[:TOPK]]) or "(no tags)"
        self.txt_res.insert("end", f"🎭 Predicted Mood Tags: {s}\n\n")

        if v_raw is not None and a_raw is not None:
            self.txt_res.insert("end", f"💖 Valence: raw {v_raw:.2f}" + (f" → {v9:.2f} (1–9)\n" if v9 is not None else "\n"))
            self.txt_res.insert("end", f"⚡ Arousal: raw {a_raw:.2f}" + (f" → {a9:.2f} (1–9)\n" if a9 is not None else "\n"))
        else:
            self.txt_res.insert("end", "💖 Valence: n/a\n⚡ Arousal: n/a\n")

        self.txt_res.insert("end", f"\n🎼 Genre (AST): {results.get('Genre', 'Unknown')}\n")
        self.txt_res.insert("end", "🥁 Instruments: " + (", ".join(results.get("Instruments", [])) or "(none)") + "\n")
        self.txt_res.insert("end", f"📝 JSON: {results.get('JSON')}\n")

        self.draw_bar(tags_all)
        self.draw_va(v_raw, a_raw)

# ---- bootstrap
if __name__ == "__main__":
    app = App()
    app.mainloop()
