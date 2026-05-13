import os
import sys
import json
import queue
import atexit
import threading
import tempfile
import subprocess
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io import wavfile

warnings.filterwarnings("ignore", category=UserWarning, message="The figure layout has changed to tight")

APP_TITLE = "Resonance Tone & Cymatics Studio"
SAMPLE_RATE = 44100
MIN_FREQ = 20
MAX_FREQ = 2000
PREVIEW_SECONDS = 2.0

PRESETS_JSON = r'''[
  {
    "id": "shreem_calm_432_mono",
    "name": "Shreem Calm 432",
    "mantra": "SHREEM",
    "background_mode": "mono",
    "carrier_hz": 432.0,
    "binaural_beat_hz": 0.0,
    "left_hz": 432.0,
    "right_hz": 432.0,
    "mantra_gain": 0.75,
    "tone_gain": 0.12,
    "ambient_gain": 0.0,
    "session_minutes": 20,
    "preview_seconds": 20,
    "five_min_download": true,
    "headphones_required": false,
    "notes": "Soft mono support tone for relaxed chanting."
  },
  {
    "id": "shreem_prosperity_528_mono",
    "name": "Shreem Prosperity 528",
    "mantra": "SHREEM",
    "background_mode": "mono",
    "carrier_hz": 528.0,
    "binaural_beat_hz": 0.0,
    "left_hz": 528.0,
    "right_hz": 528.0,
    "mantra_gain": 0.75,
    "tone_gain": 0.12,
    "ambient_gain": 0.0,
    "session_minutes": 20,
    "preview_seconds": 20,
    "five_min_download": true,
    "headphones_required": false,
    "notes": "Most practical default preset for Shreem + support tone."
  },
  {
    "id": "shreem_subtle_528_mono",
    "name": "Shreem Subtle 528",
    "mantra": "SHREEM",
    "background_mode": "mono",
    "carrier_hz": 528.0,
    "binaural_beat_hz": 0.0,
    "left_hz": 528.0,
    "right_hz": 528.0,
    "mantra_gain": 0.85,
    "tone_gain": 0.06,
    "ambient_gain": 0.0,
    "session_minutes": 15,
    "preview_seconds": 20,
    "five_min_download": true,
    "headphones_required": false,
    "notes": "Very light support tone with mantra clearly dominant."
  },
  {
    "id": "shreem_focus_432_binaural_8",
    "name": "Shreem Focus 432 / 8 Hz Binaural",
    "mantra": "SHREEM",
    "background_mode": "binaural",
    "carrier_hz": 432.0,
    "binaural_beat_hz": 8.0,
    "left_hz": 428.0,
    "right_hz": 436.0,
    "mantra_gain": 0.65,
    "tone_gain": 0.08,
    "ambient_gain": 0.0,
    "session_minutes": 15,
    "preview_seconds": 20,
    "five_min_download": true,
    "headphones_required": true,
    "notes": "Stereo headphones only; relaxed focus."
  },
  {
    "id": "shreem_relax_528_binaural_6",
    "name": "Shreem Deep Relax 528 / 6 Hz Binaural",
    "mantra": "SHREEM",
    "background_mode": "binaural",
    "carrier_hz": 528.0,
    "binaural_beat_hz": 6.0,
    "left_hz": 525.0,
    "right_hz": 531.0,
    "mantra_gain": 0.70,
    "tone_gain": 0.08,
    "ambient_gain": 0.0,
    "session_minutes": 20,
    "preview_seconds": 20,
    "five_min_download": true,
    "headphones_required": true,
    "notes": "Stereo headphones only; deeper meditation style."
  },
  {
    "id": "shreem_theta_528_binaural_4",
    "name": "Shreem Theta 528 / 4 Hz Binaural",
    "mantra": "SHREEM",
    "background_mode": "binaural",
    "carrier_hz": 528.0,
    "binaural_beat_hz": 4.0,
    "left_hz": 526.0,
    "right_hz": 530.0,
    "mantra_gain": 0.68,
    "tone_gain": 0.07,
    "ambient_gain": 0.0,
    "session_minutes": 15,
    "preview_seconds": 20,
    "five_min_download": true,
    "headphones_required": true,
    "notes": "Stereo headphones only; subtle slow binaural motion."
  }
]'''

PRESETS = json.loads(PRESETS_JSON)
PRESET_MAP = {p["name"]: p for p in PRESETS}

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    sd = None
    SOUNDDEVICE_AVAILABLE = False


def stop_all_audio():
    if sys.platform.startswith("win"):
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
    else:
        try:
            subprocess.run(["pkill", "-f", "afplay|aplay"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def play_audio_file(filepath):
    if sys.platform.startswith("win"):
        import winsound
        winsound.PlaySound(filepath, winsound.SND_FILENAME | winsound.SND_ASYNC)
        return True, "Playing audio preview with winsound."
    player = "afplay" if sys.platform == "darwin" else "aplay"
    try:
        subprocess.Popen([player, filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, f"Playing audio preview with {player}."
    except FileNotFoundError:
        return False, f"Audio player '{player}' not found on this system."


def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def normalize_audio(x, peak=0.95):
    x = np.asarray(x, dtype=np.float32)
    m = np.max(np.abs(x)) if x.size else 0.0
    if m < 1e-9:
        return x
    return (x / m) * peak


def ensure_stereo(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.column_stack([x, x])
    if x.ndim == 2 and x.shape[1] == 1:
        return np.column_stack([x[:, 0], x[:, 0]])
    return x


def mono_mixdown(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def apply_fade(signal, sample_rate=SAMPLE_RATE, fade_ms=40):
    x = np.asarray(signal, dtype=np.float32).copy()
    if len(x) == 0:
        return x
    n = max(1, int(sample_rate * fade_ms / 1000))
    n = min(n, len(x) // 2)
    if n <= 1:
        return x
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    if x.ndim == 1:
        x[:n] *= fade_in
        x[-n:] *= fade_out
    else:
        x[:n, :] *= fade_in[:, None]
        x[-n:, :] *= fade_out[:, None]
    return x


def loop_audio_to_length(signal, total_samples, sample_rate=SAMPLE_RATE, crossfade_ms=35):
    x = np.asarray(signal, dtype=np.float32)
    if x.size == 0:
        return np.zeros(total_samples, dtype=np.float32) if x.ndim == 1 else np.zeros((total_samples, x.shape[1]), dtype=np.float32)
    if len(x) >= total_samples:
        return x[:total_samples].copy()
    out = np.zeros((total_samples,) + x.shape[1:], dtype=np.float32)
    idx = 0
    fade_n = max(1, int(sample_rate * crossfade_ms / 1000))
    while idx < total_samples:
        remaining = total_samples - idx
        chunk = x[: min(len(x), remaining)].copy()
        if idx > 0 and len(chunk) > fade_n:
            if chunk.ndim == 1:
                chunk[:fade_n] *= np.linspace(0.0, 1.0, fade_n)
                out[idx: idx + fade_n] *= np.linspace(1.0, 0.0, fade_n)
            else:
                ramp_in = np.linspace(0.0, 1.0, fade_n)[:, None]
                ramp_out = np.linspace(1.0, 0.0, fade_n)[:, None]
                chunk[:fade_n] *= ramp_in
                out[idx: idx + fade_n] *= ramp_out
        out[idx: idx + len(chunk)] += chunk
        idx += len(chunk)
    return out[:total_samples]


def resample_audio(data, src_rate, dst_rate=SAMPLE_RATE):
    x = np.asarray(data, dtype=np.float32)
    if src_rate == dst_rate or len(x) == 0:
        return x
    src_len = len(x)
    dst_len = int(round(src_len * dst_rate / src_rate))
    src_idx = np.linspace(0, src_len - 1, src_len)
    dst_idx = np.linspace(0, src_len - 1, dst_len)
    if x.ndim == 1:
        return np.interp(dst_idx, src_idx, x).astype(np.float32)
    channels = [np.interp(dst_idx, src_idx, x[:, ch]) for ch in range(x.shape[1])]
    return np.column_stack(channels).astype(np.float32)


def generate_sine(freq, duration, sample_rate=SAMPLE_RATE, amp=0.1):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def generate_binaural(carrier=432.0, beat_hz=6.0, duration=300.0, sample_rate=SAMPLE_RATE, amp=0.08):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    left = np.sin(2 * np.pi * (carrier - beat_hz / 2.0) * t)
    right = np.sin(2 * np.pi * (carrier + beat_hz / 2.0) * t)
    stereo = np.column_stack([left, right]).astype(np.float32)
    return apply_fade(normalize_audio(stereo, peak=amp), sample_rate=sample_rate, fade_ms=2000)


def write_wav(path, data, sample_rate=SAMPLE_RATE):
    x = np.asarray(data, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(path, sample_rate, np.int16(x * 32767))


class ImprovedCymaticsDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1300x930")
        self.root.configure(bg="#F4F7FA")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.temp_audio_path = os.path.join(tempfile.gettempdir(), "resonance_preview.wav")
        self.temp_mix_preview_path = os.path.join(tempfile.gettempdir(), "resonance_mix_preview.wav")
        self.recorded_temp_path = os.path.join(tempfile.gettempdir(), "resonance_recorded_mantra.wav")

        self.current_density_matrix = None
        self.current_n = 2
        self.current_m = 4
        self.target_frequency = 528.0
        self.energy_gain_amplitude = 0.35
        self.current_keyword = "SHREEM"
        self.last_count = 108
        self.is_updating_from_slider = False
        self.busy = False
        self.status_queue = queue.Queue()
        self.progress_value = tk.DoubleVar(value=0.0)
        self.progress_percent_var = tk.StringVar(value="0%")
        self.mantra_audio = None
        self.mantra_source_path = None
        self.mantra_sample_rate = SAMPLE_RATE
        self.active_preset = PRESETS[1]

        atexit.register(lambda: safe_remove(self.temp_audio_path))
        atexit.register(lambda: safe_remove(self.temp_mix_preview_path))
        atexit.register(lambda: safe_remove(self.recorded_temp_path))

        self._setup_style()
        self._build_layout()
        self.reset_canvas_viewports()
        self.apply_selected_preset()
        self.poll_status_queue()
        self.update_log("Ready. Load or record a mantra, choose a preset, then compute and export.")

    def _setup_style(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#F4F7FA", foreground="#1F2933")
        self.style.configure("TScale", background="#FFFFFF")
        self.style.configure("TProgressbar", troughcolor="#E5EAF0", background="#2563EB", bordercolor="#E5EAF0")

    def _build_layout(self):
        header = tk.Frame(self.root, bg="#FFFFFF", height=58, highlightthickness=1, highlightbackground="#D8E0E8")
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text=APP_TITLE, fg="#2563EB", bg="#FFFFFF", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT, padx=18)
        tk.Label(header, text="Preset + 5-minute download edition", fg="#5B6570", bg="#FFFFFF", font=("Segoe UI", 9, "bold")).pack(side=tk.RIGHT, padx=18)

        main = tk.Frame(self.root, bg="#F4F7FA")
        main.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        left = tk.Frame(main, bg="#FFFFFF", width=420, highlightthickness=1, highlightbackground="#D8E0E8")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        left.pack_propagate(False)

        right = tk.Frame(main, bg="#F4F7FA")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        left_canvas = tk.Canvas(left, bg="#FFFFFF", highlightthickness=0, bd=0)
        left_scroll = tk.Scrollbar(left, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inputs = tk.Frame(left_canvas, bg="#FFFFFF")
        canvas_window = left_canvas.create_window((0, 0), window=inputs, anchor="nw")

        def _sync_left_scrollregion(event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _resize_embedded_frame(event):
            left_canvas.itemconfig(canvas_window, width=event.width)

        inputs.bind("<Configure>", _sync_left_scrollregion)
        left_canvas.bind("<Configure>", _resize_embedded_frame)

        tk.Label(inputs, text="Preset", bg="#FFFFFF", fg="#425466", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.preset_var = tk.StringVar(value=PRESETS[1]["name"])
        self.preset_combo = ttk.Combobox(inputs, textvariable=self.preset_var, state="readonly", values=[p["name"] for p in PRESETS], width=38)
        self.preset_combo.pack(fill=tk.X, pady=(0, 6))
        self.preset_combo.bind("<<ComboboxSelected>>", self.apply_selected_preset)
        self.preset_notes_var = tk.StringVar(value="")
        tk.Label(inputs, textvariable=self.preset_notes_var, wraplength=360, justify=tk.LEFT, bg="#FFFFFF", fg="#6B7280", font=("Segoe UI", 8)).pack(anchor=tk.W, pady=(0, 12))

        tk.Label(inputs, text="1. Keyword / mantra", bg="#FFFFFF", fg="#425466", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.word_entry = tk.Entry(inputs, font=("Segoe UI", 11), bg="#FBFCFD", fg="#1F2933", bd=0, highlightthickness=1, highlightbackground="#CAD4DE", highlightcolor="#2563EB")
        self.word_entry.pack(fill=tk.X, ipady=7, pady=(0, 8))
        self.word_entry.insert(0, "SHREEM")
        tk.Button(inputs, text="Convert text to suggested Hz", bg="#2563EB", fg="#FFFFFF", font=("Segoe UI", 8, "bold"), bd=0, cursor="hand2", command=self.convert_text_to_frequency).pack(anchor=tk.E, pady=(0, 12))

        tk.Label(inputs, text="2. Frequency target (Hz)", bg="#FFFFFF", fg="#425466", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.freq_var = tk.StringVar(value="528")
        self.freq_var.trace_add("write", self.on_text_box_modified)
        self.freq_entry = tk.Entry(inputs, textvariable=self.freq_var, font=("Segoe UI", 11), bg="#FBFCFD", fg="#1F2933", bd=0, highlightthickness=1, highlightbackground="#CAD4DE", highlightcolor="#2563EB")
        self.freq_entry.pack(fill=tk.X, ipady=7, pady=(0, 5))
        self.freq_slider = ttk.Scale(inputs, from_=MIN_FREQ, to=MAX_FREQ, orient=tk.HORIZONTAL, command=self.on_slider_dragged)
        self.freq_slider.pack(fill=tk.X, pady=(0, 10))
        self.freq_slider.set(528)

        tk.Label(inputs, text="3. Repetition count", bg="#FFFFFF", fg="#425466", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.count_entry = tk.Entry(inputs, font=("Segoe UI", 11), bg="#FBFCFD", fg="#1F2933", bd=0, highlightthickness=1, highlightbackground="#CAD4DE", highlightcolor="#2563EB")
        self.count_entry.pack(fill=tk.X, ipady=7, pady=(0, 12))
        self.count_entry.insert(0, "108")

        tk.Label(inputs, text="Text-to-Hz is a creative mapping for experimentation, not a scientific or traditional fixed mantra frequency.", wraplength=360, justify=tk.LEFT, bg="#FFFFFF", fg="#6B7280", font=("Segoe UI", 8)).pack(anchor=tk.W, pady=(0, 12))
        tk.Frame(inputs, height=1, bg="#E5EAF0").pack(fill=tk.X, pady=6)

        tk.Label(inputs, text="4. Mantra input", bg="#FFFFFF", fg="#425466", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(6, 6))
        row = tk.Frame(inputs, bg="#FFFFFF")
        row.pack(fill=tk.X, pady=(0, 6))
        tk.Button(row, text="Upload WAV", bg="#2563EB", fg="#FFFFFF", font=("Segoe UI", 8, "bold"), bd=0, cursor="hand2", command=self.load_mantra_wav).pack(side=tk.LEFT, padx=(0, 6), ipadx=8, ipady=5)
        self.record_btn = tk.Button(row, text="Record mic", bg="#16A34A", fg="#FFFFFF", font=("Segoe UI", 8, "bold"), bd=0, cursor="hand2", command=self.record_mantra_dialog)
        self.record_btn.pack(side=tk.LEFT, padx=(0, 6), ipadx=8, ipady=5)
        tk.Button(row, text="Clear", bg="#E5EAF0", fg="#1F2933", font=("Segoe UI", 8, "bold"), bd=0, cursor="hand2", command=self.clear_mantra).pack(side=tk.LEFT, ipadx=8, ipady=5)
        self.mantra_status_var = tk.StringVar(value="No mantra loaded. Tone-only export will be used.")
        tk.Label(inputs, textvariable=self.mantra_status_var, wraplength=360, justify=tk.LEFT, bg="#FFFFFF", fg="#6B7280", font=("Segoe UI", 8)).pack(anchor=tk.W, pady=(0, 12))
        tk.Frame(inputs, height=1, bg="#E5EAF0").pack(fill=tk.X, pady=6)

        tk.Label(inputs, text="5. Mix settings", bg="#FFFFFF", fg="#425466", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(6, 6))
        r1 = tk.Frame(inputs, bg="#FFFFFF"); r1.pack(fill=tk.X, pady=(0, 8))
        tk.Label(r1, text="Background mode", bg="#FFFFFF", fg="#425466", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.background_mode = tk.StringVar(value="mono")
        ttk.Combobox(r1, textvariable=self.background_mode, state="readonly", values=["off", "mono", "binaural"], width=12).pack(side=tk.RIGHT)

        r2 = tk.Frame(inputs, bg="#FFFFFF"); r2.pack(fill=tk.X, pady=(0, 8))
        tk.Label(r2, text="Binaural beat (Hz)", bg="#FFFFFF", fg="#425466", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.beat_hz_var = tk.StringVar(value="6")
        ttk.Combobox(r2, textvariable=self.beat_hz_var, state="readonly", values=["0", "4", "6", "8", "10"], width=12).pack(side=tk.RIGHT)

        r3 = tk.Frame(inputs, bg="#FFFFFF"); r3.pack(fill=tk.X, pady=(0, 8))
        tk.Label(r3, text="Session duration (min)", bg="#FFFFFF", fg="#425466", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="20")
        ttk.Combobox(r3, textvariable=self.duration_var, state="readonly", values=["5", "15", "20", "30", "60"], width=12).pack(side=tk.RIGHT)

        tk.Label(inputs, text="Mantra level", bg="#FFFFFF", fg="#425466", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.mantra_gain_scale = ttk.Scale(inputs, from_=0.1, to=1.0, orient=tk.HORIZONTAL)
        self.mantra_gain_scale.pack(fill=tk.X, pady=(2, 8))
        self.mantra_gain_scale.set(0.75)

        tk.Label(inputs, text="Tone level", bg="#FFFFFF", fg="#425466", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.tone_gain_scale = ttk.Scale(inputs, from_=0.0, to=0.4, orient=tk.HORIZONTAL)
        self.tone_gain_scale.pack(fill=tk.X, pady=(2, 8))
        self.tone_gain_scale.set(0.12)

        self.headphone_var = tk.BooleanVar(value=True)
        tk.Checkbutton(inputs, text="Headphones required for binaural mode", variable=self.headphone_var, bg="#FFFFFF", fg="#425466", activebackground="#FFFFFF", activeforeground="#425466", selectcolor="#FFFFFF").pack(anchor=tk.W, pady=(0, 10))

        tk.Button(inputs, text="Compute visuals + tone preview", bg="#16A34A", fg="#FFFFFF", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", command=self.execute_acoustics_pipeline).pack(fill=tk.X, ipady=9, pady=(6, 8))
        self.preview_mix_btn = tk.Button(inputs, text="▶ Preview mantra + tone mix", bg="#2563EB", fg="#FFFFFF", disabledforeground="#FFFFFF", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", command=self.preview_mix)
        self.preview_mix_btn.pack(fill=tk.X, ipady=9, pady=(0, 8))
        self.play_btn = tk.Button(inputs, text="▶ Preview tone only", bg="#E5EAF0", fg="#1F2933", disabledforeground="#8A94A6", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", state=tk.DISABLED, command=self.audio_playback_trigger)
        self.play_btn.pack(fill=tk.X, ipady=9, pady=(0, 8))
        tk.Button(inputs, text="■ Stop audio", bg="#E5EAF0", fg="#1F2933", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", command=stop_all_audio).pack(fill=tk.X, ipady=9, pady=(0, 12))

        self.export_audio_btn = tk.Button(inputs, text="Save full session WAV", bg="#2563EB", fg="#FFFFFF", disabledforeground="#FFFFFF", font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2", state=tk.DISABLED, command=self.export_audio_file)
        self.export_audio_btn.pack(fill=tk.X, ipady=7, pady=(0, 8))
        self.export_5min_btn = tk.Button(inputs, text="Download 5-minute preset WAV", bg="#0F766E", fg="#FFFFFF", disabledforeground="#FFFFFF", font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2", state=tk.DISABLED, command=self.export_five_minute_audio)
        self.export_5min_btn.pack(fill=tk.X, ipady=7, pady=(0, 8))
        self.export_plot_btn = tk.Button(inputs, text="Save pattern PNG", bg="#2563EB", fg="#FFFFFF", disabledforeground="#FFFFFF", font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2", state=tk.DISABLED, command=self.export_cymatics_plot)
        self.export_plot_btn.pack(fill=tk.X, ipady=7, pady=(0, 8))

        tk.Label(inputs, text="Safety: keep volume low, stop if uncomfortable, and use stereo headphones for binaural mode.", wraplength=360, justify=tk.LEFT, bg="#FFFFFF", fg="#8B5E00", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W, pady=(0, 10))
        self.progress = ttk.Progressbar(inputs, mode="determinate", maximum=100, variable=self.progress_value)
        self.progress.pack(fill=tk.X, pady=(0, 4))
        tk.Label(inputs, textvariable=self.progress_percent_var, bg="#FFFFFF", fg="#425466", font=("Segoe UI", 8, "bold")).pack(anchor=tk.E, pady=(0, 6))
        self.status_var = tk.StringVar(value="Idle")
        tk.Label(inputs, textvariable=self.status_var, bg="#FFFFFF", fg="#6B7280", font=("Segoe UI", 8, "italic")).pack(anchor=tk.W)

        tk.Label(inputs, text="Console log", fg="#6B7280", bg="#FFFFFF", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W, pady=(10, 2))
        self.log_display = tk.Text(inputs, height=12, bg="#FBFCFD", fg="#1F2933", font=("Consolas", 9), bd=0, highlightthickness=1, highlightbackground="#E5EAF0", wrap=tk.WORD)
        self.log_display.pack(fill=tk.BOTH, expand=False, pady=(0, 14))
        self.log_display.config(state=tk.DISABLED)

        self.fig, self.axes = plt.subplots(1, 2, figsize=(8.8, 5.0), facecolor="#FFFFFF")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_log(self, text):
        self.log_display.config(state=tk.NORMAL)
        self.log_display.insert(tk.END, text + "\n")
        self.log_display.see(tk.END)
        self.log_display.config(state=tk.DISABLED)

    def reset_canvas_viewports(self):
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor("#FBFCFD")
            ax.tick_params(colors="#5B6570", labelsize=8)
            for side in ["bottom", "top", "left", "right"]:
                ax.spines[side].set_color("#E5EAF0")
        self.axes[0].set_title("Pattern field", color="#1F2933", fontsize=10, fontweight="bold", pad=10)
        self.axes[1].set_title("Waveform preview", color="#1F2933", fontsize=10, fontweight="bold", pad=10)
        self.fig.tight_layout()
        self.canvas.draw()

    def update_progress(self, value, message=None):
        value = max(0.0, min(100.0, float(value)))
        self.progress_value.set(value)
        self.progress_percent_var.set(f"{int(round(value))}%")
        if message:
            self.status_var.set(message)

    def set_busy(self, state, message="Working..."):
        self.busy = state
        if state:
            self.preview_mix_btn.config(state=tk.DISABLED, bg="#2563EB", activebackground="#2563EB")
            self.play_btn.config(state=tk.DISABLED, bg="#E5EAF0", activebackground="#E5EAF0")
            self.export_audio_btn.config(state=tk.DISABLED, bg="#2563EB", activebackground="#2563EB")
            self.export_5min_btn.config(state=tk.DISABLED, bg="#0F766E", activebackground="#0F766E")
            self.export_plot_btn.config(state=tk.DISABLED, bg="#2563EB", activebackground="#2563EB")
            self.record_btn.config(state=tk.DISABLED, bg="#16A34A", activebackground="#16A34A")
            self.update_progress(0, message)
        else:
            self.update_progress(100, "Completed")
            self.status_var.set("Idle")
            self.preview_mix_btn.config(state=tk.NORMAL, bg="#2563EB", activebackground="#1D4ED8")
            self.record_btn.config(state=tk.NORMAL, bg="#16A34A", activebackground="#15803D")
            self.play_btn.config(state=tk.NORMAL if os.path.exists(self.temp_audio_path) else tk.DISABLED, bg="#E5EAF0", activebackground="#D7DEE7")
            s = tk.NORMAL if self.current_density_matrix is not None else tk.DISABLED
            self.export_audio_btn.config(state=s, bg="#2563EB", activebackground="#1D4ED8")
            self.export_5min_btn.config(state=s, bg="#0F766E", activebackground="#115E59")
            self.export_plot_btn.config(state=s, bg="#2563EB", activebackground="#1D4ED8")

    def poll_status_queue(self):
        try:
            while True:
                item = self.status_queue.get_nowait()
                kind = item[0]
                if kind == "ready":
                    self._apply_compute_result(*item[1:])
                elif kind == "progress":
                    self.update_progress(item[1], item[2] if len(item) > 2 else None)
                elif kind == "export_done":
                    self.update_progress(100, "Completed")
                    self.set_busy(False)
                    self.update_log(item[1])
                    messagebox.showinfo("Export complete", item[2])
                elif kind == "mix_preview_done":
                    self.update_progress(100, "Completed")
                    self.set_busy(False)
                    self.update_log(item[1])
                    ok, msg = play_audio_file(self.temp_mix_preview_path)
                    self.update_log(msg)
                    if not ok:
                        messagebox.showwarning("Playback unavailable", msg)
                elif kind == "record_done":
                    self.update_progress(100, "Completed")
                    self.set_busy(False)
                    self.load_mantra_file(item[1])
                elif kind == "error":
                    self.set_busy(False)
                    self.update_log(f"[Error] {item[1]}")
                    messagebox.showerror("Error", item[1])
        except queue.Empty:
            pass
        self.root.after(120, self.poll_status_queue)

    def apply_selected_preset(self, event=None):
        preset = PRESET_MAP[self.preset_var.get()]
        self.active_preset = preset
        self.word_entry.delete(0, tk.END)
        self.word_entry.insert(0, preset["mantra"])
        self.freq_var.set(str(preset["carrier_hz"]))
        self.background_mode.set(preset["background_mode"])
        self.beat_hz_var.set(str(int(preset["binaural_beat_hz"])))
        self.duration_var.set(str(int(preset["session_minutes"])))
        self.mantra_gain_scale.set(preset["mantra_gain"])
        self.tone_gain_scale.set(preset["tone_gain"])
        self.headphone_var.set(bool(preset["headphones_required"]))
        self.preset_notes_var.set(preset["notes"])
        self.update_log(f"Preset loaded: {preset['name']}")

    def convert_text_to_frequency(self):
        raw_word = self.word_entry.get().strip()
        if not raw_word:
            messagebox.showerror("Input required", "Enter text in the keyword field first.")
            return
        try:
            count = int(self.count_entry.get().strip())
            if count <= 0:
                raise ValueError
        except ValueError:
            count = 108
        string_hash = sum(ord(char) * (idx + 1) for idx, char in enumerate(raw_word))
        hz = int(np.clip((140 + (string_hash % 560)) * (1.0 + 0.075 * np.log1p(count)), MIN_FREQ, MAX_FREQ))
        self.freq_var.set(str(hz))
        self.update_log(f"Suggested frequency for '{raw_word}': {hz} Hz (creative mapping)")

    def on_slider_dragged(self, value):
        self.is_updating_from_slider = True
        self.freq_var.set(str(int(float(value))))
        self.is_updating_from_slider = False

    def on_text_box_modified(self, *_):
        if self.is_updating_from_slider:
            return
        try:
            val = float(self.freq_var.get())
            if MIN_FREQ <= val <= MAX_FREQ:
                self.freq_slider.set(val)
        except ValueError:
            pass

    def validate_inputs(self):
        raw_freq = self.freq_var.get().strip()
        raw_count = self.count_entry.get().strip()
        raw_word = self.word_entry.get().strip() or "OM"
        try:
            frequency = float(raw_freq)
            if not (MIN_FREQ <= frequency <= MAX_FREQ):
                raise ValueError
        except Exception:
            raise ValueError(f"Frequency must be between {MIN_FREQ} and {MAX_FREQ} Hz.")
        try:
            count = int(raw_count)
            if count <= 0:
                raise ValueError
        except Exception:
            raise ValueError("Repetition count must be a positive whole number.")
        return raw_word, frequency, count

    def generate_harmonic_wave(self, target_frequency, duration, sampling_frequency=SAMPLE_RATE, amplitude=0.35):
        t = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)
        signal = np.sin(2*np.pi*target_frequency*t) + 0.35*np.sin(2*np.pi*target_frequency*2*t) + 0.12*np.sin(2*np.pi*target_frequency*3*t)
        return apply_fade((signal / np.max(np.abs(signal)) * amplitude), sample_rate=sampling_frequency, fade_ms=40)

    def execute_acoustics_pipeline(self):
        if self.busy:
            return
        try:
            keyword, frequency, count = self.validate_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self.set_busy(True, "Computing preview and pattern...")
        threading.Thread(target=self._compute_worker, args=(keyword, frequency, count), daemon=True).start()

    def _compute_worker(self, keyword, frequency, count):
        try:
            self.status_queue.put(("progress", 10, "Preparing preview tone..."))
            amplitude = float(min(0.92, 0.14 + 0.11 * np.log1p(count)))
            text_fingerprint = sum(ord(c) for c in keyword) if keyword else 1
            current_n = int(2 + ((int(frequency) + text_fingerprint) % 5) + np.floor(np.log1p(count)))
            current_m = int(3 + ((int(frequency * 3) + text_fingerprint) % 6) + np.floor(0.5 * np.log1p(count)))
            write_wav(self.temp_audio_path, self.generate_harmonic_wave(frequency, PREVIEW_SECONDS, amplitude=amplitude), SAMPLE_RATE)
            self.status_queue.put(("progress", 45, "Building cymatics matrix..."))
            res = 360
            x = np.linspace(-1, 1, res)
            y = np.linspace(-1, 1, res)
            X, Y = np.meshgrid(x, y)
            Z = np.cos(current_n*np.pi*X)*np.cos(current_m*np.pi*Y) - np.cos(current_m*np.pi*X)*np.cos(current_n*np.pi*Y)
            density = np.exp(-np.abs(Z) / 0.035)
            self.status_queue.put(("progress", 80, "Rendering waveform snapshot..."))
            t_snapshot = np.linspace(0, 0.012, 600)
            scope_signal = amplitude * (np.sin(2*np.pi*frequency*t_snapshot) + 0.35*np.sin(2*np.pi*frequency*2*t_snapshot) + 0.12*np.sin(2*np.pi*frequency*3*t_snapshot))
            self.status_queue.put(("progress", 100, "Compute finished."))
            self.status_queue.put(("ready", keyword, frequency, count, amplitude, current_n, current_m, density, t_snapshot, scope_signal))
        except Exception as exc:
            self.status_queue.put(("error", str(exc)))

    def _apply_compute_result(self, keyword, frequency, count, amplitude, current_n, current_m, density, t_snapshot, scope_signal):
        self.current_keyword = keyword
        self.target_frequency = frequency
        self.last_count = count
        self.energy_gain_amplitude = amplitude
        self.current_n = current_n
        self.current_m = current_m
        self.current_density_matrix = density
        self.axes[0].clear()
        self.axes[0].imshow(density, cmap="viridis", extent=[-1,1,-1,1], origin="lower")
        self.axes[0].set_title(f"Pattern modes ({current_n}, {current_m})", color="#2563EB", fontsize=10, fontweight="bold", pad=10)
        self.axes[0].axis("off")
        self.axes[1].clear()
        self.axes[1].plot(t_snapshot*1000, scope_signal, color="#16A34A", linewidth=1.5)
        self.axes[1].set_title(f"Waveform: {frequency:.1f} Hz", color="#2563EB", fontsize=10, fontweight="bold", pad=10)
        self.axes[1].set_xlabel("Time (ms)", color="#5B6570", fontsize=8)
        self.axes[1].set_ylabel("Amplitude", color="#5B6570", fontsize=8)
        self.axes[1].set_facecolor("#FBFCFD")
        self.axes[1].tick_params(colors="#5B6570", labelsize=8)
        self.axes[1].grid(True, linestyle=":", color="#D1D9E0", alpha=0.85)
        self.axes[1].set_ylim(-1.1, 1.1)
        self.fig.tight_layout()
        self.canvas.draw()
        self.update_log(f"Profile: '{keyword}' | {frequency:.2f} Hz | count {count}")
        self.update_log(f"Pattern mode indices: n={current_n}, m={current_m}")
        self.update_log(f"Tone preview ready: {self.temp_audio_path}")
        self.set_busy(False)

    def audio_playback_trigger(self):
        if not os.path.exists(self.temp_audio_path):
            messagebox.showwarning("No preview", "Generate the tone preview first.")
            return
        ok, msg = play_audio_file(self.temp_audio_path)
        self.update_log(msg)
        if not ok:
            messagebox.showwarning("Playback unavailable", msg)

    def load_mantra_wav(self):
        path = filedialog.askopenfilename(filetypes=[("Wave Audio Files", "*.wav")])
        if path:
            self.load_mantra_file(path)

    def load_mantra_file(self, path):
        try:
            rate, data = wavfile.read(path)
            data = data.astype(np.float32) / np.iinfo(data.dtype).max if data.dtype.kind in "iu" else data.astype(np.float32)
            data = normalize_audio(apply_fade(mono_mixdown(resample_audio(data, rate, SAMPLE_RATE)), SAMPLE_RATE, 35), peak=0.95)
            self.mantra_audio = data
            self.mantra_sample_rate = SAMPLE_RATE
            self.mantra_source_path = path
            self.mantra_status_var.set(f"Loaded mantra WAV: {os.path.basename(path)}")
            self.update_log(f"Loaded mantra WAV: {path}")
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not load WAV file.\n{exc}")

    def clear_mantra(self):
        self.mantra_audio = None
        self.mantra_source_path = None
        self.mantra_status_var.set("No mantra loaded. Tone-only export will be used.")
        self.update_log("Cleared mantra audio.")

    def record_mantra_dialog(self):
        if not SOUNDDEVICE_AVAILABLE:
            messagebox.showwarning("Recording unavailable", "Microphone recording requires the 'sounddevice' package and PortAudio. Install with: python -m pip install sounddevice")
            return
        if self.busy:
            return
        win = tk.Toplevel(self.root)
        win.title("Record mantra")
        win.geometry("300x160")
        win.transient(self.root)
        win.grab_set()
        tk.Label(win, text="Recording duration (seconds)", font=("Segoe UI", 10)).pack(pady=(16, 8))
        duration_var = tk.StringVar(value="10")
        tk.Entry(win, textvariable=duration_var, font=("Segoe UI", 11)).pack(pady=(0, 12))

        def start_recording():
            try:
                duration = float(duration_var.get())
                if not (0 < duration <= 120):
                    raise ValueError
            except Exception:
                messagebox.showerror("Invalid duration", "Enter a duration between 1 and 120 seconds.", parent=win)
                return
            win.destroy()
            self.set_busy(True, f"Recording microphone for {duration:.1f} seconds...")
            threading.Thread(target=self._record_worker, args=(duration,), daemon=True).start()

        tk.Button(win, text="Start recording", bg="#16A34A", fg="#FFFFFF", font=("Segoe UI", 10, "bold"), bd=0, command=start_recording).pack(ipadx=10, ipady=6)

    def _record_worker(self, duration):
        try:
            self.status_queue.put(("progress", 5, "Starting microphone recording..."))
            recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            steps = max(1, int(duration * 10))
            for i in range(steps):
                if sd.wait(int(100)):
                    pass
                self.status_queue.put(("progress", 5 + (85 * (i + 1) / steps), f"Recording... {int((i + 1) / steps * 100)}%"))
            sd.wait()
            self.status_queue.put(("progress", 94, "Finalizing recording..."))
            data = normalize_audio(apply_fade(recording[:, 0], SAMPLE_RATE, 35), peak=0.95)
            write_wav(self.recorded_temp_path, data, SAMPLE_RATE)
            self.status_queue.put(("progress", 100, "Recording complete."))
            self.status_queue.put(("record_done", self.recorded_temp_path))
        except Exception as exc:
            self.status_queue.put(("error", f"Recording failed: {exc}"))

    def get_mix_settings(self):
        try:
            duration_seconds = int(self.duration_var.get()) * 60
        except ValueError:
            duration_seconds = 20 * 60
        try:
            beat_hz = float(self.beat_hz_var.get())
        except ValueError:
            beat_hz = 6.0
        return duration_seconds, self.background_mode.get().strip().lower(), beat_hz, float(self.mantra_gain_scale.get()), float(self.tone_gain_scale.get())

    def build_full_mix(self, duration_seconds, mode, beat_hz, mantra_gain, tone_gain):
        total_samples = int(duration_seconds * SAMPLE_RATE)
        mantra = loop_audio_to_length(self.mantra_audio, total_samples, SAMPLE_RATE, 35) if self.mantra_audio is not None and len(self.mantra_audio) > 0 else np.zeros(total_samples, dtype=np.float32)
        mantra_stereo = ensure_stereo(normalize_audio(mantra, peak=1.0) * mantra_gain)
        if mode == "binaural":
            bg = generate_binaural(carrier=self.target_frequency, beat_hz=beat_hz, duration=duration_seconds, sample_rate=SAMPLE_RATE, amp=tone_gain)
            if not self.headphone_var.get():
                self.update_log("Warning: binaural mode is intended for stereo headphone listening.")
        elif mode == "mono":
            bg = ensure_stereo(apply_fade(generate_sine(self.target_frequency, duration_seconds, SAMPLE_RATE, tone_gain), SAMPLE_RATE, 2000))
        else:
            bg = np.zeros((total_samples, 2), dtype=np.float32)
        return normalize_audio(apply_fade(mantra_stereo + bg, SAMPLE_RATE, 3000), peak=0.92)

    def preview_mix(self):
        if self.busy:
            return
        try:
            _, frequency, _ = self.validate_inputs()
            self.target_frequency = frequency
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self.set_busy(True, "Rendering short mantra + tone preview...")
        threading.Thread(target=self._preview_mix_worker, daemon=True).start()

    def _preview_mix_worker(self):
        try:
            self.status_queue.put(("progress", 10, "Preparing preview mix..."))
            duration_seconds, mode, beat_hz, mantra_gain, tone_gain = self.get_mix_settings()
            preview_seconds = min(self.active_preset.get("preview_seconds", 20), duration_seconds)
            self.status_queue.put(("progress", 55, "Mixing audio layers..."))
            write_wav(self.temp_mix_preview_path, self.build_full_mix(preview_seconds, mode, beat_hz, mantra_gain, tone_gain), SAMPLE_RATE)
            src = "mantra + background" if self.mantra_audio is not None else "background only"
            self.status_queue.put(("progress", 100, "Preview ready."))
            self.status_queue.put(("mix_preview_done", f"Prepared {preview_seconds}-second preview mix ({src})."))
        except Exception as exc:
            self.status_queue.put(("error", str(exc)))

    def export_audio_file(self):
        if self.busy:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Wave Audio Files", "*.wav")])
        if not save_path:
            return
        self.set_busy(True, "Rendering full session WAV export...")
        threading.Thread(target=self._export_audio_worker, args=(save_path,), daemon=True).start()

    def _export_audio_worker(self, save_path):
        try:
            duration_seconds, mode, beat_hz, mantra_gain, tone_gain = self.get_mix_settings()
            self.status_queue.put(("progress", 8, "Preparing session export..."))
            mix = self.build_full_mix(duration_seconds, mode, beat_hz, mantra_gain, tone_gain)
            self.status_queue.put(("progress", 78, "Writing WAV file..."))
            write_wav(save_path, mix, SAMPLE_RATE)
            self.status_queue.put(("progress", 100, "Export complete."))
            self.status_queue.put(("export_done", f"Saved session WAV -> {os.path.basename(save_path)}", "The full mantra + tone session has been generated successfully."))
        except Exception as exc:
            self.status_queue.put(("error", str(exc)))

    def export_five_minute_audio(self):
        if self.busy:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Wave Audio Files", "*.wav")])
        if not save_path:
            return
        self.set_busy(True, "Rendering 5-minute preset WAV...")
        threading.Thread(target=self._export_five_minute_worker, args=(save_path,), daemon=True).start()

    def _export_five_minute_worker(self, save_path):
        try:
            _, mode, beat_hz, mantra_gain, tone_gain = self.get_mix_settings()
            self.status_queue.put(("progress", 10, "Preparing 5-minute mix..."))
            mix = self.build_full_mix(300, mode, beat_hz, mantra_gain, tone_gain)
            self.status_queue.put(("progress", 80, "Writing 5-minute WAV file..."))
            write_wav(save_path, mix, SAMPLE_RATE)
            self.status_queue.put(("progress", 100, "5-minute export complete."))
            self.status_queue.put(("export_done", f"Saved 5-minute preset WAV -> {os.path.basename(save_path)}", "The 5-minute mantra + tone audio has been generated successfully."))
        except Exception as exc:
            self.status_queue.put(("error", str(exc)))

    def export_cymatics_plot(self):
        if self.current_density_matrix is None:
            messagebox.showwarning("No image", "Compute a pattern before exporting the PNG.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image Files", "*.png")])
        if not save_path:
            return
        try:
            fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
            ax.imshow(self.current_density_matrix, cmap="viridis", extent=[-1,1,-1,1], origin="lower")
            ax.axis("off")
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            self.update_log(f"Saved PNG export -> {os.path.basename(save_path)}")
            messagebox.showinfo("Export complete", "Pattern image saved successfully.")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def on_close(self):
        stop_all_audio()
        safe_remove(self.temp_audio_path)
        safe_remove(self.temp_mix_preview_path)
        safe_remove(self.recorded_temp_path)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImprovedCymaticsDashboard(root)
    root.mainloop()

