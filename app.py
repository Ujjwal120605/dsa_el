"""
Emotion-Aware Voice Bot — Streamlit Dashboard
================================================
RV College of Engineering | EC355TDD
Authors: Ujjwal Bajpai (1RV23EC175) & Pranshul Bhargava (1RV23EC101)

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import librosa
import pickle
import json
import os
import io
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import soundfile as sf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion-Aware Voice Bot",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTIONS = ["neutral", "happy", "sad", "angry"]
EMOTION_EMOJI = {
    "neutral": "😐",
    "happy":   "😄",
    "sad":     "😢",
    "angry":   "😠",
}
EMOTION_COLOR = {
    "neutral": "#8ecae6",
    "happy":   "#f9c74f",
    "sad":     "#577590",
    "angry":   "#f94144",
}
EMPATHETIC_RESPONSES = {
    "neutral": [
        "I hear you — things seem steady right now.",
        "You sound calm and collected.",
        "Let me know how I can assist you today.",
    ],
    "happy": [
        "That's wonderful! Your positivity is contagious! 🌟",
        "Love the energy! What exciting things are happening?",
        "You sound absolutely delightful today!",
    ],
    "sad": [
        "I'm really sorry you're feeling down. 💙 I'm here for you.",
        "It's okay to feel sad sometimes. Would you like to talk about it?",
        "Take it one step at a time. You've got this.",
    ],
    "angry": [
        "I understand you're frustrated. Let's work through this together.",
        "It's okay to feel angry. Take a breath — I'm listening.",
        "Your feelings are valid. How can I help resolve this?",
    ],
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0d1117; }

.stApp { background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%); }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f2027 50%, #16213e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 70% 30%, rgba(56,139,253,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5d6ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.hero-sub {
    color: #8b949e; font-size: 0.95rem; margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
}

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-label { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #e6edf3; font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-delta { font-size: 0.85rem; }

/* Emotion result */
.emotion-result {
    background: #161b22;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    border: 2px solid #30363d;
    margin: 16px 0;
    transition: all 0.3s;
}
.emotion-emoji { font-size: 4rem; display: block; margin-bottom: 8px; }
.emotion-label { font-size: 1.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; }
.emotion-conf { color: #8b949e; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; margin-top: 4px; }

/* Response box */
.response-box {
    background: linear-gradient(135deg, #1c2333, #21262d);
    border-left: 4px solid #58a6ff;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    color: #e6edf3;
    font-size: 1.05rem;
    line-height: 1.6;
    margin: 16px 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Pills / badges */
.badge {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #8b949e;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* Section header */
.section-header {
    color: #e6edf3;
    font-size: 1.1rem;
    font-weight: 600;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin: 20px 0 16px 0;
}

/* History item */
.history-item {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

.stAlert { border-radius: 8px !important; }

/* ── Live recording UI ─────────────────────────────── */
.live-recorder {
    background: linear-gradient(135deg, #161b22 0%, #0f1923 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
@keyframes pulse-ring {
    0%   { transform: scale(1);    opacity: 0.8; }
    50%  { transform: scale(1.18); opacity: 0.3; }
    100% { transform: scale(1);    opacity: 0.8; }
}
@keyframes pulse-ring2 {
    0%   { transform: scale(1);    opacity: 0.5; }
    50%  { transform: scale(1.35); opacity: 0.1; }
    100% { transform: scale(1);    opacity: 0.5; }
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open(os.path.join(MODEL_DIR, "pipeline.pkl"), "rb") as f:
            pipe = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
            meta = json.load(f)
        cm = np.load(os.path.join(MODEL_DIR, "confusion_matrix.npy"))
        return pipe, le, meta, cm
    except FileNotFoundError:
        return None, None, None, None


# ── Feature extraction (must match train_model.py) ────────────────────────────
def extract_features(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    feats = []
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    d1   = librosa.feature.delta(mfcc, order=1)
    d2   = librosa.feature.delta(mfcc, order=2)
    for mat in (mfcc, d1, d2):
        feats.extend(np.mean(mat, axis=1).tolist())
        feats.extend(np.std(mat,  axis=1).tolist())
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    feats.extend(np.mean(chroma, axis=1).tolist())
    feats.extend(np.std(chroma,  axis=1).tolist())
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    feats.append(float(np.mean(mel)))
    feats.append(float(np.std(mel)))
    sc = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    feats.extend(np.mean(sc, axis=1).tolist())
    feats.extend(np.std(sc,  axis=1).tolist())
    zcr = librosa.feature.zero_crossing_rate(audio)
    feats.append(float(np.mean(zcr)))
    feats.append(float(np.std(zcr)))
    rms = librosa.feature.rms(y=audio)
    feats.append(float(np.mean(rms)))
    feats.append(float(np.std(rms)))
    feats.append(float(np.max(rms)))
    for fn in (
        librosa.feature.spectral_centroid,
        librosa.feature.spectral_bandwidth,
        librosa.feature.spectral_rolloff,
    ):
        arr = fn(y=audio, sr=sr)
        feats.append(float(np.mean(arr)))
        feats.append(float(np.std(arr)))
    flat = librosa.feature.spectral_flatness(y=audio)
    feats.append(float(np.mean(flat)))
    feats.append(float(np.std(flat)))
    try:
        harm = librosa.effects.harmonic(audio)
        tonn = librosa.feature.tonnetz(y=harm, sr=sr)
        feats.extend(np.mean(tonn, axis=1).tolist())
        feats.extend(np.std(tonn,  axis=1).tolist())
    except Exception:
        feats.extend([0.0] * 12)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        if f0 is None:
            raise ValueError
        f0_clean = f0[~np.isnan(f0)]
        voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
        if len(f0_clean) > 0:
            feats.append(float(np.mean(f0_clean)))
            feats.append(float(np.std(f0_clean)))
            feats.append(float(np.ptp(f0_clean)))
        else:
            feats.extend([0.0, 0.0, 0.0])
        feats.append(voiced_ratio)
    except Exception:
        feats.extend([0.0, 0.0, 0.0, 0.0])
    try:
        f0_jitter, _, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            frame_length=512,
        )
        f0j = f0_jitter[~np.isnan(f0_jitter)] if f0_jitter is not None else np.array([0.0])
        jitter = float(np.mean(np.abs(np.diff(f0j)) / (np.mean(f0j) + 1e-9))) if len(f0j) > 1 else 0.0
        feats.append(jitter)
    except Exception:
        feats.append(0.0)
    rms_arr = librosa.feature.rms(y=audio)[0]
    shimmer = float(np.mean(np.abs(np.diff(rms_arr)) / (np.mean(rms_arr) + 1e-9)))
    feats.append(shimmer)
    try:
        stft = np.abs(librosa.stft(audio))
        pcen = librosa.pcen(stft * (2 ** 31), sr=sr)
        feats.append(float(np.mean(pcen)))
        feats.append(float(np.std(pcen)))
    except Exception:
        feats.extend([0.0, 0.0])
    bands = [(85, 255), (255, 900), (900, 2800), (2800, 5000)]
    stft_mag = np.abs(librosa.stft(audio, n_fft=2048))
    freqs    = librosa.fft_frequencies(sr=sr, n_fft=2048)
    for flo, fhi in bands:
        idx = np.where((freqs >= flo) & (freqs < fhi))[0]
        feats.append(float(np.mean(stft_mag[idx, :])) if len(idx) > 0 else 0.0)
    try:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        feats.append(float(tempo))
        feats.append(float(np.std(np.diff(beats))) if len(beats) > 1 else 0.0)
    except Exception:
        feats.extend([0.0, 0.0])
    return np.array(feats, dtype=np.float32)


def predict_emotion(audio: np.ndarray, sr: int, pipe, le):
    features = extract_features(audio, sr=sr)
    proba    = pipe.predict_proba(features.reshape(1, -1))[0]
    idx      = np.argmax(proba)
    label    = le.inverse_transform([idx])[0]
    conf     = float(proba[idx])
    all_probs = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
    return label, conf, all_probs


# ── Waveform / spectrogram plots ──────────────────────────────────────────────
def plot_waveform(audio, sr, emotion_color="#58a6ff"):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#0d1117")
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.fill_between(t, audio, alpha=0.6, color=emotion_color)
    ax.plot(t, audio, color=emotion_color, linewidth=0.8, alpha=0.9)
    ax.set_xlabel("Time (s)", color="#8b949e", fontsize=9)
    ax.set_ylabel("Amplitude", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    ax.set_title("Waveform", color="#e6edf3", fontsize=10, pad=8)
    plt.tight_layout(); return fig


def plot_spectrogram(audio, sr):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#0d1117")
    D    = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    cmap = LinearSegmentedColormap.from_list("emo", ["#0d1117", "#1f6feb", "#58a6ff", "#f9c74f"])
    img  = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap=cmap)
    fig.colorbar(img, ax=ax, format="%+2.0f dB").ax.tick_params(colors="#8b949e", labelsize=7)
    ax.set_xlabel("Time (s)", color="#8b949e", fontsize=9)
    ax.set_ylabel("Hz", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    ax.set_title("Spectrogram", color="#e6edf3", fontsize=10, pad=8)
    plt.tight_layout(); return fig


def plot_mfcc(audio, sr):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#0d1117")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    cmap = LinearSegmentedColormap.from_list("mfcc", ["#0d1117", "#1c2333", "#58a6ff"])
    librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax, cmap=cmap)
    ax.set_ylabel("MFCC", color="#8b949e", fontsize=9)
    ax.set_xlabel("Time (s)", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    ax.set_title("MFCC Coefficients", color="#e6edf3", fontsize=10, pad=8)
    plt.tight_layout(); return fig


def plot_probability_bars(probs: dict):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#161b22")
    labels = list(probs.keys())
    values = [probs[l] * 100 for l in labels]
    colors = [EMOTION_COLOR.get(l, "#58a6ff") for l in labels]
    bars   = ax.barh(labels, values, color=colors, height=0.55, edgecolor="none")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    for bar, val in zip(bars, values):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="#e6edf3", fontsize=8, fontfamily="monospace")
    ax.set_title("Emotion Probabilities", color="#e6edf3", fontsize=10, pad=8)
    plt.tight_layout(); return fig


def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#161b22")
    cmap = LinearSegmentedColormap.from_list("cm", ["#0d1117", "#1f6feb", "#58a6ff"])
    im   = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax).ax.tick_params(colors="#8b949e", labelsize=8)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel="True label", xlabel="Predicted label")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.xaxis.label.set_color("#8b949e"); ax.yaxis.label.set_color("#8b949e")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "#0d1117", fontsize=10)
    ax.set_title("Confusion Matrix (Test Set)", color="#e6edf3", fontsize=11, pad=10)
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    plt.tight_layout(); return fig


# ── Demo audio generator ──────────────────────────────────────────────────────
def gen_demo_audio(emotion: str, sr: int = 22050, dur: float = 2.5) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    params = dict(
        neutral=(220, .05, .5, 0),
        happy=  (400, .04, .7, 6),
        sad=    (150, .02, .3, 1),
        angry=  (300, .20, .9, 3),
    )
    freq, noise, amp, vib = params.get(emotion, params["neutral"])
    a = amp * np.sin(2 * np.pi * freq * t)
    a += 0.3 * amp * np.sin(2 * np.pi * freq * 2 * t)
    if vib:
        a += 0.1 * amp * np.sin(2 * np.pi * (freq + vib * np.sin(2 * np.pi * 5 * t)) * t)
    a += noise * np.random.randn(len(t))
    a /= np.max(np.abs(a) + 1e-8)
    return a.astype(np.float32)


def audio_to_bytes(audio: np.ndarray, sr: int = 22050) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


# ── Session state ─────────────────────────────────────────────────────────────
if "history"         not in st.session_state: st.session_state.history         = []
if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
if "live_result"     not in st.session_state: st.session_state.live_result     = None


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0;'>
        <div style='font-size:2.5rem'>🎙️</div>
        <div style='color:#e6edf3; font-weight:700; font-size:1.1rem; margin-top:4px'>Voice Bot</div>
        <div style='color:#8b949e; font-size:0.75rem; font-family:monospace'>EC355TDD | RVCE</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    pipe, le, meta, cm_data = load_model()

    if meta:
        acc_key = meta.get("accuracy", meta.get("hold_out_accuracy", 0))
        st.markdown("**🤖 Model Info**")
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Accuracy</div>
            <div class='metric-value' style='color:#3fb950'>{acc_key*100:.1f}%</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Type</div>
            <div style='color:#e6edf3; font-size:0.85rem; font-family:monospace'>{meta['model_type']}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Features</div>
            <div class='metric-value' style='font-size:1.4rem'>{meta['feature_dim']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='margin:8px 0'>
        {''.join(f"<span class='badge'>{EMOTION_EMOJI[e]} {e}</span>" for e in EMOTIONS)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model not found. Run `train_model.py` first.")

    st.divider()

    st.markdown("**📜 Session History**")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-8:]):
            st.markdown(f"""
            <div class='history-item'>
                <span style='font-size:1.4rem'>{EMOTION_EMOJI[item['emotion']]}</span>
                <div>
                    <div style='color:#e6edf3; font-size:0.85rem; font-weight:600'>{item['emotion'].title()}</div>
                    <div style='color:#8b949e; font-size:0.75rem; font-family:monospace'>{item['conf']:.0%} · {item['source']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("<div style='color:#8b949e; font-size:0.85rem'>No analyses yet.</div>",
                    unsafe_allow_html=True)


# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🎙️ Emotion-Aware Voice Bot</div>
    <div class='hero-sub'>Deep Learning · Speech Emotion Recognition · Real-Time Analysis</div>
    <div style='margin-top:12px'>
        <span class='badge'>EC355TDD</span>
        <span class='badge'>Ujjwal Bajpai · 1RV23EC175</span>
        <span class='badge'>Pranshul Bhargava · 1RV23EC101</span>
        <span class='badge'>RV College of Engineering</span>
    </div>
</div>
""", unsafe_allow_html=True)

if pipe is None:
    st.error("⚠️ Model not loaded. Please run `python train_model.py` first.")
    st.stop()

tab1, tab_live, tab2, tab3, tab4 = st.tabs([
    "🎤 Analyze Audio", "🔴 Live Record", "🎭 Demo Mode", "📊 Model Analytics", "ℹ️ About"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Analyze Audio
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.markdown("<div class='section-header'>Upload Voice Recording</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#8b949e; font-size:0.9rem; margin-bottom:16px'>
        Upload a <b>.wav</b>, <b>.mp3</b>, or <b>.ogg</b> audio file containing speech.
        The model will analyze vocal features to detect your emotional state.
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop audio file here",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            label_visibility="collapsed"
        )

        if uploaded:
            with st.spinner("🔬 Extracting audio features..."):
                try:
                    audio_bytes = uploaded.read()
                    audio, sr   = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
                    dur         = len(audio) / sr

                    st.audio(audio_bytes, format="audio/" + uploaded.name.split(".")[-1])
                    st.markdown(f"""
                    <div style='display:flex; gap:12px; flex-wrap:wrap; margin:8px 0'>
                        <span class='badge'>⏱ {dur:.1f}s</span>
                        <span class='badge'>🎵 {sr} Hz</span>
                        <span class='badge'>📁 {uploaded.name}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    t0 = time.time()
                    emotion, conf, all_probs = predict_emotion(audio, sr, pipe, le)
                    inference_ms = (time.time() - t0) * 1000

                    st.session_state.analysis_result = {
                        "audio": audio, "sr": sr,
                        "emotion": emotion, "conf": conf,
                        "probs": all_probs, "ms": inference_ms,
                    }
                    st.session_state.history.append({
                        "emotion": emotion, "conf": conf, "source": uploaded.name[:20]
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing file: {e}")

    with col_right:
        if st.session_state.analysis_result:
            r     = st.session_state.analysis_result
            emo   = r["emotion"]
            color = EMOTION_COLOR[emo]

            st.markdown(f"""
            <div class='emotion-result' style='border-color:{color}'>
                <span class='emotion-emoji'>{EMOTION_EMOJI[emo]}</span>
                <div class='emotion-label' style='color:{color}'>{emo}</div>
                <div class='emotion-conf'>Confidence: {r['conf']:.1%} · {r['ms']:.0f}ms inference</div>
            </div>
            """, unsafe_allow_html=True)

            import random
            response = random.choice(EMPATHETIC_RESPONSES[emo])
            st.markdown(f"""
            <div class='response-box' style='border-left-color:{color}'>
                <div style='color:#8b949e; font-size:0.75rem; margin-bottom:6px; font-family:monospace'>BOT RESPONSE</div>
                {response}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Probability Distribution**")
            fig_bars = plot_probability_bars(r["probs"])
            st.pyplot(fig_bars, use_container_width=True)
            plt.close()
        else:
            st.markdown("""
            <div style='height:300px; display:flex; align-items:center; justify-content:center;
                 flex-direction:column; color:#8b949e; border:1px dashed #30363d; border-radius:12px'>
                <div style='font-size:3rem'>🎤</div>
                <div style='margin-top:12px'>Upload an audio file to begin analysis</div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.analysis_result:
        r     = st.session_state.analysis_result
        color = EMOTION_COLOR[r["emotion"]]
        st.divider()
        st.markdown("<div class='section-header'>Audio Visualizations</div>", unsafe_allow_html=True)
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.pyplot(plot_waveform(r["audio"], r["sr"], color), use_container_width=True); plt.close()
        with vc2:
            st.pyplot(plot_spectrogram(r["audio"], r["sr"]), use_container_width=True); plt.close()
        with vc3:
            st.pyplot(plot_mfcc(r["audio"], r["sr"]), use_container_width=True); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB LIVE — Live Voice Recording
# ─────────────────────────────────────────────────────────────────────────────
with tab_live:
    import random as _random

    st.markdown("""
    <div style='color:#8b949e; font-size:0.95rem; margin-bottom:20px'>
    Record your voice directly from the browser. Click <b>Start Recording</b>, speak for a few seconds,
    then click <b>Stop & Analyze</b> — the bot will detect your emotion in real time.
    </div>
    """, unsafe_allow_html=True)

    lc1, lc2 = st.columns([1, 1.4], gap="large")

    with lc1:
        st.markdown("<div class='section-header'>🔴 Voice Recorder</div>", unsafe_allow_html=True)

        recorder_html = """
        <style>
          body { margin: 0; background: transparent; font-family: 'Space Grotesk', sans-serif; }
          .wrap {
            display: flex; flex-direction: column; align-items: center;
            gap: 16px; padding: 24px 16px;
            background: linear-gradient(135deg, #161b22, #0f1923);
            border: 1px solid #30363d; border-radius: 16px;
          }
          .mic-outer {
            width: 110px; height: 110px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            position: relative; cursor: pointer;
            background: linear-gradient(135deg, #1f6feb, #388bfd);
            transition: transform 0.2s, background 0.3s;
            box-shadow: 0 4px 24px rgba(56,139,253,0.3);
            border: none; outline: none;
          }
          .mic-outer:hover { transform: scale(1.06); }
          .mic-outer.recording {
            background: linear-gradient(135deg, #da3633, #f85149);
            box-shadow: 0 4px 24px rgba(248,81,73,0.4);
            animation: pulse 1.4s ease-in-out infinite;
          }
          @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.07); } }
          .mic-outer.recording::after {
            content: '';
            position: absolute; inset: -12px; border-radius: 50%;
            border: 2px solid rgba(248,81,73,0.35);
            animation: ring 1.4s ease-in-out infinite;
          }
          @keyframes ring { 0%,100% { opacity: 0.7; transform: scale(1); } 50% { opacity: 0.1; transform: scale(1.25); } }
          .mic-icon { font-size: 2.8rem; user-select: none; }
          .timer { font-family: monospace; font-size: 2.2rem; font-weight: 700; color: #e6edf3; min-height: 2.8rem; letter-spacing: 2px; }
          .timer.active { color: #f85149; }
          .status { font-family: monospace; font-size: 0.8rem; color: #8b949e; min-height: 1.2rem; text-align: center; }
          .btn-row { display: flex; gap: 10px; }
          .btn { padding: 10px 22px; border-radius: 8px; border: none; font-weight: 600; font-size: 0.85rem; cursor: pointer; transition: opacity 0.2s; }
          .btn:disabled { opacity: 0.3; cursor: default; }
          .btn-start { background: linear-gradient(135deg,#1f6feb,#388bfd); color: #fff; }
          .btn-stop  { background: linear-gradient(135deg,#da3633,#f85149); color: #fff; }
          .btn-reset { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
          .wave-canvas { width: 100%; height: 60px; background: #0d1117; border-radius: 8px; border: 1px solid #21262d; }
          .tip { font-size: 0.72rem; color: #484f58; text-align: center; font-family: monospace; }
        </style>
        <div class="wrap">
          <button class="mic-outer" id="micBtn" onclick="toggleRecord()">
            <span class="mic-icon" id="micIcon">🎙️</span>
          </button>
          <div class="timer" id="timer">00:00</div>
          <div class="status" id="status">Click the mic to start recording</div>
          <canvas class="wave-canvas" id="waveCanvas" width="320" height="60"></canvas>
          <div class="btn-row">
            <button class="btn btn-start" id="btnStart" onclick="startRec()">▶ Start</button>
            <button class="btn btn-stop"  id="btnStop"  onclick="stopRec()" disabled>■ Stop & Analyze</button>
            <button class="btn btn-reset" id="btnReset" onclick="resetRec()" disabled>↺ Reset</button>
          </div>
          <div class="tip">Speak clearly for 2–5 seconds · WAV · 22kHz mono</div>
        </div>
        <script>
        let mediaRecorder, audioChunks = [], timerInterval, startTime;
        let stream, analyser, animFrameId, isRecording = false;
        const micBtn=document.getElementById('micBtn'),micIcon=document.getElementById('micIcon'),
              timerEl=document.getElementById('timer'),statusEl=document.getElementById('status'),
              btnStart=document.getElementById('btnStart'),btnStop=document.getElementById('btnStop'),
              btnReset=document.getElementById('btnReset'),canvas=document.getElementById('waveCanvas'),
              ctx=canvas.getContext('2d');
        function toggleRecord(){ if(!isRecording) startRec(); else stopRec(); }
        async function startRec(){
          try{ stream=await navigator.mediaDevices.getUserMedia({audio:true}); }
          catch(e){ statusEl.textContent='❌ Microphone access denied'; return; }
          const audioCtx=new(window.AudioContext||window.webkitAudioContext)();
          const source=audioCtx.createMediaStreamSource(stream);
          analyser=audioCtx.createAnalyser(); analyser.fftSize=256;
          source.connect(analyser); drawWave();
          audioChunks=[]; mediaRecorder=new MediaRecorder(stream);
          mediaRecorder.ondataavailable=e=>audioChunks.push(e.data);
          mediaRecorder.onstop=processAudio; mediaRecorder.start(); isRecording=true;
          micBtn.classList.add('recording'); micIcon.textContent='⏹';
          btnStart.disabled=true; btnStop.disabled=false; btnReset.disabled=true;
          statusEl.textContent='🔴 Recording…'; timerEl.classList.add('active');
          startTime=Date.now();
          timerInterval=setInterval(()=>{
            const s=Math.floor((Date.now()-startTime)/1000),m=Math.floor(s/60);
            timerEl.textContent=String(m).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
          },500);
        }
        function stopRec(){
          if(!isRecording) return;
          mediaRecorder.stop(); stream.getTracks().forEach(t=>t.stop());
          clearInterval(timerInterval); cancelAnimationFrame(animFrameId); isRecording=false;
          micBtn.classList.remove('recording'); micIcon.textContent='⏳';
          btnStop.disabled=true; statusEl.textContent='⚙️ Processing audio…';
        }
        function resetRec(){
          timerEl.textContent='00:00'; timerEl.classList.remove('active');
          statusEl.textContent='Click the mic to start recording'; micIcon.textContent='🎙️';
          btnStart.disabled=false; btnStop.disabled=true; btnReset.disabled=true;
          ctx.clearRect(0,0,canvas.width,canvas.height);
          window.parent.postMessage({type:'streamlit:setComponentValue',value:'RESET'},'*');
        }
        async function processAudio(){
          const blob=new Blob(audioChunks,{type:'audio/webm'});
          const arrayBuf=await blob.arrayBuffer();
          const audioCtx=new(window.AudioContext||window.webkitAudioContext)({sampleRate:22050});
          let decoded;
          try{ decoded=await audioCtx.decodeAudioData(arrayBuf); }
          catch(e){ statusEl.textContent='❌ Could not decode audio.'; micIcon.textContent='🎙️'; btnStart.disabled=false; btnReset.disabled=false; return; }
          const numCh=decoded.numberOfChannels,length=decoded.length,mono=new Float32Array(length);
          for(let ch=0;ch<numCh;ch++){ const d=decoded.getChannelData(ch); for(let i=0;i<length;i++) mono[i]+=d[i]/numCh; }
          const wavBytes=encodeWAV(mono,22050),b64=arrayBufferToBase64(wavBytes);
          statusEl.textContent='✅ Done! See results on the right →';
          micIcon.textContent='🎙️'; timerEl.classList.remove('active');
          btnStart.disabled=false; btnReset.disabled=false;
          window.parent.postMessage({type:'streamlit:setComponentValue',value:b64},'*');
        }
        function encodeWAV(samples,sr){
          const buf=new ArrayBuffer(44+samples.length*2),view=new DataView(buf);
          function writeStr(o,s){ for(let i=0;i<s.length;i++) view.setUint8(o+i,s.charCodeAt(i)); }
          writeStr(0,'RIFF'); view.setUint32(4,36+samples.length*2,true); writeStr(8,'WAVE');
          writeStr(12,'fmt '); view.setUint32(16,16,true); view.setUint16(20,1,true);
          view.setUint16(22,1,true); view.setUint32(24,sr,true); view.setUint32(28,sr*2,true);
          view.setUint16(32,2,true); view.setUint16(34,16,true); writeStr(36,'data');
          view.setUint32(40,samples.length*2,true);
          let offset=44;
          for(let i=0;i<samples.length;i++,offset+=2){
            const s=Math.max(-1,Math.min(1,samples[i]));
            view.setInt16(offset,s<0?s*0x8000:s*0x7FFF,true);
          }
          return buf;
        }
        function arrayBufferToBase64(buf){
          let bin='',bytes=new Uint8Array(buf);
          for(let i=0;i<bytes.byteLength;i++) bin+=String.fromCharCode(bytes[i]);
          return window.btoa(bin);
        }
        function drawWave(){
          animFrameId=requestAnimationFrame(drawWave);
          const bufLen=analyser.frequencyBinCount,data=new Uint8Array(bufLen);
          analyser.getByteTimeDomainData(data);
          ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,canvas.width,canvas.height);
          ctx.lineWidth=2; ctx.strokeStyle='#f85149'; ctx.beginPath();
          const sliceW=canvas.width/bufLen; let x=0;
          for(let i=0;i<bufLen;i++){
            const v=data[i]/128.0,y=(v*canvas.height)/2;
            if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); x+=sliceW;
          }
          ctx.lineTo(canvas.width,canvas.height/2); ctx.stroke();
        }
        </script>
        """

        import streamlit.components.v1 as components
        components.html(recorder_html, height=400, scrolling=False)

    with lc2:
        st.markdown("<div class='section-header'>Analysis Result</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#8b949e; font-size:0.85rem; margin-bottom:12px'>
        Use the recorder on the left <b>or</b> the native mic widget below:
        </div>
        """, unsafe_allow_html=True)

        try:
            wav_audio = st.audio_input("🎤 Click to record", key="live_mic")
        except AttributeError:
            wav_audio = None
            st.info("Upgrade Streamlit (`pip install -U streamlit`) to enable the native mic widget.")

        if wav_audio is not None:
            with st.spinner("🔬 Analyzing your voice..."):
                try:
                    raw   = wav_audio.read()
                    audio, sr = librosa.load(io.BytesIO(raw), sr=22050, mono=True)
                    dur   = len(audio) / sr
                    if dur < 0.5:
                        st.warning("Recording too short — please speak for at least 1 second.")
                    else:
                        st.audio(raw, format="audio/wav")
                        st.markdown(f"""
                        <div style='display:flex; gap:10px; flex-wrap:wrap; margin:6px 0'>
                            <span class='badge'>⏱ {dur:.1f}s</span>
                            <span class='badge'>🎵 {sr} Hz</span>
                            <span class='badge'>🔴 live recording</span>
                        </div>
                        """, unsafe_allow_html=True)
                        t0 = time.time()
                        emotion, conf, all_probs = predict_emotion(audio, sr, pipe, le)
                        inference_ms = (time.time() - t0) * 1000
                        st.session_state.live_result = {
                            "audio": audio, "sr": sr,
                            "emotion": emotion, "conf": conf,
                            "probs": all_probs, "ms": inference_ms,
                        }
                        st.session_state.history.append({
                            "emotion": emotion, "conf": conf, "source": "live mic"
                        })
                except Exception as e:
                    st.error(f"Error processing recording: {e}")

        if st.session_state.live_result:
            r   = st.session_state.live_result
            emo = r["emotion"]
            col = EMOTION_COLOR[emo]
            st.markdown(f"""
            <div class='emotion-result' style='border-color:{col}'>
                <span class='emotion-emoji'>{EMOTION_EMOJI[emo]}</span>
                <div class='emotion-label' style='color:{col}'>{emo}</div>
                <div class='emotion-conf'>Confidence: {r['conf']:.1%} &nbsp;·&nbsp; {r['ms']:.0f}ms inference</div>
            </div>
            """, unsafe_allow_html=True)
            bot_reply = _random.choice(EMPATHETIC_RESPONSES[emo])
            st.markdown(f"""
            <div class='response-box' style='border-left-color:{col}'>
                <div style='color:#8b949e; font-size:0.75rem; margin-bottom:6px; font-family:monospace'>BOT RESPONSE</div>
                {bot_reply}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**Probability Distribution**")
            fig_live = plot_probability_bars(r["probs"])
            st.pyplot(fig_live, use_container_width=True); plt.close()
            st.divider()
            st.markdown("<div class='section-header'>Audio Visualizations</div>", unsafe_allow_html=True)
            lv1, lv2, lv3 = st.columns(3)
            with lv1:
                st.pyplot(plot_waveform(r["audio"], r["sr"], col), use_container_width=True); plt.close()
            with lv2:
                st.pyplot(plot_spectrogram(r["audio"], r["sr"]), use_container_width=True); plt.close()
            with lv3:
                st.pyplot(plot_mfcc(r["audio"], r["sr"]), use_container_width=True); plt.close()
            if st.button("🗑️ Clear Live Result", key="clear_live"):
                st.session_state.live_result = None
                st.rerun()
        else:
            st.markdown("""
            <div style='height:280px; display:flex; align-items:center; justify-content:center;
                 flex-direction:column; color:#8b949e; border:1px dashed #30363d; border-radius:12px'>
                <div style='font-size:3rem'>🔴</div>
                <div style='margin-top:12px; text-align:center; padding:0 24px'>
                    Record your voice to see the emotion analysis here
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Demo Mode
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    import random
    st.markdown("""
    <div style='color:#8b949e; font-size:0.95rem; margin-bottom:20px'>
    Don't have a recording? Generate a <b>synthetic demo audio</b> for any emotion and watch the bot analyze it in real time.
    </div>
    """, unsafe_allow_html=True)

    dcol1, dcol2 = st.columns([1, 1.5], gap="large")

    with dcol1:
        st.markdown("<div class='section-header'>Select Emotion to Simulate</div>", unsafe_allow_html=True)
        for emo in EMOTIONS:
            col_btn, col_info = st.columns([1, 2])
            with col_btn:
                if st.button(f"{EMOTION_EMOJI[emo]} {emo.title()}", key=f"demo_{emo}",
                             use_container_width=True):
                    demo_audio = gen_demo_audio(emo)
                    t0 = time.time()
                    pred_emo, pred_conf, pred_probs = predict_emotion(demo_audio, 22050, pipe, le)
                    ms = (time.time() - t0) * 1000
                    audio_bytes = audio_to_bytes(demo_audio)
                    st.session_state.demo_result = {
                        "selected": emo, "audio": demo_audio, "audio_bytes": audio_bytes,
                        "emotion": pred_emo, "conf": pred_conf, "probs": pred_probs, "ms": ms,
                    }
                    st.session_state.history.append({
                        "emotion": pred_emo, "conf": pred_conf, "source": "demo"
                    })
                    st.rerun()

    with dcol2:
        if "demo_result" in st.session_state and st.session_state.demo_result:
            dr    = st.session_state.demo_result
            color = EMOTION_COLOR[dr["emotion"]]
            st.audio(dr["audio_bytes"], format="audio/wav")
            match_icon = "✅" if dr["emotion"] == dr["selected"] else "🔄"
            st.markdown(f"""
            <div class='emotion-result' style='border-color:{color}; text-align:left; padding:20px'>
                <div style='display:flex; align-items:center; gap:16px'>
                    <span style='font-size:3.5rem'>{EMOTION_EMOJI[dr['emotion']]}</span>
                    <div>
                        <div style='color:#8b949e; font-size:0.75rem; font-family:monospace'>DETECTED</div>
                        <div class='emotion-label' style='color:{color}; font-size:1.5rem'>{dr['emotion']}</div>
                        <div class='emotion-conf'>{dr['conf']:.1%} confidence · {match_icon} vs. selected: <b>{dr['selected']}</b></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='response-box' style='border-left-color:{color}'>
                {random.choice(EMPATHETIC_RESPONSES[dr['emotion']])}
            </div>
            """, unsafe_allow_html=True)
            fig2 = plot_probability_bars(dr["probs"])
            st.pyplot(fig2, use_container_width=True); plt.close()
            fig_w = plot_waveform(dr["audio"], 22050, color)
            st.pyplot(fig_w, use_container_width=True); plt.close()
        else:
            st.markdown("""
            <div style='height:280px; display:flex; align-items:center; justify-content:center;
                 flex-direction:column; color:#8b949e; border:1px dashed #30363d; border-radius:12px'>
                <div style='font-size:3rem'>🎭</div>
                <div style='margin-top:12px'>Click an emotion button to start demo</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Analytics
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>Model Performance Overview</div>", unsafe_allow_html=True)
    acc_key = meta.get("accuracy", meta.get("hold_out_accuracy", 0))
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Test Accuracy</div>
            <div class='metric-value' style='color:#3fb950'>{acc_key*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Emotions</div>
            <div class='metric-value'>{len(EMOTIONS)}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Features</div>
            <div class='metric-value'>{meta['feature_dim']}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Train Samples</div>
            <div class='metric-value'>{meta['n_train']}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    ac1, ac2 = st.columns([1.1, 1], gap="large")
    with ac1:
        st.markdown("**Confusion Matrix**")
        if cm_data is not None:
            fig_cm = plot_confusion_matrix(cm_data, EMOTIONS)
            st.pyplot(fig_cm, use_container_width=True); plt.close()
    with ac2:
        st.markdown("**Feature Engineering Pipeline**")
        features_info = [
            ("MFCC (×40) + Δ + Δ²", "Cepstral coefficients with temporal deltas", 240),
            ("Chroma (×12)",          "Pitch class energy distribution",            24),
            ("Mel Spectrogram",       "Energy across mel-scaled frequency bands",    2),
            ("Spectral Contrast",     "Peak-valley difference across 7 bands",       14),
            ("ZCR",                   "Rate of sign changes — noisiness proxy",       2),
            ("RMS Energy",            "Loudness envelope (mean, std, max)",            3),
            ("Spectral Shape",        "Centroid, bandwidth, rolloff, flatness",        8),
            ("Tonnetz (×6)",          "Tonal centroid features",                     12),
            ("F0 + Voiced Ratio",     "Pitch statistics and voicing fraction",         4),
            ("Jitter / Shimmer",      "Micro-variation in frequency and amplitude",    2),
            ("PCEN",                  "Per-channel energy normalisation",              2),
            ("Formant-band RMS",      "Energy in 4 key speech formant regions",        4),
            ("Tempo & Beat Reg.",     "Rhythmic pace and regularity",                  2),
        ]
        for name, desc, dims in features_info:
            st.markdown(f"""
            <div class='metric-card' style='padding:12px 18px; margin-bottom:6px'>
                <div style='display:flex; justify-content:space-between; align-items:center'>
                    <div>
                        <div style='color:#e6edf3; font-size:0.82rem; font-weight:600'>{name}</div>
                        <div style='color:#8b949e; font-size:0.73rem; margin-top:2px'>{desc}</div>
                    </div>
                    <span class='badge'>{dims}d</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Session Analytics**")
    if st.session_state.history:
        from collections import Counter
        counts  = Counter(h["emotion"] for h in st.session_state.history)
        fig_sess, ax = plt.subplots(figsize=(8, 2.5))
        fig_sess.patch.set_facecolor("#161b22"); ax.set_facecolor("#161b22")
        ax.bar(
            [f"{EMOTION_EMOJI[e]} {e}" for e in counts.keys()],
            counts.values(),
            color=[EMOTION_COLOR[e] for e in counts.keys()],
            edgecolor="none", width=0.5,
        )
        ax.tick_params(colors="#8b949e", labelsize=9)
        for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
        ax.set_title("Emotions Detected This Session", color="#e6edf3", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_sess, use_container_width=True); plt.close()
    else:
        st.info("Analyze some audio files to see session analytics here.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — About
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div style='max-width:800px'>
    <div class='section-header'>About This Project</div>
    <div style='color:#e6edf3; line-height:1.8; font-size:0.95rem'>
    The <b>Emotion-Aware Voice Bot</b> is an intelligent system that enhances human-computer
    interaction by recognizing and responding to user emotions in real time using deep learning and
    classical ML techniques on audio features. It classifies speech into four core emotional states:
    <b>Neutral</b>, <b>Happy</b>, <b>Sad</b>, and <b>Angry</b>.
    </div>
    <div class='section-header' style='margin-top:28px'>Architecture</div>
    """, unsafe_allow_html=True)

    arch_cols = st.columns(6)
    steps = [
        ("🎙️", "Voice Input",     "Upload / Live mic"),
        ("🔊", "Preprocessing",   "Librosa · 22kHz"),
        ("📐", "Features",        "MFCC · Chroma · RMS"),
        ("🌲", "ML Model",        "Ensemble"),
        ("🏷️", "Classification",  "4 Emotions"),
        ("💬", "Response",        "Empathetic reply"),
    ]
    for col, (icon, title, sub) in zip(arch_cols, steps):
        with col:
            st.markdown(f"""
            <div style='background:#161b22; border:1px solid #30363d; border-radius:10px;
                 padding:16px 10px; text-align:center'>
                <div style='font-size:1.8rem'>{icon}</div>
                <div style='color:#e6edf3; font-size:0.8rem; font-weight:600; margin-top:6px'>{title}</div>
                <div style='color:#8b949e; font-size:0.7rem; font-family:monospace; margin-top:4px'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='margin-top:28px'>Tech Stack</div>",
                unsafe_allow_html=True)
    tech = {
        "Python":      "Core language for ML pipeline",
        "scikit-learn":"Ensemble model & preprocessing",
        "Librosa":     "Audio processing & feature extraction",
        "NumPy":       "Numerical computation",
        "Streamlit":   "Interactive web dashboard",
        "Matplotlib":  "Audio visualizations",
        "SoundFile":   "Audio I/O",
    }
    tc1, tc2 = st.columns(2)
    for i, (lib, desc) in enumerate(tech.items()):
        col = tc1 if i % 2 == 0 else tc2
        with col:
            st.markdown(f"""
            <div class='metric-card' style='padding:12px 16px; margin-bottom:8px'>
                <span style='color:#58a6ff; font-weight:600; font-family:monospace'>{lib}</span>
                <span style='color:#8b949e; font-size:0.8rem; margin-left:8px'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-header' style='margin-top:28px'>Research References</div>
    <div style='color:#8b949e; font-size:0.85rem; line-height:2; font-family:monospace'>
    [1] Y. Guo et al., "A Multi-Feature Fusion Speech Emotion Recognition Method," IEEE Access 2023<br>
    [2] J. Wagner et al., "Dawn of the Transformer Era in Speech Emotion Recognition," 2023<br>
    [3] W. Chen et al., "DST: Deformable Speech Transformer for Emotion Recognition," 2023<br>
    [4] J. Ye et al., "Temporal Modeling Matters for Speech Emotion Recognition," 2023<br>
    [5] K. L. Ong et al., "SCQT-MaxViT: Speech Emotion Recognition," IEEE Access 2023
    </div>
    </div>
    """, unsafe_allow_html=True)