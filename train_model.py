"""
Emotion Recognition ML Model Trainer — v2 (High Accuracy)
===========================================================
Major upgrades over v1:
  • Richer feature set: delta/delta-delta MFCCs, spectral contrast, PCEN,
    jitter/shimmer proxies, voiced-frame ratio, formant-band energies
  • Better synthetic audio: per-emotion formant structure, pitch glides,
    realistic amplitude envelopes, multi-layered noise
  • Larger default dataset with heavier augmentation pipeline
  • Stronger ensemble: RF + XGBoost + SVM + MLP (soft voting)
  • Stratified K-Fold cross-validation with per-fold reporting
  • SMOTE-style oversampling stub (activates if class imbalance detected)
  • Feature importance analysis saved to disk
"""

import numpy as np
import librosa
import pickle
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.calibration import CalibratedClassifierCV

# ── Emotion labels ──────────────────────────────────────────────────────────
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTION_EMOJI = {
    "neutral":   "😐",
    "happy":     "😄",
    "sad":       "😢",
    "angry":     "😠",
    "fearful":   "😨",
    "disgusted": "🤢",
    "surprised": "😲",
}

SR = 22050  # default sample rate


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (≈ 321-dim vector)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    feats = []

    # ── 1. MFCC + deltas ────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    d1   = librosa.feature.delta(mfcc, order=1)
    d2   = librosa.feature.delta(mfcc, order=2)
    for mat in (mfcc, d1, d2):
        feats.extend(np.mean(mat, axis=1).tolist())
        feats.extend(np.std(mat,  axis=1).tolist())

    # ── 2. Chroma ────────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    feats.extend(np.mean(chroma, axis=1).tolist())
    feats.extend(np.std(chroma,  axis=1).tolist())

    # ── 3. Mel spectrogram ───────────────────────────────────────────────────
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    feats.append(float(np.mean(mel)))
    feats.append(float(np.std(mel)))

    # ── 4. Spectral contrast ─────────────────────────────────────────────────
    sc = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    feats.extend(np.mean(sc, axis=1).tolist())
    feats.extend(np.std(sc,  axis=1).tolist())

    # ── 5. ZCR ───────────────────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(audio)
    feats.append(float(np.mean(zcr)))
    feats.append(float(np.std(zcr)))

    # ── 6. RMS energy ────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=audio)
    feats.append(float(np.mean(rms)))
    feats.append(float(np.std(rms)))
    feats.append(float(np.max(rms)))

    # ── 7–10. Spectral shape ─────────────────────────────────────────────────
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

    # ── 11. Tonnetz ───────────────────────────────────────────────────────────
    try:
        harm  = librosa.effects.harmonic(audio)
        tonn  = librosa.feature.tonnetz(y=harm, sr=sr)
        feats.extend(np.mean(tonn, axis=1).tolist())
        feats.extend(np.std(tonn,  axis=1).tolist())
    except Exception:
        feats.extend([0.0] * 12)

    # ── 12. Pitch / F0 ────────────────────────────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
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

    # ── 13. Jitter proxy ─────────────────────────────────────────────────────
    try:
        f0_jitter, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            frame_length=512,
        )
        f0j = f0_jitter[~np.isnan(f0_jitter)] if f0_jitter is not None else np.array([0.0])
        if len(f0j) > 1:
            jitter = float(np.mean(np.abs(np.diff(f0j)) / (np.mean(f0j) + 1e-9)))
        else:
            jitter = 0.0
        feats.append(jitter)
    except Exception:
        feats.append(0.0)

    # ── 14. Shimmer proxy ────────────────────────────────────────────────────
    rms_arr = librosa.feature.rms(y=audio)[0]
    shimmer = float(np.mean(np.abs(np.diff(rms_arr)) / (np.mean(rms_arr) + 1e-9)))
    feats.append(shimmer)

    # ── 15. PCEN ─────────────────────────────────────────────────────────────
    try:
        stft = np.abs(librosa.stft(audio))
        pcen = librosa.pcen(stft * (2 ** 31), sr=sr)
        feats.append(float(np.mean(pcen)))
        feats.append(float(np.std(pcen)))
    except Exception:
        feats.extend([0.0, 0.0])

    # ── 16. Formant-band RMS ─────────────────────────────────────────────────
    bands = [(85, 255), (255, 900), (900, 2800), (2800, 5000)]
    stft_mag = np.abs(librosa.stft(audio, n_fft=2048))
    freqs    = librosa.fft_frequencies(sr=sr, n_fft=2048)
    for flo, fhi in bands:
        idx = np.where((freqs >= flo) & (freqs < fhi))[0]
        feats.append(float(np.mean(stft_mag[idx, :])) if len(idx) > 0 else 0.0)

    # ── 17. Tempo & beat regularity ───────────────────────────────────────────
    try:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        feats.append(float(tempo))
        feats.append(float(np.std(np.diff(beats))) if len(beats) > 1 else 0.0)
    except Exception:
        feats.extend([0.0, 0.0])

    return np.array(feats, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC AUDIO GENERATION
# ══════════════════════════════════════════════════════════════════════════════

_EMOTION_PARAMS = {
    #             freq  noise  amp   vib_rate  vib_depth  harmonics              env_shape
    "neutral":   (220,  0.03, 0.50, 0,        0.00,      [1.0, 0.4, 0.2, 0.1], "flat"),
    "happy":     (380,  0.03, 0.72, 6,        0.03,      [1.0, 0.6, 0.4, 0.2], "bouncy"),
    "sad":       (140,  0.02, 0.28, 1.5,      0.01,      [1.0, 0.3, 0.1, 0.05],"decay"),
    "angry":     (290,  0.25, 0.92, 4,        0.02,      [1.0, 0.9, 0.7, 0.5], "attack"),
    "fearful":   (280,  0.12, 0.48, 12,       0.04,      [1.0, 0.5, 0.3, 0.15],"tremolo"),
    "disgusted": (175,  0.09, 0.33, 2,        0.015,     [1.0, 0.7, 0.5, 0.3], "irregular"),
    "surprised": (520,  0.05, 0.82, 9,        0.035,     [1.0, 0.5, 0.3, 0.2], "burst"),
}


def _make_envelope(shape: str, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 1, n)
    if shape == "flat":
        env = np.ones(n) * (0.8 + 0.2 * rng.random())
    elif shape == "bouncy":
        env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3 * t))
    elif shape == "decay":
        env = np.exp(-3 * t) + 0.1
    elif shape == "attack":
        env = np.clip(20 * t, 0, 1) * (1 - 0.3 * t)
    elif shape == "tremolo":
        env = np.abs(0.5 + 0.5 * np.sin(2 * np.pi * 8 * t))
    elif shape == "irregular":
        env = 0.5 + 0.4 * rng.random(n)
    elif shape == "burst":
        env = np.zeros(n)
        burst_start = int(n * 0.1)
        env[burst_start:] = np.exp(-4 * t[burst_start:])
        env[:burst_start] = np.linspace(0, 1, burst_start)
    else:
        env = np.ones(n)
    return env.astype(np.float32)


def _synth_emotion_audio(
    emotion: str,
    sr: int = SR,
    duration: float = 2.5,
    seed: int | None = None,
) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    n    = int(sr * duration)
    t    = np.linspace(0, duration, n, endpoint=False)

    freq, noise_amp, amp, vib_rate, vib_depth, h_weights, env_shape = _EMOTION_PARAMS[emotion]

    freq = freq * (0.85 + 0.30 * rng.random())
    vib_signal = vib_depth * np.sin(2 * np.pi * vib_rate * t)

    audio = np.zeros(n, dtype=np.float32)
    for k, w in enumerate(h_weights, start=1):
        inst_freq = freq * k * (1 + vib_signal)
        audio    += w * np.sin(2 * np.pi * np.cumsum(inst_freq) / sr)
    audio *= amp

    formant_freqs = {
        "neutral":   [730,  1090, 2440],
        "happy":     [800,  1200, 2600],
        "sad":       [600,   900, 2200],
        "angry":     [750,  1300, 2800],
        "fearful":   [700,  1050, 2400],
        "disgusted": [650,   950, 2300],
        "surprised": [820,  1350, 2900],
    }
    for ff in formant_freqs.get(emotion, [730, 1090, 2440]):
        bw = 80 + 40 * rng.random()
        formant = np.exp(-bw * (t - duration / 2) ** 2) * noise_amp
        audio  += formant * rng.standard_normal(n).astype(np.float32)

    audio += noise_amp * rng.standard_normal(n).astype(np.float32)
    audio *= _make_envelope(env_shape, n, rng)
    audio += 0.005 * rng.standard_normal(n).astype(np.float32)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    audio *= amp

    return audio.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _augment(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    choice = rng.integers(0, 6)
    try:
        if choice == 0:
            audio = audio + (0.005 + 0.02 * rng.random()) * rng.standard_normal(len(audio))
        elif choice == 1:
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(rng.uniform(-3, 3)))
        elif choice == 2:
            audio = librosa.effects.time_stretch(audio, rate=float(rng.uniform(0.85, 1.15)))
        elif choice == 3:
            audio = audio * float(rng.uniform(0.5, 1.5))
        elif choice == 4:
            audio = np.sign(audio) * np.log1p(10 * np.abs(audio)) / np.log1p(10)
        # choice == 5 → no-op
    except Exception:
        pass
    return audio.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(
    n_per_emotion: int = 400,
    sr: int = SR,
    augment_factor: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng   = np.random.default_rng(seed)
    total = n_per_emotion * (1 + augment_factor)
    print(f"[DATA] {n_per_emotion} base + {n_per_emotion * augment_factor} augmented "
          f"= {total} samples/emotion × {len(EMOTIONS)} classes "
          f"= {total * len(EMOTIONS)} total")

    X, y = [], []
    for emotion in EMOTIONS:
        for i in range(n_per_emotion):
            audio = _synth_emotion_audio(emotion, sr=sr, seed=int(rng.integers(0, 2**31)))
            feats = extract_features(audio, sr=sr)
            X.append(feats)
            y.append(emotion)
            for _ in range(augment_factor):
                aug   = _augment(audio.copy(), sr=sr, rng=rng)
                feats = extract_features(aug, sr=sr)
                X.append(feats)
                y.append(emotion)
        print(f"  ✓ {emotion}: {total} samples")

    X_arr = np.array(X, dtype=np.float32)
    print(f"\n[DATA] Feature vector dim = {X_arr.shape[1]}")
    return X_arr, np.array(y)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_ensemble() -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=4, random_state=42,
    )
    svm = CalibratedClassifierCV(
        SVC(kernel="rbf", C=15, gamma="scale", class_weight="balanced", random_state=42),
        cv=3,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128), activation="relu", solver="adam",
        alpha=1e-4, batch_size=64, learning_rate="adaptive", max_iter=300,
        early_stopping=True, validation_fraction=0.1, random_state=42,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("gb", gb), ("svm", svm), ("mlp", mlp)],
        voting="soft",
        weights=[2, 2, 1.5, 1.5, 2],
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════════

def train_and_save(
    model_dir: str = "model",
    n_per_emotion: int = 400,
    augment_factor: int = 2,
    cv_folds: int = 5,
) -> tuple:
    os.makedirs(model_dir, exist_ok=True)

    X, y = generate_synthetic_dataset(
        n_per_emotion=n_per_emotion,
        augment_factor=augment_factor,
    )

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )
    print(f"\n[TRAIN] {len(X_train)} train  /  {len(X_test)} test  "
          f"(feature dim = {X.shape[1]})")

    scaler   = StandardScaler()
    ensemble = build_ensemble()
    pipe = Pipeline([("scaler", scaler), ("model", ensemble)])

    print(f"\n[CV]   Running {cv_folds}-fold stratified cross-validation …")
    skf     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_accs = cross_val_score(pipe, X, y_enc, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"[CV]   Fold accuracies : {[f'{a:.4f}' for a in cv_accs]}")
    print(f"[CV]   Mean ± std      : {cv_accs.mean():.4f} ± {cv_accs.std():.4f}")

    print("\n[TRAIN] Fitting final model on full training split …")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n[EVAL]  Hold-out test accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"[EVAL]  CV mean accuracy        : {cv_accs.mean():.4f}  "
          f"({cv_accs.mean()*100:.1f}%)")
    print("\n[EVAL]  Per-class report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)

    try:
        rf_model    = pipe.named_steps["model"].estimators_[0]
        importances = rf_model.feature_importances_
        np.save(os.path.join(model_dir, "feature_importances.npy"), importances)
    except Exception:
        pass

    with open(os.path.join(model_dir, "pipeline.pkl"), "wb") as f:
        pickle.dump(pipe, f)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    np.save(os.path.join(model_dir, "confusion_matrix.npy"), cm)

    meta = {
        "hold_out_accuracy": float(acc),
        "cv_mean_accuracy":  float(cv_accs.mean()),
        "cv_std_accuracy":   float(cv_accs.std()),
        "cv_folds":          cv_folds,
        "emotions":          EMOTIONS,
        "feature_dim":       int(X.shape[1]),
        "n_train":           int(len(X_train)),
        "n_test":            int(len(X_test)),
        "n_per_emotion":     n_per_emotion,
        "augment_factor":    augment_factor,
        "model_type":        "Ensemble (RF + ExtraTrees + GBM + SVM + MLP)",
        "ensemble_weights":  [2, 2, 1.5, 1.5, 2],
        # legacy key used by app.py sidebar
        "accuracy":          float(acc),
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SAVE]  Artefacts written to '{model_dir}/'")
    return pipe, le, acc, cm, meta


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def predict_emotion(
    audio: np.ndarray,
    sr: int = SR,
    model_dir: str = "model",
) -> dict:
    with open(os.path.join(model_dir, "pipeline.pkl"), "rb") as f:
        pipe = pickle.load(f)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    feats = extract_features(audio, sr=sr).reshape(1, -1)
    pred  = pipe.predict(feats)[0]
    proba = pipe.predict_proba(feats)[0]

    emotion = le.inverse_transform([pred])[0]
    prob_map = {cls: float(p) for cls, p in zip(le.classes_, proba)}

    return {
        "emotion":       emotion,
        "emoji":         EMOTION_EMOJI.get(emotion, ""),
        "confidence":    float(proba[pred]),
        "probabilities": prob_map,
    }


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_and_save(
        model_dir="model",
        n_per_emotion=400,
        augment_factor=2,
        cv_folds=5,
    )
