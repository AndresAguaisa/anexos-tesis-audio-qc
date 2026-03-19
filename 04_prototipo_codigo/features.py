import numpy as np
import librosa

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False


def stereo_correlation(x: np.ndarray) -> float:
    if x.shape[1] < 2:
        return 1.0
    L = x[:, 0]
    R = x[:, 1]
    if np.std(L) < 1e-8 or np.std(R) < 1e-8:
        return 0.0
    return float(np.corrcoef(L, R)[0, 1])


def silence_ratio_db(x_mono: np.ndarray, thresh_db: float = -60.0) -> float:
    eps = 1e-12
    mag = np.abs(x_mono) + eps
    db = 20.0 * np.log10(mag)
    return float(np.mean(db < thresh_db))


def compute_features_segment(audio: np.ndarray, sr: int) -> dict:
    x_mono = np.mean(audio, axis=1)

    absx = np.abs(x_mono)
    clipping_ratio = float(np.mean(absx >= 0.999))
    near_ceiling_ratio = float(np.mean(absx >= 0.98))

    rms = float(np.sqrt(np.mean(np.square(x_mono)) + 1e-12))
    sample_peak = float(np.max(np.abs(x_mono)) + 1e-12)
    sample_peak_db = 20.0 * np.log10(sample_peak)
    crest = float(sample_peak_db - (20.0 * np.log10(rms + 1e-12)))

    # True peak proxy para feature del modelo (mantener consistente con entrenamiento)
    try:
        x_up = librosa.resample(x_mono, orig_sr=sr, target_sr=sr * 4)
        tp = float(np.max(np.abs(x_up)) + 1e-12)
        true_peak_dbfs = float(20.0 * np.log10(tp))
    except Exception:
        true_peak_dbfs = float(sample_peak_db)

    # short_term_lufs_mean (integrated del segmento) — consistente con entrenamiento
    if HAS_PYLOUDNORM:
        meter = pyln.Meter(sr)
        try:
            lufs = float(meter.integrated_loudness(x_mono))
        except Exception:
            lufs = float("nan")
    else:
        lufs = float("nan")

    # Variación proxy (RMS frames)
    try:
        hop = int(0.1 * sr)
        frame = int(0.4 * sr)
        rms_frames = librosa.feature.rms(y=x_mono, frame_length=frame, hop_length=hop)[0]
        rms_db_frames = 20.0 * np.log10(rms_frames + 1e-12)
        lufs_std_proxy = float(np.std(rms_db_frames))
    except Exception:
        lufs_std_proxy = float("nan")

    sil_ratio = silence_ratio_db(x_mono, thresh_db=-60.0)
    corr = stereo_correlation(audio)

    try:
        centroid = librosa.feature.spectral_centroid(y=x_mono, sr=sr)[0]
        centroid_mean = float(np.mean(centroid))
    except Exception:
        centroid_mean = float("nan")

    # --- NUEVO: spectral_flatness_mean ---
    # Útil para distorsión/saturación armónica (sin clipping duro)
    try:
        flat = librosa.feature.spectral_flatness(y=x_mono)[0]
        spectral_flatness_mean = float(np.mean(flat))
    except Exception:
        spectral_flatness_mean = float("nan")

    # Cap outliers
    if not np.isnan(lufs_std_proxy):
        lufs_std_proxy = float(min(lufs_std_proxy, 40.0))

    return {
        "rms_mean": rms,
        "crest_factor_db": crest,
        "true_peak_dbfs": true_peak_dbfs,
        "short_term_lufs_mean": lufs,
        "loudness_var_proxy_db": lufs_std_proxy,
        "silence_ratio": sil_ratio,
        "stereo_correlation": corr,
        "spectral_centroid_mean": centroid_mean,
        "spectral_flatness_mean": spectral_flatness_mean,
        "clipping_ratio": clipping_ratio,
        "near_ceiling_ratio": near_ceiling_ratio,
    }