import os
import uuid
import shutil
import soundfile as sf
import numpy as np

from .paths import TEMP_DIR, ensure_dirs
from .mxf_audio import validate_audio_layout, extract_audio_to_wav_auto
from .ebu_r128 import compute_ebu_r128
from .features import compute_features_segment
from .model_infer import load_model
from .decision import decide_file_result
from .report_html import render_report_html


def iter_segments(wav_path: str, segment_s: float = 5.0, min_last_s: float = 1.0):
    info = sf.info(wav_path)
    sr = info.samplerate
    total_samples = info.frames
    total_s = total_samples / sr

    start = 0.0
    while start < total_s:
        end = min(start + segment_s, total_s)

        # Omitir último segmento si es demasiado corto (cola de export / redondeos)
        if (end - start) < min_last_s:
            break

        yield start, end, sr
        start = end


def load_segment(wav_path: str, start_s: float, end_s: float):
    info = sf.info(wav_path)
    sr = info.samplerate
    start_frame = int(round(start_s * sr))
    end_frame = int(round(end_s * sr))
    frames = max(0, end_frame - start_frame)

    with sf.SoundFile(wav_path, mode="r") as f:
        f.seek(start_frame)
        audio = f.read(frames, dtype="float32", always_2d=True)

    if audio.shape[0] == 0:
        raise ValueError("Segmento vacío.")
    return audio, sr


def analyze_mxf(input_path: str, segment_s: float = 5.0, model_key: str = "logreg", output_dir: str | None = None):
    """
    Acepta .mxf o .wav.
    - Si es MXF: extrae WAV temporal (como medidor)
    - Si es WAV: usa el WAV directamente (sin temporales extra)
    """
    ensure_dirs()

    ext = os.path.splitext(input_path)[1].lower()

    temp_run = os.path.join(TEMP_DIR, str(uuid.uuid4()))
    os.makedirs(temp_run, exist_ok=True)

    try:
        # 1) Determinar WAV de trabajo
        if ext == ".wav":
            wav_path = input_path
            if not os.path.exists(wav_path):
                raise RuntimeError("No se encontró el WAV.")
        elif ext == ".mxf":
            wav_path = os.path.join(temp_run, "audio.wav")
            check = validate_audio_layout(input_path)
            if not check.ok:
                raise RuntimeError(check.reason)
            extract_audio_to_wav_auto(input_path, wav_path, check)
            if not os.path.exists(wav_path):
                raise RuntimeError("FAIL: No se generó el WAV temporal.")
        else:
            raise RuntimeError("Formato no soportado. Usa MXF o WAV.")

        # 2) EBU R128
        ebu = compute_ebu_r128(wav_path)

        # 3) IA por segmentos (LogReg)
        model, feature_cols = load_model(model_key)
        segments = []
        for start_s, end_s, sr in iter_segments(wav_path, segment_s=segment_s, min_last_s=1.0):
            audio, sr = load_segment(wav_path, start_s, end_s)
            feat = compute_features_segment(audio, sr)

            #CAMBIO

            # Reemplazo conservador de NaN/inf + registrar imputaciones (para reporte)
            imputed = []  # features imputadas

            def _is_bad(v) -> bool:
                try:
                    fv = float(v)
                    return np.isnan(fv) or np.isinf(fv)
                except Exception:
                    return True

            # Defaults conservadores (no inventan energía)
            DEFAULTS = {
                "true_peak_dbfs": -99.0,
                "short_term_lufs_mean": -99.0,
                "loudness_var_proxy_db": 0.0,
                "spectral_centroid_mean": 0.0,
                "spectral_flatness_mean": 0.0,
                "stereo_correlation": 1.0,   # neutral (evita falsos "fase" si falló el cálculo)
                "silence_ratio": 0.0,
                "rms_mean": 0.0,
                "crest_factor_db": 0.0,
                "clipping_ratio": 0.0,
                "near_ceiling_ratio": 0.0,
            }

            # Si el segmento está casi vacío, forzar valores coherentes con "silencio"
            rms_val = feat.get("rms_mean", None)
            is_empty = _is_bad(rms_val)
            if not is_empty:
                try:
                    is_empty = float(rms_val) < 1e-6
                except Exception:
                    is_empty = True

            if is_empty:
                DEFAULTS["silence_ratio"] = 1.0
                DEFAULTS["short_term_lufs_mean"] = -99.0
                DEFAULTS["true_peak_dbfs"] = -99.0

            # Aplicar imputación
            for k in list(feat.keys()):
                if _is_bad(feat[k]):
                    feat[k] = DEFAULTS.get(k, 0.0)
                    imputed.append(k)
                else:
                    # normalizar a float para evitar tipos raros
                    feat[k] = float(feat[k])

            # Guardar metadatos para el reporte
            feat["_imputed_count"] = len(imputed)
            feat["_imputed_keys"] = imputed

            #CAMBIO

            x = np.array([[feat.get(c, 0.0) for c in feature_cols]], dtype=float)
            proba_no_ok = float(model.predict_proba(x)[:, 1][0])

            # Umbral por modelo: se establece un Umbral de 0.8 para Random Forest y de 0.9 para Regresion Logistica
            if model_key == "rf":
                SEGMENT_NOOK_TH = 0.70
            else:
                SEGMENT_NOOK_TH = 0.90

            pred_no_ok = 1 if proba_no_ok >= SEGMENT_NOOK_TH else 0
            
            segments.append({
                "start_s": start_s,
                "end_s": end_s,
                "pred_no_ok": pred_no_ok,
                "proba_no_ok": proba_no_ok,
                "features": feat
            })

        # 4) Decisión final
        decision = decide_file_result(ebu, segments)

        # 5) Reporte HTML (muestra el nombre original del archivo de entrada)

        # sufijo según modelo usado
        model_suffix = "_LOGREG" if model_key == "logreg" else "_RANDOMF"
        
        report_path = render_report_html(
            input_path,
            ebu,
            segments,
            decision,
            model_suffix=model_suffix,
            output_dir=output_dir
        )

        return report_path, decision, ebu

    finally:
        # Limpieza SOLO si era MXF (si era WAV directo no pasa nada)
        try:
            shutil.rmtree(temp_run, ignore_errors=True)
        except Exception:
            pass