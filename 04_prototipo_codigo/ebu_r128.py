import os
import re
import subprocess

from .paths import FFMPEG_EXE


def analyze_ebur128_ffmpeg(wav_path: str) -> tuple[float | None, float | None, str]:
    if not os.path.exists(FFMPEG_EXE):
        raise FileNotFoundError(f"No se encontró ffmpeg.exe en: {FFMPEG_EXE}")

    cmd = [
        FFMPEG_EXE,
        "-hide_banner",
        "-nostats",
        "-loglevel", "info",
        "-i", wav_path,
        "-filter_complex", "ebur128=peak=true:framelog=verbose",
        "-f", "null", "-"
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = p.stderr or ""

    if p.returncode != 0:
        raise RuntimeError(f"FFmpeg ebur128 falló.\n\nSTDERR:\n{stderr[-4000:]}")

    integrated = None
    m = re.search(r"\bI:\s*(-?\d+(?:\.\d+)?)\s*LUFS\b", stderr)
    if m:
        try:
            integrated = float(m.group(1))
        except Exception:
            integrated = None

    true_peak = None
    m_tp = re.search(
        r"True\s*peak:\s*(?:.*\n)*?\s*Peak:\s*(-?\d+(?:\.\d+)?)\s*(?:dBFS|dBTP)\b",
        stderr,
        flags=re.IGNORECASE
    )
    if m_tp:
        try:
            true_peak = float(m_tp.group(1))
        except Exception:
            true_peak = None
    else:
        tp_vals = []
        for mt in re.finditer(r"\bPeak:\s*(-?\d+(?:\.\d+)?)\s*(?:dBFS|dBTP)\b", stderr, flags=re.IGNORECASE):
            try:
                tp_vals.append(float(mt.group(1)))
            except Exception:
                pass
        if tp_vals:
            true_peak = max(tp_vals)

    return integrated, true_peak, stderr


def compute_ebu_r128(wav_path: str):
    integrated, true_peak, _stderr = analyze_ebur128_ffmpeg(wav_path)

    reasons = []

    # ✅ LUFS manda el Estado EBU
    if integrated is None:
        reasons.append("No se pudo obtener LUFS Integrated (I) con FFmpeg ebur128.")
        ebu_ok_lufs = False
    else:
        ebu_ok_lufs = (-24.0 <= integrated <= -22.0)
        if not ebu_ok_lufs:
            reasons.append(f"LUFS Integrated fuera de rango (-23 ±1): {integrated:.2f} LUFS")

    # True Peak es “extra”: si no está, se deja advertencia sin tumbar OK
    if true_peak is None:
        reasons.append("No se pudo obtener True Peak con FFmpeg ebur128 (se deja como advertencia).")
        ebu_ok_tp = True
    else:
        ebu_ok_tp = (true_peak <= -1.0)
        if not ebu_ok_tp:
            reasons.append(f"True Peak supera -1.0 dBTP: {true_peak:.2f} dBTP")

    ebu_ok = ebu_ok_lufs and ebu_ok_tp

    return {
        "lufs_integrated": integrated,
        "true_peak_dbtp": true_peak,
        "ebu_ok": ebu_ok,
        "ebu_reasons": reasons
    }