import json
import os
import subprocess
from dataclasses import dataclass

from .paths import FFPROBE_EXE, FFMPEG_EXE


@dataclass
class MXFAudioCheck:
    ok: bool
    reason: str
    mode: str | None = None  # "stereo_stream" | "mono_pair"
    audio_stream_index: int | None = None
    mono_pair_indices: tuple[int, int] | None = None
    channels: int | None = None
    sample_rate: int | None = None
    codec: str | None = None


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Error ejecutando comando:\n"
            f"{' '.join(cmd)}\n\n"
            f"STDERR:\n{(p.stderr or '')[-4000:]}"
        )
    return p


def ffprobe_streams(mxf_path: str) -> dict:
    if not os.path.exists(FFPROBE_EXE):
        raise FileNotFoundError(f"No se encontró ffprobe.exe en: {FFPROBE_EXE}")
    cmd = [FFPROBE_EXE, "-hide_banner", "-loglevel", "error", "-show_streams", "-of", "json", mxf_path]
    p = run_cmd(cmd)
    return json.loads(p.stdout)


def validate_audio_layout(mxf_path: str) -> MXFAudioCheck:
    info = ffprobe_streams(mxf_path)
    streams = info.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]

    if not audio:
        return MXFAudioCheck(False, "FAIL: No se encontró audio en el MXF.")

    if len(audio) == 1:
        ch = int(audio[0].get("channels", 0) or 0)
        sr = int(audio[0].get("sample_rate", 0) or 0)
        codec = audio[0].get("codec_name") or ""
        idx = int(audio[0].get("index", -1))
        if ch == 2:
            return MXFAudioCheck(True, "OK: 1 stream estéreo.", mode="stereo_stream",
                                 audio_stream_index=idx, channels=ch, sample_rate=sr, codec=codec)
        return MXFAudioCheck(False, f"FAIL: 1 stream con {ch} canales (no soportado).")

    if len(audio) == 2:
        ch1 = int(audio[0].get("channels", 0) or 0)
        ch2 = int(audio[1].get("channels", 0) or 0)
        sr1 = int(audio[0].get("sample_rate", 0) or 0)
        sr2 = int(audio[1].get("sample_rate", 0) or 0)
        idx1 = int(audio[0].get("index", -1))
        idx2 = int(audio[1].get("index", -1))
        codec1 = audio[0].get("codec_name") or ""
        codec2 = audio[1].get("codec_name") or ""
        if ch1 == 1 and ch2 == 1:
            return MXFAudioCheck(True, "OK: 2 streams mono (se unirán a estéreo).",
                                 mode="mono_pair", mono_pair_indices=(idx1, idx2),
                                 channels=2, sample_rate=sr1 if sr1 else sr2, codec=f"{codec1}/{codec2}")
        return MXFAudioCheck(False, f"FAIL: 2 streams con {ch1} y {ch2} canales (no soportado).")

    return MXFAudioCheck(False, f"FAIL: {len(audio)} streams de audio (no soportado).")


def extract_stereo_stream_to_wav(mxf_path: str, out_wav: str, stream_index: int):
    if not os.path.exists(FFMPEG_EXE):
        raise FileNotFoundError(f"No se encontró ffmpeg.exe en: {FFMPEG_EXE}")
    cmd = [
        FFMPEG_EXE, "-y",
        "-i", mxf_path,
        "-map", f"0:{stream_index}",
        "-vn",
        "-ar", "48000",
        "-acodec", "pcm_s24le",
        out_wav
    ]
    run_cmd(cmd)


def extract_dual_mono_to_stereo_wav(mxf_path: str, out_wav: str, left_index: int, right_index: int):
    if not os.path.exists(FFMPEG_EXE):
        raise FileNotFoundError(f"No se encontró ffmpeg.exe en: {FFMPEG_EXE}")
    cmd = [
        FFMPEG_EXE, "-y",
        "-i", mxf_path,
        "-filter_complex", f"[0:{left_index}][0:{right_index}]join=inputs=2:channel_layout=stereo[a]",
        "-map", "[a]",
        "-vn",
        "-ar", "48000",
        "-acodec", "pcm_s24le",
        out_wav
    ]
    run_cmd(cmd)


def extract_audio_to_wav_auto(mxf_path: str, out_wav: str, check: MXFAudioCheck):
    if not check.ok:
        raise RuntimeError(check.reason)

    if check.mode == "stereo_stream":
        extract_stereo_stream_to_wav(mxf_path, out_wav, int(check.audio_stream_index))
        return

    if check.mode == "mono_pair":
        li, ri = check.mono_pair_indices or (None, None)
        if li is None or ri is None:
            raise RuntimeError("FAIL: No se pudo determinar el par mono para extracción.")
        extract_dual_mono_to_stereo_wav(mxf_path, out_wav, li, ri)
        return

    raise RuntimeError("FAIL: Modo de extracción no soportado.")