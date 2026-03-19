import os
import sys

def app_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__ + "/.."))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FFMPEG_EXE = os.path.join(BASE_DIR, "ffmpeg", "ffmpeg.exe")
FFPROBE_EXE = os.path.join(BASE_DIR, "ffmpeg", "ffprobe.exe")

MODEL_FILES = {
    "logreg": os.path.join(BASE_DIR, "model", "REGRESIONLOGISTICA_model.joblib"),
    "rf": os.path.join(BASE_DIR, "model", "RANDOMFOREST_model.joblib"),
}
TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "report_template.html")

OUTPUT_DIR = os.path.join(BASE_DIR, "output_reports")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)