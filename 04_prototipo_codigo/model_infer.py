import os
import numpy as np
import joblib

from .paths import MODEL_FILES

def load_model(model_key="logreg"):
    if model_key not in MODEL_FILES:
        raise ValueError(f"Modelo no válido: {model_key}")

    model_path = MODEL_FILES[model_key]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    obj = joblib.load(model_path)

    model = obj["model"]
    feature_cols = obj["feature_cols"]

    return model, feature_cols
  
def predict_one(feat: dict, model_key="logreg"):
    model, feature_cols = load_model(model_key)
    x = np.array([feat[c] for c in feature_cols], dtype=float).reshape(1, -1)
    proba_no_ok = float(model.predict_proba(x)[:, 1][0])
    pred_no_ok = int(model.predict(x)[0])
    return pred_no_ok, proba_no_ok