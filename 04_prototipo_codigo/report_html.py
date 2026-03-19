import os
from datetime import datetime

from .paths import OUTPUT_DIR, TEMPLATE_PATH
from .utils import safe_filename, sec_to_mmss


def probable_findings(feat: dict, proba_no_ok: float | None = None) -> list:
    findings = []

    # 1) Silencio anómalo
    if feat.get("silence_ratio", 0.0) >= 0.30:
        findings.append("Posible silencio anómalo (proporción de silencio elevada)")

    # 2) Fase / mono incompatible
    if feat.get("stereo_correlation", 1.0) <= 0.0:
        findings.append("Posible problema de fase o señal mono incompatible (correlación <= 0)")

    # 3) Picos altos (proxy)
    if feat.get("true_peak_dbfs", -99.0) > -1.0:
        findings.append("Picos de señal elevados (true peak cercano o superior a -1 dBFS)")

    # 4) Loudness extremo (por segmento)
    st_lufs = feat.get("short_term_lufs_mean", None)
    if st_lufs is not None:
        try:
            if st_lufs > -18.0:
                findings.append("Nivel de sonoridad elevado en el segmento")
            if st_lufs < -31.0:
                findings.append("Nivel de sonoridad muy bajo en el segmento")
        except Exception:
            pass

    # 5) Saturación / limitación (mejorada)
    clipping = float(feat.get("clipping_ratio", 0.0) or 0.0)
    near_ceiling = float(feat.get("near_ceiling_ratio", 0.0) or 0.0)
    crest = float(feat.get("crest_factor_db", 999.0) or 999.0)

    # A) clipping claro
    if clipping > 0.0002:
        findings.append("Posible clipping/saturación digital (muestras al techo detectadas)")

    # B) limitación agresiva: cerca del techo + señal aplastada
    if near_ceiling > 0.005 and crest <= 7.0:
        findings.append("Posible limitación agresiva / saturación audible (señal muy 'aplastada')")

    # C) NUEVO: saturación audible aunque no llegue al techo:
    #    Si la IA está muy segura y el crest es bajo, lo reportamos como limitación/saturación.
    if not any("saturación" in s.lower() or "limitación" in s.lower() or "clipping" in s.lower() for s in findings):
        if proba_no_ok is not None and proba_no_ok >= 0.75 and crest <= 6.5:
            findings.append("Posible saturación/limitación audible detectada por IA (crest factor bajo)")

    
    import math

    flatness = feat.get("spectral_flatness_mean", 0.0)
    try:
        flatness = float(flatness)
    except Exception:
        flatness = 0.0
    if math.isnan(flatness):
        flatness = 0.0

    # Distorsión / saturación armónica probable (calibrado a tus valores reales)
    if proba_no_ok is not None and proba_no_ok >= 0.75 and flatness >= 0.0035:
        findings.append("Posible distorsión/saturación audible (patrón espectral anómalo)")
    
    # Fallback claro
    if not findings:
        findings.append(
            "Advertencia leve (IA): Patrón inusual. Revisar este segmento puntual. "
        )

    return findings

def render_report_html(
    mxf_path: str,
    ebu: dict,
    segments: list,
    decision: dict,
    model_suffix: str = "",
    output_dir: str | None = None
) -> str:
    base_name = safe_filename(os.path.splitext(os.path.basename(mxf_path))[0])
    save_dir = output_dir if output_dir else OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, f"QC_{base_name}_ANALISISAUDIO{model_suffix}.html")

    if os.path.exists(TEMPLATE_PATH):
        if os.path.exists(out_path):
            os.remove(out_path)
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            tpl = f.read()
    else:
        tpl = """<!doctype html>
<html><head><meta charset="utf-8"><title>{{TITLE}}</title>
<style>
body{font-family:Arial, sans-serif; margin:24px;}
.bad{color:#b00020; font-weight:bold;}
.ok{color:#1b5e20; font-weight:bold;}
table{border-collapse:collapse; width:100%; margin-top:12px;}
th,td{border:1px solid #ddd; padding:8px; font-size:14px;}
th{background:#f3f3f3;}
small{color:#555;}
</style></head>
<body>
<h1>{{TITLE}}</h1>
{{SUMMARY}}
<h2>EBU R128</h2>
{{EBU}}
<h2>Segmentos problemáticos (IA)</h2>
{{SEGMENTS}}
</body></html>"""

    final_status = "REQUIERE REVISIÓN" if decision["final_requires_review"] else "OK"
    status_class = "bad" if decision["final_requires_review"] else "ok"

    n_imp = sum(1 for s in segments if int(s.get("features", {}).get("_imputed_count", 0) or 0) > 0)

    summary_html = f"""
    <p><b>Archivo:</b> {os.path.basename(mxf_path)}</p>
    <p><b>Resultado final:</b> <span class="{status_class}">{final_status}</span></p>
    <p><b>IA:</b> {decision["n_bad"]}/{decision["n_total"]} segmentos NO_OK
       ({decision["bad_ratio"]*100:.1f}%), max racha NO_OK = {decision["max_run_bad"]}</p>
    <p><b>Medición parcial:</b> {n_imp}/{decision["n_total"]} segmentos con features imputadas</p>
    """

    ebu_status = "OK" if ebu["ebu_ok"] else "FAIL"
    ebu_class = "ok" if ebu["ebu_ok"] else "bad"

    reasons = "<br>".join(ebu["ebu_reasons"]) if ebu["ebu_reasons"] else "Sin observaciones."
    lufs_str = "N/A" if ebu["lufs_integrated"] is None else f"{ebu['lufs_integrated']:.2f}"
    tp_str = "N/A" if ebu["true_peak_dbtp"] is None else f"{ebu['true_peak_dbtp']:.2f}"

    ebu_html = f"""
    <p><b>Estado EBU:</b> <span class="{ebu_class}">{ebu_status}</span></p>
    <p><b>LUFS Integrated:</b> {lufs_str}</p>
    <p><b>True Peak (FFmpeg ebur128):</b> {tp_str} dBTP</p>
    <p><b>Observaciones:</b><br><small>{reasons}</small></p>
    """

    rows = []
    for s in segments:
        if s["pred_no_ok"] != 1:
            continue

        #print(
        #"DEBUG:",
        #s["proba_no_ok"],
        #s["features"].get("crest_factor_db"),
        #s["features"].get("clipping_ratio"),
        #s["features"].get("near_ceiling_ratio"),
        #s["features"].get("spectral_flatness_mean"),
        #)

        fnd = probable_findings(s["features"], s["proba_no_ok"])

        #CAMBIO

        # Marcar segmentos con medición parcial (features imputadas)
        imp_n = int(s["features"].get("_imputed_count", 0) or 0)
        imp_keys = s["features"].get("_imputed_keys", []) or []
        if imp_n > 0:
            msg = f"⚠ Medición parcial: {imp_n} feature(s) imputados"
            if len(imp_keys) > 0:
                msg += f" ({', '.join(imp_keys)})"
            # Lo ponemos primero para que sea visible
            fnd.insert(0, msg)

        #cambio

        rows.append(f"""
        <tr>
          <td>{sec_to_mmss(s["start_s"])}</td>
          <td>{sec_to_mmss(s["end_s"])}</td>
          <td>{s["proba_no_ok"]:.3f}</td>
          <td><small>{"; ".join(fnd)}</small></td>
        </tr>
        """)

    if rows:
        seg_html = """
        <table>
          <thead><tr><th>Inicio</th><th>Fin</th><th>Prob(NO_OK)</th><th>Hallazgos probables</th></tr></thead>
          <tbody>
        """ + "\n".join(rows) + """
          </tbody>
        </table>
        """
    else:
        seg_html = "<p class='ok'><b>No se detectaron segmentos NO_OK.</b></p>"

    html = (
        tpl.replace("{{TITLE}}", f"Reporte QC - {base_name}")
        .replace("{{SUMMARY}}", summary_html)
        .replace("{{EBU}}", ebu_html)
        .replace("{{SEGMENTS}}", seg_html)
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path