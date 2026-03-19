def decide_file_result(ebu: dict, segments: list):
    n_total = len(segments)
    n_bad = sum(1 for s in segments if s["pred_no_ok"] == 1)
    bad_ratio = (n_bad / n_total) if n_total > 0 else 0.0

    max_run = 0
    run = 0
    for s in segments:
        if s["pred_no_ok"] == 1:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    # Más estricto: "fuerte" si >= 0.95 (y opcionalmente más de 1 vez)
    strong_count = sum(1 for s in segments if s["proba_no_ok"] >= 0.95)
    ia_strong = strong_count >= 2  # requiere 2 segmentos muy seguros

    # Más flexible: solo revisión si hay evidencia clara y persistente
    ia_requires_review = (bad_ratio >= 0.20) or (max_run >= 4) or ia_strong

    final_requires_review = (not ebu["ebu_ok"]) or ia_requires_review

    return {
        "n_total": n_total,
        "n_bad": n_bad,
        "bad_ratio": bad_ratio,
        "max_run_bad": max_run,
        "ia_requires_review": ia_requires_review,
        "final_requires_review": final_requires_review
    }