def calculate_final_score(fn, md, ml):
    bucket = fn.get("bucket", "neutral_or_real_hint")

    if bucket == "strong_ai_trigger":
        w_fn, w_md, w_ml = 0.25, 0.15, 0.60
        mode = "Filename-flagged mode"
        mode_detail = "AI keyword in filename but pixel analysis still dominates"
    elif bucket == "medium_ai_suspicion":
        w_fn, w_md, w_ml = 0.10, 0.20, 0.70
        mode = "Suspicious filename mode"
        mode_detail = "Suspicious filename patterns, models and forensics drive the verdict"
    else:
        w_fn, w_md, w_ml = 0.00, 0.12, 0.88
        mode = "Standard analysis mode"
        mode_detail = "No filename triggers, deep learning models and forensics drive the verdict"

    def share(a, r):
        t = a + r
        if t == 0:
            return 50.0, 50.0
        return (a / t) * 100, (r / t) * 100

    fn_a, fn_r = share(fn.get("ai_points", 0), fn.get("real_points", 0))
    md_a, md_r = share(md.get("ai_points", 0), md.get("real_points", 0))
    ml_a, ml_r = share(ml.get("ai_points", 0), ml.get("real_points", 0))

    ai_s = fn_a * w_fn + md_a * w_md + ml_a * w_ml
    real_s = fn_r * w_fn + md_r * w_md + ml_r * w_ml
    tot = ai_s + real_s or 1
    ai_s = round((ai_s / tot) * 100, 1)
    real_s = round((real_s / tot) * 100, 1)

    forensic_override = False
    forensics = ml.get("forensics", {})
    if forensics and bucket == "neutral_or_real_hint":
        kurt_ai = forensics.get("kurtosis", {}).get("ai_prob", 0.5)
        dfi_ai = forensics.get("dfi", {}).get("ai_prob", 0.5)
        model_ai = ml.get("weighted_ai_prob", 0.5)
        forensic_avg = (kurt_ai + dfi_ai) / 2
        if abs(forensic_avg - model_ai) > 0.40:
            forensic_override = True

    if ai_s >= 50:
        verdict = "Fake"
        if ai_s >= 85:
            confidence = "Very High"
            color = "#ef4444"
        elif ai_s >= 70:
            confidence = "High"
            color = "#f87171"
        else:
            confidence = "Medium"
            color = "#f97316"
    else:
        verdict = "Real"
        if ai_s <= 15:
            confidence = "Very High"
            color = "#22c55e"
        elif ai_s <= 30:
            confidence = "High"
            color = "#4ade80"
        else:
            confidence = "Medium"
            color = "#86efac"

    if forensic_override:
        if confidence in ("Very High", "High"):
            confidence = "Medium"
        elif confidence == "Medium":
            confidence = "Low"

    breakdown = [
        {
            "layer": "Filename Analysis",
            "ai_pts": fn.get("ai_points", 0),
            "real_pts": fn.get("real_points", 0),
            "weight_pct": f"{int(w_fn * 100)}%",
            "signals": fn.get("signals", []),
            "mode": fn.get("priority_note", ""),
        },
        {
            "layer": "Metadata Analysis",
            "ai_pts": md.get("ai_points", 0),
            "real_pts": md.get("real_points", 0),
            "weight_pct": f"{int(w_md * 100)}%",
            "signals": md.get("signals", []),
            "mode": "Metadata-provenance layer.",
        },
        {
            "layer": "AI Model and Forensic Detectors",
            "ai_pts": ml.get("ai_points", 0),
            "real_pts": ml.get("real_points", 0),
            "weight_pct": f"{int(w_ml * 100)}%",
            "signals": ml.get("signals", []),
            "votes": ml.get("votes", []),
            "forensics": ml.get("forensics", {}),
            "mode": ml.get("priority_note", ""),
        },
    ]

    summary = (
        f"Scoring mode: {mode}. "
        f"Final AI score: {ai_s}%. "
        f"Verdict: {verdict} (Confidence: {confidence}). "
        f"{mode_detail}."
    )
    if forensic_override:
        summary += " Forensic signals conflict with model predictions."

    return {
        "verdict": verdict,
        "ai_score": ai_s,
        "real_score": real_s,
        "confidence": confidence,
        "color": color,
        "breakdown": breakdown,
        "summary": summary,
        "scoring_mode": mode,
        "forensic_override": forensic_override,
        "weights": {"filename": w_fn, "metadata": w_md, "models": w_ml},
    }
