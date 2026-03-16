"""
evidence_builder.py — Production version
Combines all feature dicts into weighted module score
"""

import numpy as np
from config import CALIBRATION, BUCKET_THRESHOLDS, BUCKETS


def build_evidence(all_features: dict, speech: bool) -> dict:
    """
    Score each feature against calibration values.
    Compute weighted module score (Cohen d weighted).
    Build evidence brief for LLM.

    Returns:
        module_score   float 0-100
        evidence_brief str
        signal_rows    list of dicts for report card table
        bucket         int 1/2/3
    """
    ai_probs  = []
    weights   = []
    sig_rows  = []
    ev_lines  = []

    for feat, (rm, am, hia, d) in CALIBRATION.items():
        val = all_features.get(feat)
        if val is None:
            continue

        label, ai_prob = _label_signal(
            feat, val, rm, am, hia, d, speech
        )

        ranges = _RANGES.get(feat, "")
        defn   = _DEFINITIONS.get(feat, "")

        ev_lines.append(
            f"  {feat} = {val:.4f}  →  [{label}]"
            f"  |  {ranges}  |  {defn}"
        )

        sig_rows.append({
            "feature":  feat,
            "value":    round(float(val), 4),
            "label":    label,
            "ai_prob":  round(float(ai_prob), 4),
            "ranges":   ranges,
            "definition": defn,
        })

        if "N/A" not in label:
            ai_probs.append(ai_prob)
            weights.append(d)

    # Weighted module score
    if ai_probs:
        w            = np.array(weights)
        w            = w / w.sum()
        module_score = float(np.dot(ai_probs, w) * 100)
    else:
        module_score = 50.0

    evidence_brief = "\n".join(ev_lines)

    return {
        "module_score":   round(module_score, 1),
        "evidence_brief": evidence_brief,
        "signal_rows":    sig_rows,
    }


def assign_bucket(
    final_score: float,
    confidence:  float,
) -> int:
    """
    Assign bucket based on final score and LLM confidence.

    Score 0-40                    → B1 Human Only
    Score 40-65 OR conf < 0.6     → B2 Human + AI Tools
    Score 65+   AND conf >= 0.6   → B3 AI Generated
    """
    t = BUCKET_THRESHOLDS

    if final_score >= t["B3_min_score"] and confidence >= t["confidence_min"]:
        return 3
    elif final_score <= t["B1_max_score"] and confidence >= t["confidence_min"]:
        return 1
    else:
        return 2


def _label_signal(
    feat:     str,
    value:    float,
    rm:       float,
    am:       float,
    hia:      bool,
    d:        float,
    speech:   bool,
) -> tuple:
    """Label a signal value and return (label, ai_prob)."""

    # Conditional pitch signals
    if feat in ("pitch_std_hz", "pitch_range_hz") and not speech:
        return "N/A (no speech)", 0.5

    lo   = min(rm, am)
    hi   = max(rm, am)
    span = hi - lo

    if span < 1e-10:
        return "NEUTRAL", 0.5

    clipped  = np.clip(value, lo - 0.5 * span, hi + 0.5 * span)
    raw      = (clipped - lo) / span
    ai_prob  = float(np.clip(raw if hia else 1 - raw, 0, 1))

    # Pitch override — only flag strongly
    if feat in ("pitch_std_hz", "pitch_range_hz") and speech:
        if ai_prob < 0.7:
            return "NEUTRAL", 0.5

    if   ai_prob >= 0.75: label = "STRONG AI"
    elif ai_prob >= 0.55: label = "MODERATE AI"
    elif ai_prob >= 0.45: label = "NEUTRAL"
    elif ai_prob >= 0.25: label = "MODERATE REAL"
    else:                 label = "STRONG REAL"

    return label, ai_prob


# ─────────────────────────────────────────────
# FEATURE METADATA
# ─────────────────────────────────────────────

_RANGES = {
    "siglip_ai_mean":            "Real: 0.45-0.83 | AI: 0.62-1.00",
    "raft_temporal_entropy":     "Real: 0.96-2.19 | AI: 0.91-2.64",
    "chromatic_aberration":      "Real: 3.30-12.29 | AI: 2.93-8.64",
    "saturation_cv":             "Real: 0.28-1.61 | AI: 0.28-0.92",
    "harmonic_ratio":            "Real: 0.80-4.66 | AI: 0.00-2.92",
    "compression_artifact_mean": "Real: 0.66-2.60 | AI: 0.72-1.87",
    "spectral_bandwidth_mean":   "Real: 1337-4958Hz | AI: 0-3820Hz",
    "silence_ratio":             "Real: 0.00-0.24 | AI: 0.00-1.00",
    "tempo_regularity":          "Real: 0.93-0.97 | AI: 0.00-0.95",
    "motion_delta_cv":           "Real: 0.23-1.06 | AI: 0.46-1.56",
    "pitch_std_hz":              "Real: 24-429Hz | AI: 0-266Hz",
    "pitch_range_hz":            "Real: 244-1899Hz | AI: 0-840Hz",
    "channel_age_days":          "Real: 365+ days | AI: <30 days",
    "subscriber_count":          "Real: 10k+ | AI: <1k",
}

_DEFINITIONS = {
    "siglip_ai_mean":            "SigLIP zero-shot AI visual probability",
    "raft_temporal_entropy":     "Optical flow direction entropy — chaotic = AI",
    "chromatic_aberration":      "R/G/B edge misalignment — real lenses have this",
    "saturation_cv":             "Saturation variation — AI palettes are uniform",
    "harmonic_ratio":            "Human speech is highly harmonic",
    "compression_artifact_mean": "Real video resists re-compression",
    "spectral_bandwidth_mean":   "Real environments are spectrally broad",
    "silence_ratio":             "AI text-to-video often has no audio",
    "tempo_regularity":          "AI music is metronomically perfect",
    "motion_delta_cv":           "AI motion is uniformly static or jerky",
    "pitch_std_hz":              "Human speech has natural pitch variation",
    "pitch_range_hz":            "Human speakers cover wide pitch range",
    "channel_age_days":          "New channels are suspicious",
    "subscriber_count":          "Very low subscribers = content farm",
}