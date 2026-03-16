"""
motion_features.py — Production version
Frame delta motion analysis
"""

import numpy as np
import cv2


def extract_motion_features(all_frames: list) -> dict:
    """
    Frame-to-frame delta motion features.

    motion_delta_mean — average motion intensity
    motion_delta_std  — variation in motion intensity
    motion_delta_cv   — coefficient of variation
                        AI is either static or uniformly jerky
                        Real video has natural motion variation
    motion_delta_max  — peak motion (scene cuts)
    cut_regularity    — how regular are scene cuts
                        AI content farm often has hyper-regular cuts
    """
    deltas    = []
    prev_gray = None

    for fpath in all_frames:
        img = cv2.imread(fpath)
        if img is None:
            continue
        gray = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY
        ).astype(float)
        if prev_gray is not None:
            deltas.append(
                float(np.mean(np.abs(gray - prev_gray)))
            )
        prev_gray = gray

    if len(deltas) > 2:
        arr = np.array(deltas)
        cv  = float(np.std(arr) / (np.mean(arr) + 1e-8))
        return {
            "motion_delta_mean": float(np.mean(arr)),
            "motion_delta_std":  float(np.std(arr)),
            "motion_delta_cv":   cv,
            "motion_delta_max":  float(np.max(arr)),
            "cut_regularity":    float(1 - cv),
        }

    return {
        "motion_delta_mean": 0.0,
        "motion_delta_std":  0.0,
        "motion_delta_cv":   0.0,
        "motion_delta_max":  0.0,
        "cut_regularity":    0.0,
    }