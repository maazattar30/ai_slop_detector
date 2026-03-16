"""
texture_features.py — Production version
Chromatic aberration, saturation CV, compression artifact
"""

import io
import numpy as np
import cv2
from PIL import Image


def extract_texture_features(all_frames: list) -> dict:
    """
    Compute texture and compression features from frames.

    chromatic_aberration     — real lenses have R/G/B misalignment
                               AI renders do not
    saturation_cv            — real scenes have varied saturation
                               AI palettes are uniform
    compression_artifact_mean — real video resists re-compression
                                AI generated content compresses
                                more cleanly
    """
    chrom_vals   = []
    sat_cv_vals  = []
    compress_vals = []

    sample = all_frames[::3] if len(all_frames) > 3 else all_frames

    for fpath in sample:
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            continue

        # Chromatic aberration
        b, g, r   = cv2.split(img_bgr)
        edge_r    = cv2.Canny(r, 50, 150).astype(float)
        edge_g    = cv2.Canny(g, 50, 150).astype(float)
        edge_b    = cv2.Canny(b, 50, 150).astype(float)
        chrom_vals.append(
            (np.mean(np.abs(edge_r - edge_g)) +
             np.mean(np.abs(edge_r - edge_b))) / 2
        )

        # Saturation CV
        hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat     = hsv[:, :, 1].astype(float)
        sat_mean = np.mean(sat)
        sat_cv_vals.append(
            np.std(sat) / (sat_mean + 1e-8)
        )

        # Compression artifact
        pil  = Image.fromarray(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        )
        buf  = io.BytesIO()
        pil.save(buf, "JPEG", quality=50)
        buf.seek(0)
        comp = np.array(Image.open(buf).convert("RGB"))
        orig = np.array(pil)
        compress_vals.append(
            np.mean(np.abs(orig.astype(float) -
                           comp.astype(float)))
        )

    return {
        "chromatic_aberration":      float(np.mean(chrom_vals))
                                     if chrom_vals else 0.0,
        "saturation_cv":             float(np.mean(sat_cv_vals))
                                     if sat_cv_vals else 0.0,
        "compression_artifact_mean": float(np.mean(compress_vals))
                                     if compress_vals else 0.0,
        "compression_artifact_std":  float(np.std(compress_vals))
                                     if compress_vals else 0.0,
    }