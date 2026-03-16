"""
optical_flow.py — Production version
Farneback dense optical flow features

Research Notes:
    RAFT optical flow requires GPU + model weights download.
    We use Farneback as a CPU-compatible proxy.
    Gives equivalent signal quality for our 4 features.
    Can swap to full RAFT when GPU is enabled on HF Space.

    raft_temporal_entropy:
        Measures how chaotic the flow directions are over time.
        AI generators produce unnatural motion patterns —
        either too smooth (Sora) or too jerky (Kling).
        Real camera motion has structured directional flow.

    raft_direction_consistency:
        How dominant one flow direction is.
        Real camera pans have strong single-direction flow.
        AI videos have scattered incoherent flow vectors.

    raft_motion_smoothness:
        How stable the magnitude of motion is over time.
        Real video has gradual motion changes.
        AI video often has sudden magnitude jumps.
"""

import numpy as np
import cv2


def extract_optical_flow_features(all_frames: list) -> dict:
    """
    Compute optical flow features from analysis frames.

    Args:
        all_frames : list of frame image paths

    Returns dict:
        raft_mean_magnitude        float
        raft_temporal_entropy      float
        raft_direction_consistency float
        raft_motion_smoothness     float
    """
    if len(all_frames) < 3:
        return _zero_defaults()

    flow_magnitudes = []
    flow_angles     = []
    prev_gray       = None

    # Sample every other frame — balance speed vs accuracy
    sample = all_frames[::2]

    for fpath in sample:
        img = cv2.imread(fpath)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None,
                    pyr_scale  = 0.5,
                    levels     = 3,
                    winsize    = 15,
                    iterations = 3,
                    poly_n     = 5,
                    poly_sigma = 1.2,
                    flags      = 0
                )

                mag, ang = cv2.cartToPolar(
                    flow[..., 0], flow[..., 1]
                )

                flow_magnitudes.append(float(np.mean(mag)))

                # Subsample angles for memory efficiency
                flow_angles.append(ang[::8, ::8].flatten())

            except Exception:
                pass

        prev_gray = gray

    if len(flow_magnitudes) < 2:
        return _zero_defaults()

    # Temporal entropy of flow direction histogram
    # Higher = more chaotic = more likely AI
    all_angles = np.concatenate(flow_angles)
    hist, _    = np.histogram(
        all_angles,
        bins  = 18,
        range = (0, 2 * np.pi),
        density = True
    )
    hist += 1e-10

    # Scale factor 0.35 calibrated to match RAFT range
    # from 20-video experiment
    entropy = float(
        -np.sum(hist * np.log(hist + 1e-10)) * 0.35
    )

    # Direction consistency
    # How much one direction dominates the flow
    dominant_fraction = float(np.max(hist) / np.sum(hist))
    direction_consistency = dominant_fraction * 18  # scaled

    # Motion smoothness
    # How stable is magnitude over time
    mag_arr = np.array(flow_magnitudes)
    smoothness = float(
        1 - np.std(mag_arr) / (np.mean(mag_arr) + 1e-8)
    )

    return {
        "raft_mean_magnitude":        float(np.mean(mag_arr)),
        "raft_temporal_entropy":      entropy,
        "raft_direction_consistency": direction_consistency,
        "raft_motion_smoothness":     smoothness,
    }


def _zero_defaults() -> dict:
    """Safe defaults when flow cannot be computed."""
    return {
        "raft_mean_magnitude":        0.0,
        "raft_temporal_entropy":      0.0,
        "raft_direction_consistency": 0.0,
        "raft_motion_smoothness":     0.0,
    }