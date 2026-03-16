"""
audio_features.py — Production version
13 audio signals using librosa
"""

import os
import subprocess
import warnings
import numpy as np
import librosa

warnings.filterwarnings("ignore")

from config import (
    AUDIO_SAMPLE_RATE,
    SILENCE_THRESHOLD,
    PITCH_MIN_HZ,
    PITCH_MAX_HZ,
)


def extract_audio_features(
    video_path: str,
    work_dir:   str,
) -> dict:
    """Extract all audio features from video."""
    audio_path = os.path.join(work_dir, "audio.wav")
    audio_ok   = _extract_audio(video_path, audio_path)

    if not audio_ok:
        return _silent_defaults()

    try:
        y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
    except Exception:
        return _silent_defaults()

    if float(np.sqrt(np.mean(y ** 2))) < 0.001:
        return _silent_defaults()

    features = {"audio_present": True}
    features.update(_pitch_features(y, sr))
    features.update(_silence_features(y, sr))
    features.update(_harmonic_features(y, sr))
    features.update(_spectral_features(y, sr))
    features.update(_rhythm_features(y, sr))
    features["speech_detected"] = _detect_speech(features)

    return features


def _extract_audio(video_path: str, audio_path: str) -> bool:
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "1", audio_path,
        "-y", "-loglevel", "quiet"
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return (os.path.exists(audio_path) and
            os.path.getsize(audio_path) > 1000)


def _pitch_features(y, sr):
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=PITCH_MIN_HZ, fmax=PITCH_MAX_HZ,
            frame_length=2048, hop_length=512
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)] \
            if f0 is not None else np.array([])
        if len(voiced_f0) > 10:
            return {
                "pitch_std_hz":   float(np.std(voiced_f0)),
                "pitch_range_hz": float(np.ptp(voiced_f0)),
                "pitch_mean_hz":  float(np.mean(voiced_f0)),
                "voiced_ratio":   float(np.mean(voiced_flag)),
            }
    except Exception:
        pass
    return {
        "pitch_std_hz": 0.0, "pitch_range_hz": 0.0,
        "pitch_mean_hz": 0.0, "voiced_ratio": 0.0,
    }


def _silence_features(y, sr):
    try:
        rms = librosa.feature.rms(
            y=y, frame_length=2048, hop_length=512
        )[0]
        return {
            "silence_ratio": float(np.mean(rms < SILENCE_THRESHOLD))
        }
    except Exception:
        return {"silence_ratio": 0.0}


def _harmonic_features(y, sr):
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        he    = float(np.mean(y_harm ** 2))
        pe    = float(np.mean(y_perc ** 2))
        total = he + pe + 1e-10
        return {
            "harmonic_ratio":    float(he / total * 10),
            "percussive_energy": float(pe),
        }
    except Exception:
        return {"harmonic_ratio": 0.0, "percussive_energy": 0.0}


def _spectral_features(y, sr):
    try:
        spec  = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr)
        return {
            "spectral_bandwidth_mean": float(np.mean(
                librosa.feature.spectral_bandwidth(
                    S=spec, sr=sr, freq=freqs))),
            "spectral_rolloff_mean": float(np.mean(
                librosa.feature.spectral_rolloff(
                    S=spec, sr=sr, roll_percent=0.85))),
            "spectral_flatness_mean": float(np.mean(
                librosa.feature.spectral_flatness(S=spec))),
            "spectral_flatness_std": float(np.std(
                librosa.feature.spectral_flatness(S=spec))),
        }
    except Exception:
        return {
            "spectral_bandwidth_mean": 0.0,
            "spectral_rolloff_mean":   0.0,
            "spectral_flatness_mean":  0.0,
            "spectral_flatness_std":   0.0,
        }


def _rhythm_features(y, sr):
    try:
        oe     = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=oe, sr=sr
        )
        bpm = float(
            tempo[0] if hasattr(tempo, "__len__") else tempo
        )
        reg = float(
            1 - np.std(np.diff(beats)) /
            (np.mean(np.diff(beats)) + 1e-8)
        ) if len(beats) > 2 else 0.0
        return {"tempo_bpm": bpm, "tempo_regularity": reg}
    except Exception:
        return {"tempo_bpm": 0.0, "tempo_regularity": 0.0}


def _detect_speech(features: dict) -> bool:
    return (features.get("voiced_ratio", 0) > 0.1 and
            features.get("pitch_std_hz", 0) > 5)


def _silent_defaults() -> dict:
    return {
        "audio_present":           False,
        "pitch_std_hz":            0.0,
        "pitch_range_hz":          0.0,
        "pitch_mean_hz":           0.0,
        "voiced_ratio":            0.0,
        "silence_ratio":           1.0,
        "harmonic_ratio":          0.0,
        "percussive_energy":       0.0,
        "spectral_bandwidth_mean": 0.0,
        "spectral_rolloff_mean":   0.0,
        "spectral_flatness_mean":  0.0,
        "spectral_flatness_std":   0.0,
        "tempo_bpm":               0.0,
        "tempo_regularity":        0.0,
        "speech_detected":         False,
    }