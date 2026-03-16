"""
frame_extractor.py — Production version
Hybrid FFmpeg frame extraction with grid LLM frames
"""

import os
import subprocess
import shutil
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

from config import (
    FRAME_RESOLUTION,
    LLM_FRAME_WIDTH,
    LLM_FRAME_COUNT,
    LLM_FRAMES_PER_IMAGE,
    MAX_FRAMES_CAP,
    SAMPLING_RATES,
)


def extract_frames(
    video_path: str,
    work_dir:   str,
    duration:   Optional[float] = None,
) -> dict:
    """Extract analysis frames and LLM grid frames."""
    if duration is None:
        duration = _probe_duration(video_path)

    all_frames_dir = os.path.join(work_dir, "all_frames")
    llm_frames_dir = os.path.join(work_dir, "llm_frames")
    os.makedirs(all_frames_dir, exist_ok=True)
    os.makedirs(llm_frames_dir, exist_ok=True)

    fps        = _get_sampling_rate(duration)
    all_frames = _extract_all_frames(video_path, all_frames_dir, fps)

    if len(all_frames) > MAX_FRAMES_CAP:
        all_frames = _subsample_frames(all_frames, MAX_FRAMES_CAP)

    all_frames = _motion_adaptive_densify(
        video_path, all_frames_dir, all_frames, fps
    )

    llm_grids = _extract_llm_grid_frames(all_frames, llm_frames_dir)

    return {
        "all_frames":  all_frames,
        "llm_frames":  llm_grids,
        "fps_used":    fps,
        "frame_count": len(all_frames),
        "work_dir":    work_dir,
    }


def cleanup_frames(work_dir: str) -> None:
    for subdir in ["all_frames", "llm_frames"]:
        path = os.path.join(work_dir, subdir)
        if os.path.exists(path):
            shutil.rmtree(path)


def _get_sampling_rate(duration: float) -> float:
    for tier in SAMPLING_RATES.values():
        if duration <= tier["max_duration"]:
            return tier["fps"]
    return 0.1


def _extract_all_frames(
    video_path: str,
    out_dir:    str,
    fps:        float
) -> list:
    for f in os.listdir(out_dir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(out_dir, f))

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps},scale={FRAME_RESOLUTION}:-1",
        "-q:v", "2",
        os.path.join(out_dir, "frame_%04d.jpg"),
        "-y", "-loglevel", "quiet"
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    return sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith(".jpg")
    ])


def _subsample_frames(frames: list, cap: int) -> list:
    indices = np.linspace(0, len(frames) - 1, cap, dtype=int)
    return [frames[i] for i in indices]


def _motion_adaptive_densify(
    video_path:  str,
    out_dir:     str,
    frames:      list,
    base_fps:    float,
) -> list:
    if len(frames) < 4:
        return frames

    deltas    = []
    prev_gray = None

    for fpath in frames:
        img = cv2.imread(fpath)
        if img is None:
            deltas.append(0.0)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        if prev_gray is not None:
            deltas.append(float(np.mean(np.abs(gray - prev_gray))))
        else:
            deltas.append(0.0)
        prev_gray = gray

    mean_delta  = float(np.mean(deltas))
    threshold   = mean_delta * 1.5
    high_motion = [i for i, d in enumerate(deltas) if d > threshold]
    dense_fps   = min(base_fps * 2, 2.0)

    for idx in high_motion:
        time_sec  = idx / base_fps
        start     = max(0, time_sec - 1.0)
        dense_out = os.path.join(out_dir, f"dense_{idx:04d}_%02d.jpg")
        cmd = [
            "ffmpeg", "-ss", str(start), "-i", video_path,
            "-t", "2.0",
            "-vf", f"fps={dense_fps},scale={FRAME_RESOLUTION}:-1",
            "-q:v", "2", dense_out, "-y", "-loglevel", "quiet"
        ]
        subprocess.run(cmd, capture_output=True, text=True)

    all_frames = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith(".jpg")
    ])

    if len(all_frames) > MAX_FRAMES_CAP:
        all_frames = _subsample_frames(all_frames, MAX_FRAMES_CAP)

    return all_frames


def _extract_llm_grid_frames(
    all_frames: list,
    llm_dir:    str,
) -> list:
    """
    Stitch frames into grid images for LLM.
    Each grid = LLM_FRAMES_PER_IMAGE frames in NxN layout.
    Returns list of grid image paths.
    """
    n          = len(all_frames)
    fpg        = LLM_FRAMES_PER_IMAGE
    cols       = math.ceil(math.sqrt(fpg))
    rows       = math.ceil(fpg / cols)
    total_slots = LLM_FRAME_COUNT * fpg

    # Pick evenly spaced frames across full video
    indices = np.linspace(0, n - 1, total_slots, dtype=int)
    chunks  = [
        indices[i:i + fpg].tolist()
        for i in range(0, total_slots, fpg)
    ]

    grid_paths = []

    for grid_idx, chunk in enumerate(chunks):
        cell_w = LLM_FRAME_WIDTH // cols
        cell_h = int(cell_w * 9 / 16)   # 16:9 aspect
        grid   = Image.new(
            "RGB",
            (cell_w * cols, cell_h * rows),
            (20, 20, 20)
        )
        draw = ImageDraw.Draw(grid)

        for pos, frame_idx in enumerate(chunk):
            img = Image.open(
                all_frames[frame_idx]
            ).convert("RGB").resize(
                (cell_w, cell_h), Image.LANCZOS
            )
            r, c = divmod(pos, cols)
            grid.paste(img, (c * cell_w, r * cell_h))

            # Frame label
            label = f"F{frame_idx + 1}"
            draw.rectangle(
                [c * cell_w, r * cell_h,
                 c * cell_w + 30, r * cell_h + 16],
                fill=(0, 0, 0)
            )
            draw.text(
                (c * cell_w + 2, r * cell_h + 2),
                label, fill=(255, 255, 0)
            )

        dst = os.path.join(llm_dir, f"grid_{grid_idx:02d}.jpg")
        grid.save(dst, "JPEG", quality=85)
        grid_paths.append(dst)

    return grid_paths


def _probe_duration(path: str) -> float:
    import json
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(
            json.loads(result.stdout)["format"]["duration"]
        )
    except Exception:
        return 0.0