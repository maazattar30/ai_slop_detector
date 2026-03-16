"""
downloader.py — Production version
Handles YouTube download + metadata + smart clipping
"""

import os
import json
import subprocess
import shutil
from typing import Optional

from config import TEMP_DIR

YT_DLP_CLIENTS   = ["web", "android", "ios"]
MIN_FILE_SIZE_KB  = 100
SHORT_VIDEO_MAX   = 180
FORMAT_STRING     = (
    "bestvideo[ext=mp4][height<=720]"
    "+bestaudio[ext=m4a]"
    "/best[ext=mp4]"
    "/best"
)


def get_video_info(url: str) -> Optional[dict]:
    """Fetch video metadata without downloading."""
    cmd = [
        "yt-dlp", "--dump-json",
        "--no-download", "--quiet", url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    return {
        "title":               raw.get("title", ""),
        "duration":            float(raw.get("duration") or 0),
        "channel":             raw.get("channel", ""),
        "channel_id":          raw.get("channel_id", ""),
        "upload_date":         raw.get("upload_date", ""),
        "view_count":          raw.get("view_count"),
        "like_count":          raw.get("like_count"),
        "comment_count":       raw.get("comment_count"),
        "subscriber_count":    raw.get("channel_follower_count"),
        "description":         raw.get("description", ""),
        "tags":                raw.get("tags", []),
        "categories":          raw.get("categories", []),
        "automatic_captions":  _has_auto_captions(raw),
        "thumbnail":           raw.get("thumbnail", ""),
        "webpage_url":         raw.get("webpage_url", url),
        "is_live":             raw.get("is_live", False),
        "channel_url":         raw.get("channel_url", ""),
    }


def download_video(
    url:      str,
    out_path: str,
    duration: Optional[float] = None
) -> bool:
    """Download video and apply smart clipping if needed."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        os.remove(out_path)

    raw_path = out_path.replace(".mp4", "_raw.mp4")
    success  = _try_download(url, raw_path)

    if not success:
        return False

    if duration is None or duration <= SHORT_VIDEO_MAX:
        shutil.move(raw_path, out_path)
        return True

    clipped = _smart_clip(raw_path, out_path, duration)
    if os.path.exists(raw_path):
        os.remove(raw_path)

    return clipped


def probe_video(path: str) -> dict:
    """Get technical metadata from local video."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}

    raw          = json.loads(result.stdout)
    video_stream = next(
        (s for s in raw.get("streams", [])
         if s.get("codec_type") == "video"), {}
    )
    audio_stream = next(
        (s for s in raw.get("streams", [])
         if s.get("codec_type") == "audio"), {}
    )
    fmt = raw.get("format", {})

    fps = 0.0
    try:
        num, den = video_stream.get(
            "avg_frame_rate", "0/1"
        ).split("/")
        fps = float(num) / float(den) if float(den) > 0 else 0.0
    except Exception:
        pass

    return {
        "duration":    float(fmt.get("duration", 0)),
        "width":       int(video_stream.get("width", 0)),
        "height":      int(video_stream.get("height", 0)),
        "fps":         round(fps, 2),
        "codec":       video_stream.get("codec_name", ""),
        "audio_codec": audio_stream.get("codec_name", ""),
        "bitrate":     int(fmt.get("bit_rate", 0)),
        "file_size":   int(fmt.get("size", 0)),
    }


def _try_download(url: str, out_path: str) -> bool:
    for client in YT_DLP_CLIENTS:
        cmd = [
            "yt-dlp",
            "--extractor-args", f"youtube:player_client={client}",
            "-f", FORMAT_STRING,
            "--merge-output-format", "mp4",
            "-o", out_path,
            "--no-playlist", "--quiet", url
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        if (os.path.exists(out_path) and
                os.path.getsize(out_path) > MIN_FILE_SIZE_KB * 1024):
            return True
    return False


def _smart_clip(
    input_path:  str,
    output_path: str,
    duration:    float
) -> bool:
    segments   = _calculate_segments(duration)
    temp_files = []
    concat_list = input_path.replace(".mp4", "_concat.txt").replace("_raw", "")

    try:
        for i, seg in enumerate(segments):
            seg_path = input_path.replace(
                ".mp4", f"_seg{i}.mp4"
            ).replace("_raw", "")
            cmd = [
                "ffmpeg",
                "-ss", str(seg["start"]),
                "-i", input_path,
                "-t", str(seg["duration"]),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                seg_path, "-y", "-loglevel", "quiet"
            ]
            subprocess.run(cmd, capture_output=True, text=True)
            if (os.path.exists(seg_path) and
                    os.path.getsize(seg_path) > 1000):
                temp_files.append(seg_path)

        if not temp_files:
            return False

        if len(temp_files) == 1:
            shutil.move(temp_files[0], output_path)
            return True

        with open(concat_list, "w") as f:
            for fp in temp_files:
                f.write(f"file '{fp}'\n")

        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", concat_list, "-c", "copy",
            output_path, "-y", "-loglevel", "quiet"
        ]
        subprocess.run(cmd, capture_output=True, text=True)

        return (os.path.exists(output_path) and
                os.path.getsize(output_path) > 1000)

    finally:
        for fp in temp_files:
            if os.path.exists(fp):
                os.remove(fp)
        if os.path.exists(concat_list):
            os.remove(concat_list)


def _calculate_segments(duration: float) -> list:
    seg1_dur   = min(90, duration * 0.4)
    seg2_dur   = min(60, duration * 0.2)
    seg3_dur   = min(30, duration * 0.1)
    seg2_start = (duration / 2) - (seg2_dur / 2)
    seg3_start = duration - seg3_dur
    return [
        {"start": 0,           "duration": seg1_dur},
        {"start": seg2_start,  "duration": seg2_dur},
        {"start": seg3_start,  "duration": seg3_dur},
    ]


def _has_auto_captions(raw: dict) -> bool:
    return bool(raw.get("automatic_captions")) and \
           not bool(raw.get("subtitles"))


def _probe_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(
            json.loads(result.stdout)["format"]["duration"]
        )
    except Exception:
        return 0.0