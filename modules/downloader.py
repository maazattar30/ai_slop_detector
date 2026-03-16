"""
downloader.py — Production version
Handles YouTube download + metadata + smart clipping
"""

import os
import json
import subprocess
import shutil
import shlex
from typing import Optional

from config import TEMP_DIR

# tv_embedded is most reliable on datacenter/HF IPs
YT_DLP_CLIENTS   = ["tv_embedded", "ios", "android", "web"]
MIN_FILE_SIZE_KB  = 100
SHORT_VIDEO_MAX   = 180
FORMAT_STRING     = (
    "bestvideo[ext=mp4][height<=720]"
    "+bestaudio[ext=m4a]"
    "/best[ext=mp4]"
    "/best"
)

FFMPEG_PATH = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"


def get_video_info(url: str) -> Optional[dict]:
    """Fetch video metadata without downloading."""
    cmd = [
        "yt-dlp", "--dump-json",
        "--no-download",
        "--extractor-args", "youtube:player_client=tv_embedded",
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[get_video_info] yt-dlp stderr: {result.stderr[:500]}")
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
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

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
            "--ffmpeg-location", FFMPEG_PATH,
            "-f", FORMAT_STRING,
            "--merge-output-format", "mp4",
            "-o", out_path,
            "--no-playlist",
            "--retries", "3",
            "--fragment-retries", "3",
            url
        ]
        print(f"[_try_download] Trying client={client}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[_try_download] client={client} failed: {result.stderr[:300]}")
        
        if (os.path.exists(out_path) and
                os.path.getsize(out_path) > MIN_FILE_SIZE_KB * 1024):
            print(f"[_try_download] Success with client={client}")
            return True

    return False