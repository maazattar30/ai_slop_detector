"""
pipeline.py — Production orchestrator
Runs all modules in sequence, handles errors gracefully
"""

import os
import shutil
import traceback
from config import (
    TEMP_DIR,
    MODULE_WEIGHT,
    LLM_WEIGHT,
    BUCKETS,
)
from modules.downloader       import get_video_info, download_video, probe_video
from modules.frame_extractor  import extract_frames, cleanup_frames
from modules.audio_features   import extract_audio_features
from modules.texture_features import extract_texture_features
from modules.motion_features  import extract_motion_features
from modules.optical_flow     import extract_optical_flow_features
from modules.siglip_classifier import extract_siglip_features
from modules.metadata_features import extract_metadata_features
from modules.text_features    import extract_text_features
from modules.evidence_builder import build_evidence, assign_bucket
from llm_judge                import run_llm_judge


def run_pipeline(
    url:         str,
    title:       str = "",
    description: str = "",
    progress_cb  = None,
) -> dict:
    """
    Full pipeline from YouTube URL to classification result.

    Args:
        url         : YouTube URL
        title       : video title (optional — fetched if empty)
        description : video description (optional)
        progress_cb : optional callback(step, total, message)
                      for Gradio progress updates

    Returns complete result dict for frontend rendering.
    """

    def progress(step, total, msg):
        print(f"  [{step}/{total}] {msg}")
        if progress_cb:
            progress_cb(step / total, msg)

    work_dir = os.path.join(TEMP_DIR, _safe_id(url))
    os.makedirs(work_dir, exist_ok=True)

    result = {
        "url":         url,
        "title":       title,
        "description": description,
        "error":       None,
        "module_score":  50.0,
        "llm_score":     50.0,
        "final_score":   50.0,
        "bucket":        2,
        "bucket_name":   "Human + AI Tools",
        "confidence":    0.5,
        "signal_rows":   [],
        "llm_result":    {},
        "llm_frames":    [],
        "video_info":    {},
    }

    try:
        total_steps = 11

        # ── Step 1: Metadata ──────────────────────────────────
        progress(1, total_steps, "Fetching video metadata...")
        video_info = get_video_info(url)

        if video_info:
            result["video_info"] = video_info
            if not title:
                title       = video_info.get("title", "")
                result["title"] = title
            if not description:
                description = video_info.get("description", "")
                result["description"] = description

        duration = video_info.get("duration", 0) if video_info else 0

        # ── Step 2: Download ──────────────────────────────────
        progress(2, total_steps, "Downloading video...")
        video_path = os.path.join(work_dir, "video.mp4")
        ok = download_video(url, video_path, duration)

        if not ok:
            result["error"] = "Video download failed"
            return result

        probe = probe_video(video_path)
        duration = probe.get("duration", duration)

        # ── Step 3: Frame extraction ──────────────────────────
        progress(3, total_steps, "Extracting frames (hybrid FFmpeg)...")
        frame_result = extract_frames(video_path, work_dir, duration)
        all_frames   = frame_result["all_frames"]
        llm_frames   = frame_result["llm_frames"]
        result["llm_frames"] = llm_frames

        # ── Step 4: Audio features ────────────────────────────
        progress(4, total_steps, "Analyzing audio...")
        audio_feats = extract_audio_features(video_path, work_dir)

        # ── Step 5: Texture features ──────────────────────────
        progress(5, total_steps, "Analyzing texture & compression...")
        texture_feats = extract_texture_features(all_frames)

        # ── Step 6: Motion features ───────────────────────────
        progress(6, total_steps, "Analyzing motion patterns...")
        motion_feats = extract_motion_features(all_frames)

        # ── Step 7: Optical flow ──────────────────────────────
        progress(7, total_steps, "Computing optical flow...")
        flow_feats = extract_optical_flow_features(all_frames)

        # ── Step 8: SigLIP ────────────────────────────────────
        progress(8, total_steps, "Running SigLIP classifier...")
        siglip_feats = extract_siglip_features(all_frames)

        # ── Step 9: Metadata + Text features ─────────────────
        progress(9, total_steps, "Computing metadata & text signals...")
        meta_feats = extract_metadata_features(
            video_info or {}, title, description
        )
        text_feats = extract_text_features(title, description)

        # ── Step 10: Evidence builder ─────────────────────────
        progress(10, total_steps, "Building evidence brief...")
        all_feats = {
            **audio_feats,
            **texture_feats,
            **motion_feats,
            **flow_feats,
            **siglip_feats,
            **meta_feats,
            **text_feats,
        }

        speech   = audio_feats.get("speech_detected", False)
        evidence = build_evidence(all_feats, speech)

        result["module_score"] = evidence["module_score"]
        result["signal_rows"]  = evidence["signal_rows"]

        # ── Step 11: LLM Judge ────────────────────────────────
        progress(11, total_steps, "Calling LLM judge (Groq Llama 4)...")
        llm_result = run_llm_judge(
            title           = title,
            description     = description,
            module_score    = evidence["module_score"],
            evidence_brief  = evidence["evidence_brief"],
            llm_frames      = llm_frames,
            speech_detected = speech,
        )

        result["llm_result"] = llm_result
        result["llm_score"]  = llm_result.get(
            "ai_score", evidence["module_score"]
        )

        # ── Final score + bucket ──────────────────────────────
        result["confidence"]  = llm_result.get("confidence", 0.5)

        # If LLM overrides module, trust LLM bucket directly
        if llm_result.get("module_override"):
            print(f"  [pipeline] LLM override active — using LLM bucket {llm_result.get('bucket')}")
            result["bucket"]      = llm_result.get("bucket", 2)
            result["bucket_name"] = BUCKETS[result["bucket"]]["name"]
            result["final_score"] = round(result["llm_score"], 1)
        else:
            result["final_score"] = round(
                MODULE_WEIGHT * result["module_score"] +
                LLM_WEIGHT    * result["llm_score"],
                1
            )
            result["bucket"]      = assign_bucket(
                result["final_score"],
                result["confidence"]
            )
            result["bucket_name"] = BUCKETS[result["bucket"]]["name"]

    except Exception as e:
        result["error"] = str(e)
        print(f"  ❌ Pipeline error: {traceback.format_exc()}")

    finally:
        # Clean up frames — keep video for now
        cleanup_frames(work_dir)

    return result


def _safe_id(url: str) -> str:
    """Create a safe directory name from URL."""
    import hashlib
    return hashlib.md5(url.encode()).hexdigest()[:12]