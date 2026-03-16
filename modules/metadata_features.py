"""
metadata_features.py — Production version
Channel and video metadata signals
"""

from datetime import datetime
from config import AI_KEYWORDS, SLOP_KEYWORDS


def extract_metadata_features(
    video_info: dict,
    title:      str = "",
    description: str = "",
) -> dict:
    """
    Metadata-based signals from yt-dlp video info.

    channel_age_days     — new channels are suspicious
    subscriber_count     — very low = suspicious
    view_like_ratio      — abnormally high views/likes = bots
    auto_captions_only   — no manual captions = no real speaker
    upload_date_recent   — very recent upload
    has_description      — empty description = content farm
    comment_count_zero   — disabled comments = hiding engagement
    """
    if not video_info:
        return _neutral_defaults()

    features = {}

    # Channel age
    upload_date = video_info.get("upload_date", "")
    if upload_date and len(upload_date) == 8:
        try:
            uploaded = datetime.strptime(upload_date, "%Y%m%d")
            features["channel_age_days"] = max(
                (datetime.now() - uploaded).days, 0
            )
        except Exception:
            features["channel_age_days"] = 365
    else:
        features["channel_age_days"] = 365

    # Subscriber count
    subs = video_info.get("subscriber_count")
    features["subscriber_count"] = float(subs) if subs else 0.0

    # View / like ratio
    views = video_info.get("view_count") or 0
    likes = video_info.get("like_count") or 0
    features["view_like_ratio"] = (
        float(views) / float(likes)
        if likes and likes > 0 else 0.0
    )

    # Auto captions only
    features["auto_captions_only"] = float(
        video_info.get("automatic_captions", False)
    )

    # Has description
    desc = video_info.get("description", "") or description
    features["has_description"] = float(bool(desc and len(desc) > 20))

    # Comment count
    comments = video_info.get("comment_count")
    features["comment_count_zero"] = float(
        comments is None or comments == 0
    )

    # Tag count
    tags = video_info.get("tags", [])
    features["tag_count"] = float(len(tags))

    return features


def _neutral_defaults() -> dict:
    return {
        "channel_age_days":   365.0,
        "subscriber_count":   0.0,
        "view_like_ratio":    0.0,
        "auto_captions_only": 0.0,
        "has_description":    1.0,
        "comment_count_zero": 0.0,
        "tag_count":          0.0,
    }