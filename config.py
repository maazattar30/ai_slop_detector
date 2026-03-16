"""
config.py
─────────────────────────────────────────────────────────────────
Production config for HF Space deployment.
Reads secrets from HF environment variables.
─────────────────────────────────────────────────────────────────
"""

import os

# ─────────────────────────────────────────────
# SECRETS — set these in HF Space settings
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PASSWORD     = os.getenv("APP_PASSWORD", "")

# ─────────────────────────────────────────────
# PATHS — all temp, nothing persistent on HF
# ─────────────────────────────────────────────
TEMP_DIR = "/tmp/ai_slop_pipeline"

# ─────────────────────────────────────────────
# FRAME EXTRACTION — HYBRID FFMPEG
# ─────────────────────────────────────────────
FRAME_RESOLUTION  = 1024
LLM_FRAME_WIDTH   = 768
LLM_FRAME_COUNT   = 5
LLM_FRAMES_PER_IMAGE = 4    # frames per grid image
MAX_FRAMES_CAP    = 120

SAMPLING_RATES = {
    "short":  {"max_duration": 120,   "fps": 1.0},
    "medium": {"max_duration": 300,   "fps": 0.5},
    "long":   {"max_duration": 900,   "fps": 0.25},
    "vlong":  {"max_duration": 99999, "fps": 0.1},
}

# ─────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────
AUDIO_SAMPLE_RATE  = 22050
SILENCE_THRESHOLD  = 0.01
PITCH_MIN_HZ       = 50
PITCH_MAX_HZ       = 600

# ─────────────────────────────────────────────
# SIGLIP — disabled on CPU Basic
# ─────────────────────────────────────────────
SIGLIP_ENABLED    = False   # flip to True when GPU upgraded
SIGLIP_MODEL      = "google/siglip-base-patch16-224"
SIGLIP_AI_PROMPTS = [
    "photorealistic AI generated image",
    "AI generated digital art",
    "synthetic computer generated video frame"
]
SIGLIP_REAL_PROMPTS = [
    "real photograph",
    "authentic video frame from a camera",
    "genuine human filmed video"
]

# ─────────────────────────────────────────────
# LLM JUDGE
# ─────────────────────────────────────────────
GROQ_MODEL      = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_GROQ_IMAGES = 5
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 1500

# ─────────────────────────────────────────────
# CALIBRATION — from 20 video experiment
# Format: (real_mean, ai_mean, higher_is_ai, cohens_d)
# ─────────────────────────────────────────────
CALIBRATION = {
    "siglip_ai_mean":            (0.640, 0.889, True,  1.52),
    "raft_temporal_entropy":     (1.515, 1.964, True,  0.89),
    "chromatic_aberration":      (7.848, 5.812, False, 0.81),
    "saturation_cv":             (0.850, 0.586, False, 0.82),
    "harmonic_ratio":            (2.041, 1.004, False, 0.99),
    "compression_artifact_mean": (1.411, 1.021, False, 0.84),
    "spectral_bandwidth_mean":   (1681,  1262,  False, 0.80),
    "silence_ratio":             (0.102, 0.301, True,  0.69),
    "tempo_regularity":          (0.951, 0.754, False, 0.70),
    "motion_delta_cv":           (0.577, 0.738, True,  0.54),
    "pitch_std_hz":              (142.6, 69.5,  False, 0.73),
    "pitch_range_hz":            (911.3, 341.6, False, 1.16),
    "channel_age_days":          (365,   15,    False, 0.80),
    "subscriber_count":          (10000, 100,   False, 0.75),
}

# ─────────────────────────────────────────────
# SCORING & BUCKETS
# ─────────────────────────────────────────────
MODULE_WEIGHT = 0.55
LLM_WEIGHT    = 0.45

BUCKET_THRESHOLDS = {
    "B1_max_score":  40,
    "B2_max_score":  65,
    "B3_min_score":  65,
    "confidence_min": 0.60,
}

BUCKETS = {
    1: {
        "name":        "Human Only",
        "description": "Real camera, real people, no AI in pipeline",
        "color":       "#16A34A",
        "emoji":       "🟢"
    },
    2: {
        "name":        "Human + AI Tools",
        "description": "Human creative control with AI assistance. Also uncertain cases.",
        "color":       "#2563EB",
        "emoji":       "🔵"
    },
    3: {
        "name":        "AI Generated",
        "description": "Core content is machine generated",
        "color":       "#DC2626",
        "emoji":       "🔴"
    },
}

# ─────────────────────────────────────────────
# TEXT / METADATA KEYWORDS
# ─────────────────────────────────────────────
AI_KEYWORDS = [
    "ai", "artificial intelligence", "generated", "sora", "runway",
    "gen-3", "gen3", "midjourney", "kling", "pika", "luma", "haiper",
    "synthesia", "deepfake", "text to video", "text-to-video",
    "stable diffusion", "dall-e", "dalle", "chatgpt", "gpt",
    "machine learning", "neural network"
]

SLOP_KEYWORDS = [
    "viral", "mind-blowing", "mind blowing", "shocking", "unbelievable",
    "amazing", "must watch", "insane", "crazy", "epic", "incredible",
    "you wont believe", "you won't believe", "wait for it",
    "watch till end", "subscribe"
]