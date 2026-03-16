import json
import base64
import io
from PIL import Image
from groq import Groq
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    BUCKETS,
)

SYSTEM_PROMPT = """You are an expert AI video forensics analyst.
Classify YouTube videos into one of 3 buckets:
- Bucket 1 (Human Only): Real camera, real people, no AI in production
- Bucket 2 (Human + AI Tools): Human creative control, AI used as assistant. Also use this when uncertain.
- Bucket 3 (AI Generated): Core content is machine generated — text-to-video, TTS, deepfake, AI avatar
You will receive:
1. A structured signal brief from automated modules
2. A module-computed AI score (0-100)
3. Video frames as grid images (each grid contains multiple frames)
4. Video title and description
Your task:
- Use signal brief as primary quantitative evidence
- Use frames to visually confirm or override signals
- Look for: natural eye movement, skin texture, motion physics,
  audio-visual coherence, lighting consistency, background stability
- When uncertain → assign Bucket 2
Known caveats:
- Heavy CGI in real films (Avatar, trailers) → NOT AI generated content
- High-production studios can score high on AI signals visually
- Deepfakes show real body but AI face — look for skin/boundary artifacts
Respond ONLY with valid JSON, no text before or after:
{
  "bucket": <1|2|3>,
  "ai_score": <0-100>,
  "confidence": <0.0-1.0>,
  "bucket_name": "<Human Only|Human + AI Tools|AI Generated>",
  "primary_signals": ["<signal 1>", "<signal 2>", "<signal 3>"],
  "visual_evidence": ["<what you saw in frames>"],
  "module_override": <true|false>,
  "override_reason": "<why overrode or empty string>",
  "explanation": "<2-4 sentence plain English for client>",
  "red_flags": ["<specific AI artifacts noticed>"]
}"""


def run_llm_judge(
    title:          str,
    description:    str,
    module_score:   float,
    evidence_brief: str,
    llm_frames:     list,
    speech_detected: bool,
) -> dict:
    """
    Send evidence brief + grid frames to Groq Llama 4 Scout.
    Returns structured verdict dict.
    """
    if not GROQ_API_KEY:
        return _fallback(module_score, "No GROQ_API_KEY set")

    try:
        client = Groq(api_key=GROQ_API_KEY)

        user_text = (
            f"VIDEO ANALYSIS REQUEST\n\n"
            f"Title: {title}\n"
            f"Description: {description or '(none)'}\n"
            f"Speech detected: {speech_detected}\n\n"
            f"MODULE SCORE: {module_score:.1f} / 100\n"
            f"(0-40=likely real, 40-65=uncertain, 65+=likely AI)\n\n"
            f"SIGNAL EVIDENCE BRIEF:\n{evidence_brief}\n\n"
            f"Examine the {len(llm_frames)} grid image(s) attached. "
            f"Each grid contains multiple frames sampled across the video."
        )

        content = [{"type": "text", "text": user_text}]

        for i, grid_path in enumerate(llm_frames):
            b64 = _encode_image(grid_path)
            if b64:
                content.append({
                    "type": "text",
                    "text": f"Grid {i + 1} of {len(llm_frames)}:"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": content}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    except json.JSONDecodeError:
        return _fallback(module_score, "JSON parse failed")
    except Exception as e:
        return _fallback(module_score, str(e))


def _encode_image(path: str) -> str:
    """Encode image file as base64 string."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


def _fallback(module_score: float, reason: str) -> dict:
    """Safe fallback if LLM call fails."""
    bucket = 3 if module_score > 65 else (
        1 if module_score < 40 else 2
    )
    return {
        "bucket":          bucket,
        "ai_score":        int(module_score),
        "confidence":      0.5,
        "bucket_name":     BUCKETS[bucket]["name"],
        "primary_signals": ["module_score"],
        "visual_evidence": [f"LLM unavailable: {reason}"],
        "module_override": False,
        "override_reason": "",
        "explanation":     (
            f"Module score: {module_score:.1f}/100. "
            f"LLM analysis unavailable: {reason}"
        ),
        "red_flags":       [],
    }
