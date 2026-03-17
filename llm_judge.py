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
1. A structured signal brief from automated modules (reference only)
2. A module-computed AI score (treat as ONE data point, NOT ground truth)
3. Video frames as grid images (each grid contains multiple frames)
4. Video title and description

YOUR TASK — compute your OWN independent ai_score (0-100):
- 0-40  = likely real/human
- 41-64 = uncertain, mixed signals
- 65-100 = likely AI generated

Scoring rules:
- Visual inspection of frames is your PRIMARY evidence
- Signal brief is SECONDARY supporting evidence
- DO NOT echo back the module score as your ai_score
- If frames clearly show AI-generated content (animation, CGI faces, text-to-video artifacts) score 70-95
- If frames clearly show real human/camera footage score 10-35
- If uncertain score 45-55

Look for: natural eye movement, skin texture, motion physics,
audio-visual coherence, lighting consistency, background stability,
animated or synthetic faces, AI art style, text-to-video artifacts.

When uncertain, assign Bucket 2.

Known caveats:
- Heavy CGI in real films (Avatar, trailers) are NOT AI generated content
- High-production studios can score high on AI signals visually
- Deepfakes show real body but AI face, look for skin/boundary artifacts
- Animated fruits/animals/objects with synthetic AI faces = Bucket 3

CRITICAL: Your bucket MUST match your ai_score:
- bucket 1 requires ai_score 0-40
- bucket 2 requires ai_score 41-64
- bucket 3 requires ai_score 65-100

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
    if not GROQ_API_KEY:
        return _fallback(module_score, "No GROQ_API_KEY set")

    try:
        client = Groq(api_key=GROQ_API_KEY)

        user_text = (
            f"VIDEO ANALYSIS REQUEST\n\n"
            f"Title: {title}\n"
            f"Description: {description or '(none)'}\n"
            f"Speech detected: {speech_detected}\n\n"
            f"MODULE SCORE (reference only, do NOT copy this as your ai_score): "
            f"{module_score:.1f} / 100\n"
            f"(0-40=likely real, 40-65=uncertain, 65+=likely AI)\n\n"
            f"SIGNAL EVIDENCE BRIEF:\n{evidence_brief}\n\n"
            f"Examine the {len(llm_frames)} grid image(s) attached. "
            f"Each grid contains multiple frames sampled across the video. "
            f"Use what you SEE in the frames as your primary evidence to "
            f"compute your own independent ai_score."
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
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
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

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw.strip())
        result = _fix_bucket_score_consistency(result)
        return result

    except json.JSONDecodeError:
        return _fallback(module_score, "JSON parse failed")
    except Exception as e:
        return _fallback(module_score, str(e))


def _fix_bucket_score_consistency(result: dict) -> dict:
    """
    Ensure bucket and ai_score are consistent.
    The bucket (visual judgment) is trusted over the score number.
    """
    bucket = result.get("bucket", 2)
    score  = result.get("ai_score", 50)

    if bucket == 3 and score < 65:
        print(f"[llm_judge] Fixing: bucket=3 but ai_score={score} → 80")
        result["ai_score"] = 80
    elif bucket == 1 and score > 40:
        print(f"[llm_judge] Fixing: bucket=1 but ai_score={score} → 20")
        result["ai_score"] = 20
    elif bucket == 2 and (score < 41 or score > 64):
        print(f"[llm_judge] Fixing: bucket=2 but ai_score={score} → 52")
        result["ai_score"] = 52

    return result


def _encode_image(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


def _fallback(module_score: float, reason: str) -> dict:
    bucket = 3 if module_score > 65 else (1 if module_score < 40 else 2)
    return {
        "bucket":          bucket,
        "ai_score":        int(module_score),
        "confidence":      0.5,
        "bucket_name":     BUCKETS[bucket]["name"],
        "primary_signals": ["module_score"],
        "visual_evidence": [f"LLM unavailable: {reason}"],
        "module_override": False,
        "override_reason": "",
        "explanation":     f"Module score: {module_score:.1f}/100. LLM analysis unavailable: {reason}",
        "red_flags":       [],
    }