"""
siglip_classifier.py — Production version
Disabled on CPU Basic — returns neutral defaults
Flip SIGLIP_ENABLED in config when GPU upgraded
"""

import numpy as np
from config import (
    SIGLIP_ENABLED,
    SIGLIP_MODEL,
    SIGLIP_AI_PROMPTS,
    SIGLIP_REAL_PROMPTS,
)


def extract_siglip_features(all_frames: list) -> dict:
    """
    SigLIP zero-shot AI probability per frame.
    Returns neutral 0.5 when disabled (CPU mode).
    """
    if not SIGLIP_ENABLED:
        return {
            "siglip_ai_mean":       0.5,
            "siglip_ai_max":        0.5,
            "siglip_high_prob_frac": 0.5,
            "siglip_enabled":       False,
        }

    # GPU path — activated when SIGLIP_ENABLED = True
    try:
        import torch
        from transformers import AutoProcessor, AutoModel
        from PIL import Image

        device = (
            "cuda"  if torch.cuda.is_available()  else
            "mps"   if torch.backends.mps.is_available() else
            "cpu"
        )

        processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
        model     = AutoModel.from_pretrained(
            SIGLIP_MODEL
        ).to(device).eval()

        all_prompts = SIGLIP_AI_PROMPTS + SIGLIP_REAL_PROMPTS
        ai_probs    = []

        with torch.no_grad():
            for fpath in all_frames[::4]:
                img    = Image.open(fpath).convert("RGB")
                inputs = processor(
                    text=all_prompts,
                    images=[img] * len(all_prompts),
                    return_tensors="pt",
                    padding="max_length"
                ).to(device)

                outputs = model(**inputs)
                logits  = outputs.logits_per_image[0]
                ai_l    = logits[:len(SIGLIP_AI_PROMPTS)].mean()
                re_l    = logits[len(SIGLIP_AI_PROMPTS):].mean()
                prob    = torch.softmax(
                    torch.stack([ai_l, re_l]), dim=0
                )[0].item()
                ai_probs.append(prob)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "siglip_ai_mean":        float(np.mean(ai_probs)),
            "siglip_ai_max":         float(np.max(ai_probs)),
            "siglip_high_prob_frac": float(
                np.mean(np.array(ai_probs) > 0.5)
            ),
            "siglip_enabled":        True,
        }

    except Exception as e:
        return {
            "siglip_ai_mean":        0.5,
            "siglip_ai_max":         0.5,
            "siglip_high_prob_frac": 0.5,
            "siglip_enabled":        False,
        }