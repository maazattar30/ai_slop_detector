"""
text_features.py — Production version
Title and description text signals
"""

from config import AI_KEYWORDS, SLOP_KEYWORDS


def extract_text_features(
    title:       str = "",
    description: str = "",
) -> dict:
    """
    Text-based signals from title and description.

    ai_keyword_count     — explicit AI tool mentions
    slop_keyword_count   — clickbait/content farm language
    title_exclamation    — excessive punctuation
    title_word_count     — very short titles = lazy content
    lexical_diversity    — low diversity = templated description
    description_length   — empty = no human effort
    """
    combined = (title + " " + description).lower()

    ai_kw    = sum(1 for kw in AI_KEYWORDS   if kw in combined)
    slop_kw  = sum(1 for kw in SLOP_KEYWORDS if kw in combined)

    words = description.lower().split() if description else []
    lex   = (
        len(set(words)) / len(words)
        if len(words) > 5 else 1.0
    )

    return {
        "ai_keyword_count":    float(ai_kw),
        "ai_keyword_prob":     float(min(ai_kw / 3.0, 1.0)),
        "slop_keyword_count":  float(slop_kw),
        "title_exclamation":   float("!" in title),
        "title_word_count":    float(len(title.split())),
        "lexical_diversity":   lex,
        "description_length":  float(len(description)),
    }