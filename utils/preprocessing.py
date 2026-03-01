"""
Text preprocessing utilities for the EVG Course Recommender.
Consolidates cleaning logic from the original SBERT and TF-IDF recommenders.
"""

import re
import nltk

try:
    from nltk.corpus import stopwords as _sw
    _sw.words("portuguese")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

# ── Accent translation table (reused from original TF-IDF recommender) ──────
_ACCENT_MAP = {
    "ã": "a", "Ã": "a", "â": "a", "Â": "a", "á": "a", "Á": "a", "à": "a", "À": "a",
    "ê": "e", "Ê": "e", "é": "e", "É": "e", "è": "e", "È": "e",
    "î": "i", "Î": "i", "í": "i", "Í": "i", "ì": "i", "Ì": "i", "ï": "i", "Ï": "i",
    "õ": "o", "Õ": "o", "ô": "o", "Ô": "o", "ó": "o", "Ó": "o", "ò": "o", "Ò": "o",
    "û": "u", "Û": "u", "ú": "u", "Ú": "u", "ù": "u", "Ù": "u", "ü": "u", "Ü": "u",
    "ç": "c", "Ç": "c", "ñ": "n", "Ñ": "n",
    "/": " ", "\\": " ", "'": " ", "-": " ", '"': " ",
    "<": " ", ">": " ", ",": " ", ".": " ", "?": " ", "!": " ",
}
_TRANS_TABLE = str.maketrans(_ACCENT_MAP)

STOP_WORDS = set(stopwords.words("portuguese"))


def clean_text(text: str, stop_words: set | None = None) -> str:
    """Lowercase, remove accents, strip stopwords and extra punctuation."""
    if not isinstance(text, str) or not text.strip():
        return ""
    sw = stop_words or STOP_WORDS
    text = text.translate(_TRANS_TABLE).lower()
    text = " ".join(w for w in text.split() if w not in sw)
    text = re.sub(r"[;,.]", "", text)
    return text.strip()


def compile_text(row, text_cols: list[str]) -> str:
    """Join several columns into a single text blob for embedding."""
    parts = [str(row.get(c, "")) for c in text_cols]
    return " ".join(p for p in parts if p)
