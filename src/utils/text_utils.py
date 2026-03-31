"""
Text Utility Functions

Common text operations used across the project:
- Cleaning, normalisation, tokenisation
- Zero-width character removal (adversarial defence)
- Word/sentence counting

Author: SynthDetect Team
"""

import re
import unicodedata
from typing import List


# Zero-width characters commonly used in adversarial evasion
ZERO_WIDTH_CHARS = [
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\ufeff",  # Zero-width no-break space (BOM)
    "\u00ad",  # Soft hyphen
    "\u200e",  # Left-to-right mark
    "\u200f",  # Right-to-left mark
]


def clean_text(text: str) -> str:
    """
    Normalize and clean text for processing.

    Operations:
    - UTF-8 normalisation (NFC)
    - Remove zero-width characters
    - Normalise whitespace
    - Strip leading/trailing whitespace

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)

    # Remove zero-width characters
    for char in ZERO_WIDTH_CHARS:
        text = text.replace(char, "")

    # Normalise whitespace (collapse multiple spaces/newlines)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def count_words(text: str) -> int:
    """Count words via whitespace split."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def count_sentences(text: str) -> int:
    """
    Estimate sentence count using punctuation heuristic.
    For precise counts, use spaCy via the chunking module.
    """
    if not text:
        return 0
    # Match sentence-ending punctuation followed by space or end
    sentences = re.split(r"[.!?]+\s+", text.strip())
    return len([s for s in sentences if s.strip()])


def split_into_sentences(text: str) -> List[str]:
    """
    Basic sentence splitting using regex.
    For production, prefer spaCy's sentence segmenter.

    Args:
        text: Input text.

    Returns:
        List of sentence strings.
    """
    if not text:
        return []

    # Split on sentence-ending punctuation followed by whitespace
    # Handles Mr., Dr., etc. imperfectly — use spaCy for accuracy
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def truncate_text(text: str, max_words: int = 5000) -> str:
    """Truncate text to max_words, preserving sentence boundaries where possible."""
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])
    # Try to end at a sentence boundary
    last_period = truncated.rfind(".")
    if last_period > len(truncated) * 0.8:  # Only if we don't lose too much
        truncated = truncated[: last_period + 1]

    return truncated


def has_enough_content(text: str, min_words: int = 50) -> bool:
    """Check if text has minimum content for analysis."""
    return count_words(text) >= min_words
