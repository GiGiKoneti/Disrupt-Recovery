"""
Input Validation Helpers

Validates text input before entering the detection pipeline.

Author: SynthDetect Team
"""

from typing import Optional, Tuple


def validate_text_input(
    text: str,
    min_words: int = 50,
    max_words: int = 5000,
) -> Tuple[bool, Optional[str]]:
    """
    Validate text input for detection.

    Args:
        text: Input text to validate.
        min_words: Minimum word count.
        max_words: Maximum word count.

    Returns:
        Tuple of (is_valid, error_message).
        error_message is None if valid.
    """
    if not text:
        return False, "Text input is empty."

    if not isinstance(text, str):
        return False, f"Expected string, got {type(text).__name__}."

    # Check printable content
    printable_chars = sum(c.isprintable() for c in text)
    if printable_chars < 10:
        return False, "Text contains insufficient printable characters."

    # Word count check
    word_count = len(text.split())

    if word_count < min_words:
        return False, f"Text too short: {word_count} words (minimum: {min_words})."

    if word_count > max_words:
        return False, f"Text too long: {word_count} words (maximum: {max_words})."

    return True, None


def validate_api_key(key: str, provider: str = "google") -> Tuple[bool, Optional[str]]:
    """
    Basic validation of API key format.

    Args:
        key: API key string.
        provider: Provider name for error messages.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not key or not key.strip():
        return False, f"{provider} API key is not set."

    if key.startswith("your-") or key == "sk-your-key-here":
        return False, f"{provider} API key appears to be a placeholder."

    return True, None
