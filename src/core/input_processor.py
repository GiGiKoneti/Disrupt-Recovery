"""
Input Processing Module

Responsibilities:
- Text normalization (encoding, whitespace, special characters)
- Length-based routing (short, standard, long text strategies)
- Metadata extraction (language detection, domain classification)
- Input validation

Author: SynthDetect Team
"""

from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass

from src.utils.text_utils import clean_text, count_words
from src.utils.validators import validate_text_input
from src.utils.logger import get_logger

logger = get_logger("core.input_processor")


class TextRouting(str, Enum):
    """Pipeline routing strategy based on text length."""
    SHORT = "short"        # <100 words → FAID only
    STANDARD = "standard"  # 100-1000 words → Full dual pipeline
    LONG = "long"          # >1000 words → Sliding window approach


@dataclass
class ProcessedInput:
    """Result of input preprocessing."""
    text: str
    word_count: int
    routing: TextRouting
    metadata: Dict
    is_valid: bool
    error: Optional[str] = None


class InputProcessor:
    """
    Preprocesses and validates input text for detection pipelines.

    Handles text cleaning, length validation, routing decisions,
    and metadata extraction before text enters D&R or FAID pipelines.
    """

    MIN_LENGTH_WORDS = 50
    MAX_LENGTH_WORDS = 5000
    SHORT_TEXT_THRESHOLD = 100
    LONG_TEXT_THRESHOLD = 1000

    def __init__(
        self,
        min_words: int = None,
        max_words: int = None,
        short_threshold: int = None,
        long_threshold: int = None,
    ):
        self.min_words = min_words or self.MIN_LENGTH_WORDS
        self.max_words = max_words or self.MAX_LENGTH_WORDS
        self.short_threshold = short_threshold or self.SHORT_TEXT_THRESHOLD
        self.long_threshold = long_threshold or self.LONG_TEXT_THRESHOLD

        logger.info(
            f"InputProcessor initialized: min={self.min_words}, max={self.max_words}, "
            f"short_threshold={self.short_threshold}, long_threshold={self.long_threshold}"
        )

    def preprocess(self, text: str, metadata: Optional[Dict] = None) -> ProcessedInput:
        """
        Main preprocessing pipeline.

        Args:
            text: Raw input text.
            metadata: Optional metadata (author_id, domain, etc.).

        Returns:
            ProcessedInput with cleaned text, routing, and metadata.
        """
        # Validate input
        is_valid, error = validate_text_input(text, self.min_words, self.max_words)

        if not is_valid:
            logger.warning(f"Input validation failed: {error}")
            return ProcessedInput(
                text="",
                word_count=0,
                routing=TextRouting.STANDARD,
                metadata={},
                is_valid=False,
                error=error,
            )

        # Clean text
        cleaned_text = clean_text(text)
        word_count = count_words(cleaned_text)

        # Re-validate after cleaning (cleaning may reduce word count)
        if word_count < self.min_words:
            error = f"Text too short after cleaning: {word_count} words (min: {self.min_words})"
            logger.warning(error)
            return ProcessedInput(
                text=cleaned_text,
                word_count=word_count,
                routing=TextRouting.STANDARD,
                metadata={},
                is_valid=False,
                error=error,
            )

        # Determine routing
        routing = self._determine_routing(word_count)

        # Extract metadata
        extracted_metadata = self._extract_metadata(cleaned_text)
        if metadata:
            extracted_metadata.update(metadata)

        logger.info(
            f"Input processed: {word_count} words, routing={routing.value}, "
            f"formality={extracted_metadata.get('formality', 'unknown')}"
        )

        return ProcessedInput(
            text=cleaned_text,
            word_count=word_count,
            routing=routing,
            metadata=extracted_metadata,
            is_valid=True,
        )

    def _determine_routing(self, word_count: int) -> TextRouting:
        """Determine pipeline routing based on text length."""
        if word_count < self.short_threshold:
            return TextRouting.SHORT
        elif word_count <= self.long_threshold:
            return TextRouting.STANDARD
        else:
            return TextRouting.LONG

    def _extract_metadata(self, text: str) -> Dict:
        """
        Extract basic metadata from text.

        Extractions:
        - Language (default English for now)
        - Formality heuristic (contraction presence)
        - Estimated reading level
        """
        metadata = {}

        # Language detection placeholder
        metadata["language"] = "en"

        # Formality heuristic
        contractions = ["don't", "can't", "won't", "i'm", "you're", "they're", "isn't", "wasn't"]
        text_lower = text.lower()
        has_contractions = any(c in text_lower for c in contractions)
        metadata["formality"] = "informal" if has_contractions else "formal"

        # Average word length (proxy for complexity)
        words = text.split()
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            metadata["avg_word_length"] = round(avg_word_len, 2)

        return metadata
