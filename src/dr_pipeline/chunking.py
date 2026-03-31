"""
Text Chunking Module for D&R Pipeline

Splits text into processable chunks using three strategies:
- Fixed-length: every N words (simplest)
- Semantic: respects sentence boundaries via spaCy
- Sliding window: overlapping chunks for long texts

Author: SynthDetect Team
"""

from typing import List
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("dr_pipeline.chunking")


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    SLIDING = "sliding"


class ChunkingEngine:
    """
    Splits text into processable chunks for the D&R pipeline.

    Default: semantic chunking (~200 words per chunk, sentence-boundary-aware).
    """

    def __init__(
        self,
        chunk_size: int = 200,
        overlap: int = 50,
        strategy: ChunkStrategy = ChunkStrategy.SEMANTIC,
    ):
        """
        Args:
            chunk_size: Target words per chunk.
            overlap: Word overlap for sliding window strategy.
            strategy: Chunking strategy to use.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

        # Lazy-load spaCy only when needed
        self._nlp = None

        logger.info(
            f"ChunkingEngine initialized: strategy={strategy.value}, "
            f"chunk_size={chunk_size}, overlap={overlap}"
        )

    @property
    def nlp(self):
        """Lazy-load spaCy model to save memory."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            logger.info("spaCy model loaded for chunking")
        return self._nlp

    def chunk_text(self, text: str) -> List[str]:
        """
        Main chunking method.

        Args:
            text: Input text to chunk.

        Returns:
            List of text chunks. Empty list for empty input.
        """
        if not text or not text.strip():
            return []

        if self.strategy == ChunkStrategy.FIXED:
            chunks = self._fixed_chunking(text)
        elif self.strategy == ChunkStrategy.SEMANTIC:
            chunks = self._semantic_chunking(text)
        elif self.strategy == ChunkStrategy.SLIDING:
            chunks = self._sliding_window_chunking(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

        logger.debug(f"Chunked text into {len(chunks)} chunks")
        return chunks

    def _fixed_chunking(self, text: str) -> List[str]:
        """
        Simple fixed-length chunking.
        Splits every N words regardless of sentence boundaries.
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk_words))

        return chunks

    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Semantic chunking that respects sentence boundaries.
        Aims for chunks ~chunk_size words without splitting mid-sentence.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        current_chunk: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            # If adding this sentence exceeds chunk_size, finalize current chunk
            if (
                current_word_count + sentence_word_count > self.chunk_size
                and current_chunk
            ):
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        # Add remaining sentences as final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _sliding_window_chunking(self, text: str) -> List[str]:
        """
        Sliding window for long texts.
        Creates overlapping chunks to avoid boundary effects.
        """
        words = text.split()
        chunks = []

        step_size = max(1, self.chunk_size - self.overlap)

        for i in range(0, len(words), step_size):
            chunk_words = words[i : i + self.chunk_size]

            # Skip very small final chunks
            if len(chunk_words) < self.chunk_size // 2 and chunks:
                break

            chunks.append(" ".join(chunk_words))

        return chunks
