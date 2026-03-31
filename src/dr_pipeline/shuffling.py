"""
Within-Chunk Shuffling Module for D&R Pipeline

Applies controlled permutation of sentences within a text chunk.
The key insight: AI-generated text recovers better from shuffling
due to posterior concentration.

Shuffle Levels:
- Sentence-level: reorder sentences within chunk
- Clause-level: reorder clauses (future work)

Constraints:
- Preserve first/last sentences (context anchors)
- Controlled randomness via preserve_ratio

Author: SynthDetect Team
"""

import random
from typing import List
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("dr_pipeline.shuffling")


class ShuffleLevel(str, Enum):
    SENTENCE = "sentence"
    CLAUSE = "clause"


class ShuffleEngine:
    """
    Applies within-chunk shuffling to disrupt text structure.

    This is the "Disrupt" step in D&R. The degree of disruption
    is controlled by preserve_ratio and preserve_boundaries.
    """

    def __init__(
        self,
        shuffle_level: ShuffleLevel = ShuffleLevel.SENTENCE,
        preserve_ratio: float = 0.2,
        preserve_boundaries: bool = True,
    ):
        """
        Args:
            shuffle_level: Granularity of shuffling.
            preserve_ratio: Fraction of sentences to keep in original position (0.0-1.0).
            preserve_boundaries: If True, keep first and last sentences in place.
        """
        self.shuffle_level = shuffle_level
        self.preserve_ratio = preserve_ratio
        self.preserve_boundaries = preserve_boundaries

        # Lazy-load spaCy
        self._nlp = None

        logger.info(
            f"ShuffleEngine initialized: level={shuffle_level.value}, "
            f"preserve_ratio={preserve_ratio}, preserve_boundaries={preserve_boundaries}"
        )

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        return self._nlp

    def disrupt(self, chunk: str, seed: int = None) -> str:
        """
        Main shuffling method.

        Args:
            chunk: Text chunk to shuffle.
            seed: Random seed for reproducibility.

        Returns:
            Shuffled version of the chunk.
        """
        if not chunk or not chunk.strip():
            return chunk

        if seed is not None:
            random.seed(seed)

        if self.shuffle_level == ShuffleLevel.SENTENCE:
            return self._shuffle_sentences(chunk)
        elif self.shuffle_level == ShuffleLevel.CLAUSE:
            # Fallback to sentence-level for MVP
            logger.debug("Clause-level shuffling not implemented, using sentence-level")
            return self._shuffle_sentences(chunk)
        else:
            raise ValueError(f"Unknown shuffle level: {self.shuffle_level}")

    def _shuffle_sentences(self, chunk: str) -> str:
        """
        Shuffle sentences within chunk with controlled preservation.

        Algorithm:
        1. Parse into sentences via spaCy
        2. Separate boundary sentences (first/last) if preserve_boundaries
        3. Determine which middle sentences to preserve in place
        4. Shuffle remaining middle sentences
        5. Reconstruct
        """
        doc = self.nlp(chunk)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if len(sentences) <= 2:
            # Too few sentences to shuffle meaningfully
            logger.debug(f"Only {len(sentences)} sentences — skipping shuffle")
            return chunk

        # Separate boundary and middle sentences
        if self.preserve_boundaries:
            first_sent = sentences[0]
            last_sent = sentences[-1]
            middle_sents = sentences[1:-1]
        else:
            first_sent = None
            last_sent = None
            middle_sents = list(sentences)

        if not middle_sents:
            return chunk

        # Determine how many middle sentences to preserve in place
        num_to_preserve = max(0, int(len(middle_sents) * self.preserve_ratio))
        num_to_preserve = min(num_to_preserve, len(middle_sents) - 1)  # Must shuffle at least 1

        if num_to_preserve > 0:
            preserve_indices = set(
                random.sample(range(len(middle_sents)), num_to_preserve)
            )
        else:
            preserve_indices = set()

        # Separate shuffleable and preserved
        shuffleable = [
            s for i, s in enumerate(middle_sents) if i not in preserve_indices
        ]
        random.shuffle(shuffleable)

        # Reconstruct middle with preserved sentences in place
        shuffled_middle = []
        shuffleable_iter = iter(shuffleable)

        for i in range(len(middle_sents)):
            if i in preserve_indices:
                shuffled_middle.append(middle_sents[i])
            else:
                shuffled_middle.append(next(shuffleable_iter))

        # Reconstruct final text
        result_parts = []
        if first_sent:
            result_parts.append(first_sent)
        result_parts.extend(shuffled_middle)
        if last_sent:
            result_parts.append(last_sent)

        result = " ".join(result_parts)

        logger.debug(
            f"Shuffled {len(sentences)} sentences: "
            f"{len(preserve_indices)} preserved, "
            f"{len(shuffleable)} shuffled"
        )

        return result

    def compute_disruption_score(self, original: str, shuffled: str) -> float:
        """
        Measure how much the text was disrupted (0.0 = identical, 1.0 = fully shuffled).

        Uses normalized Levenshtein-like comparison at sentence level.

        Args:
            original: Original text.
            shuffled: Shuffled text.

        Returns:
            Disruption score in [0.0, 1.0].
        """
        orig_sents = [s.text.strip() for s in self.nlp(original).sents]
        shuf_sents = [s.text.strip() for s in self.nlp(shuffled).sents]

        if not orig_sents:
            return 0.0

        # Count sentences that changed position
        max_len = max(len(orig_sents), len(shuf_sents))
        mismatches = 0

        for i in range(min(len(orig_sents), len(shuf_sents))):
            if orig_sents[i] != shuf_sents[i]:
                mismatches += 1

        # Add any extra sentences as mismatches
        mismatches += abs(len(orig_sents) - len(shuf_sents))

        return mismatches / max_len if max_len > 0 else 0.0
