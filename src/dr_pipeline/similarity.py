"""
Semantic-Structural Similarity Module for D&R Pipeline

Computes hybrid similarity between original text and LLM-recovered text:
  sim(C, Ĉ) = α · semantic(C, Ĉ) + (1−α) · structural(C, Ĉ)

Where:
- semantic: cosine similarity of sentence-transformer embeddings
- structural: normalised sentence-level edit distance
- α = 0.6 (tuned on validation data)

High similarity → text is likely AI-generated (posterior concentration)

Author: SynthDetect Team
"""

import numpy as np
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("dr_pipeline.similarity")


class SimilarityEngine:
    """
    Computes hybrid semantic-structural similarity between
    original and recovered text.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Args:
            alpha: Weight for semantic similarity (1−α for structural).
            embedding_model: Sentence-transformer model name.
                             Using MiniLM for M1 memory efficiency.
        """
        self.alpha = alpha
        self.embedding_model_name = embedding_model

        # Lazy-load models
        self._sentence_model = None
        self._nlp = None

        logger.info(
            f"SimilarityEngine initialized: alpha={alpha}, "
            f"embedding_model={embedding_model}"
        )

    @property
    def sentence_model(self):
        """Lazy-load sentence-transformer model."""
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Sentence-transformer loaded: {self.embedding_model_name}")
        return self._sentence_model

    @property
    def nlp(self):
        """Lazy-load spaCy for sentence splitting."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        return self._nlp

    def compute_similarity(
        self,
        original: str,
        recovered: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute hybrid semantic-structural similarity.

        Args:
            original: Original text before shuffling.
            recovered: Recovered text from LLM.
            weights: Custom weights {'semantic': w1, 'structural': w2}.

        Returns:
            Dictionary with:
            - 'semantic': semantic similarity [0, 1]
            - 'structural': structural similarity [0, 1]
            - 'combined': weighted combination [0, 1]

        Raises:
            ValueError: If texts are empty.
        """
        if not original or not recovered:
            raise ValueError("Both original and recovered texts must be non-empty")

        alpha = self.alpha
        if weights:
            alpha = weights.get("semantic", self.alpha)

        # Compute semantic similarity
        semantic_sim = self._compute_semantic_similarity(original, recovered)

        # Compute structural similarity
        structural_sim = self._compute_structural_similarity(original, recovered)

        # Weighted combination
        combined = alpha * semantic_sim + (1 - alpha) * structural_sim

        result = {
            "semantic": round(semantic_sim, 4),
            "structural": round(structural_sim, 4),
            "combined": round(combined, 4),
        }

        logger.debug(
            f"Similarity: semantic={result['semantic']}, "
            f"structural={result['structural']}, "
            f"combined={result['combined']}"
        )

        return result

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity of sentence-transformer embeddings.

        Encodes full texts as single embeddings and computes cosine sim.
        """
        embeddings = self.sentence_model.encode(
            [text1, text2],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Cosine similarity
        cos_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
        )

        return float(np.clip(cos_sim, 0.0, 1.0))

    def _compute_structural_similarity(self, text1: str, text2: str) -> float:
        """
        Compute structural similarity based on sentence order.

        Uses normalised Kendall tau distance on sentence positions.
        Higher similarity means sentences are in more similar order.
        """
        # Split into sentences
        sents1 = [s.text.strip() for s in self.nlp(text1).sents if s.text.strip()]
        sents2 = [s.text.strip() for s in self.nlp(text2).sents if s.text.strip()]

        if not sents1 or not sents2:
            return 0.0

        # Find matching sentences between original and recovered
        # Use fuzzy matching: sentence from sents2 matches sents1[i]
        # if it's the most similar sentence
        matched_positions = []

        for sent2 in sents2:
            best_match_idx = -1
            best_score = 0.0

            for i, sent1 in enumerate(sents1):
                # Simple word overlap score
                words1 = set(sent1.lower().split())
                words2 = set(sent2.lower().split())

                if not words1 or not words2:
                    continue

                overlap = len(words1 & words2) / max(len(words1), len(words2))

                if overlap > best_score:
                    best_score = overlap
                    best_match_idx = i

            if best_match_idx >= 0 and best_score > 0.5:
                matched_positions.append(best_match_idx)

        if len(matched_positions) < 2:
            return 0.5  # Not enough data to judge

        # Compute normalised inversion count (Kendall tau distance)
        n = len(matched_positions)
        inversions = 0
        total_pairs = n * (n - 1) / 2

        for i in range(n):
            for j in range(i + 1, n):
                if matched_positions[i] > matched_positions[j]:
                    inversions += 1

        # Convert to similarity (0 inversions = perfect order = 1.0)
        structural_sim = 1.0 - (inversions / total_pairs) if total_pairs > 0 else 1.0

        return float(np.clip(structural_sim, 0.0, 1.0))

    def compute_batch_similarity(
        self,
        originals: List[str],
        recovereds: List[str],
    ) -> List[Dict[str, float]]:
        """
        Compute similarity for multiple chunk pairs.

        Args:
            originals: List of original text chunks.
            recovereds: List of recovered text chunks.

        Returns:
            List of similarity result dictionaries.
        """
        assert len(originals) == len(recovereds), "Lists must have equal length"

        results = []
        for orig, rec in zip(originals, recovereds):
            results.append(self.compute_similarity(orig, rec))

        return results
