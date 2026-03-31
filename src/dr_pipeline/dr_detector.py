"""
D&R Detector — Main Orchestrator

Chains the full Disrupt-and-Recover pipeline:
  Input Text → Chunking → Shuffling → Recovery (LLM) → Similarity → Score

The final D&R score S_DR(T) indicates likelihood of AI authorship:
  - High S_DR (>0.85): Likely AI-generated (posterior concentration)
  - Low S_DR (<0.65): Likely human-written
  - High variance σ²: Possible collaborative text (mixed authorship)

Author: SynthDetect Team
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.dr_pipeline.chunking import ChunkingEngine, ChunkStrategy
from src.dr_pipeline.shuffling import ShuffleEngine, ShuffleLevel
from src.dr_pipeline.recovery import RecoveryEngine
from src.dr_pipeline.similarity import SimilarityEngine
from src.utils.logger import get_logger

logger = get_logger("dr_pipeline.detector")


@dataclass
class DRResult:
    """Result from D&R detection pipeline."""

    score: float                            # S_DR(T) — mean similarity across chunks
    variance: float                         # σ²_DR — variance (high = collaborative)
    chunk_scores: List[Dict[str, float]]    # Per-chunk similarity details
    num_chunks: int
    label: str                              # "ai_generated" | "human_written" | "uncertain"
    confidence: float                       # Confidence in the label [0, 1]
    metadata: Dict = field(default_factory=dict)


class DRDetector:
    """
    Main D&R pipeline orchestrator.

    Full pipeline:
    1. Chunk text into ~200-word segments
    2. Shuffle sentences within each chunk (controlled disruption)
    3. Ask LLM to recover (reconstruct) each shuffled chunk
    4. Measure similarity between original and recovered chunks
    5. Aggregate scores across chunks

    Decision logic:
    - score > ai_threshold → AI-generated
    - score < human_threshold → Human-written
    - Otherwise → Uncertain (possible collaborative text)
    """

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_strategy: str = "semantic",
        shuffle_preserve_ratio: float = 0.2,
        similarity_alpha: float = 0.6,
        ai_threshold: float = 0.85,
        human_threshold: float = 0.65,
        variance_threshold: float = 0.05,
        gemini_client=None,
        seed: int = 42,
    ):
        """
        Args:
            chunk_size: Target words per chunk.
            chunk_strategy: Chunking strategy (fixed/semantic/sliding).
            shuffle_preserve_ratio: Fraction of sentences to preserve.
            similarity_alpha: Weight for semantic vs structural similarity.
            ai_threshold: Score above this → AI-generated.
            human_threshold: Score below this → human-written.
            variance_threshold: High variance above this → collaborative.
            gemini_client: Pre-configured GeminiClient.
            seed: Random seed for reproducibility.
        """
        self.ai_threshold = ai_threshold
        self.human_threshold = human_threshold
        self.variance_threshold = variance_threshold
        self.seed = seed

        # Initialize pipeline components
        self.chunking_engine = ChunkingEngine(
            chunk_size=chunk_size,
            strategy=ChunkStrategy(chunk_strategy),
        )

        self.shuffle_engine = ShuffleEngine(
            shuffle_level=ShuffleLevel.SENTENCE,
            preserve_ratio=shuffle_preserve_ratio,
            preserve_boundaries=True,
        )

        self.recovery_engine = RecoveryEngine(
            gemini_client=gemini_client,
        )

        self.similarity_engine = SimilarityEngine(
            alpha=similarity_alpha,
        )

        logger.info(
            f"DRDetector initialized: chunk_size={chunk_size}, "
            f"ai_threshold={ai_threshold}, human_threshold={human_threshold}"
        )

    def detect(self, text: str) -> DRResult:
        """
        Run full D&R detection pipeline on input text.

        Args:
            text: Input text to analyse.

        Returns:
            DRResult with score, label, and per-chunk details.
        """
        logger.info(f"Starting D&R detection on {len(text.split())} words")

        # Step 1: Chunk
        chunks = self.chunking_engine.chunk_text(text)
        logger.info(f"Text chunked into {len(chunks)} segments")

        if not chunks:
            return DRResult(
                score=0.5,
                variance=0.0,
                chunk_scores=[],
                num_chunks=0,
                label="uncertain",
                confidence=0.0,
                metadata={"error": "No chunks produced"},
            )

        # Step 2-4: For each chunk — shuffle → recover → similarity
        chunk_scores = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)}")

            # Shuffle (disrupt)
            shuffled = self.shuffle_engine.disrupt(
                chunk, seed=self.seed + i
            )

            # Recover (reconstruct via LLM)
            recovered = self.recovery_engine.recover(shuffled)

            # Measure similarity between ORIGINAL and RECOVERED
            similarity = self.similarity_engine.compute_similarity(
                original=chunk,
                recovered=recovered,
            )

            chunk_scores.append({
                "chunk_index": i,
                "original_words": len(chunk.split()),
                **similarity,
            })

        # Step 5: Aggregate
        combined_scores = [cs["combined"] for cs in chunk_scores]
        mean_score = float(np.mean(combined_scores))
        variance = float(np.var(combined_scores))

        # Classification
        label, confidence = self._classify(mean_score, variance)

        result = DRResult(
            score=round(mean_score, 4),
            variance=round(variance, 6),
            chunk_scores=chunk_scores,
            num_chunks=len(chunks),
            label=label,
            confidence=round(confidence, 4),
            metadata={
                "llm_usage": self.recovery_engine.usage_stats(),
            },
        )

        logger.info(
            f"D&R result: score={result.score}, label={result.label}, "
            f"confidence={result.confidence}, variance={result.variance}"
        )

        return result

    def _classify(self, score: float, variance: float) -> tuple:
        """
        Classify based on D&R score and variance.

        Returns:
            Tuple of (label, confidence).
        """
        # Check for collaborative text (high variance)
        is_collaborative = variance > self.variance_threshold

        if score >= self.ai_threshold:
            label = "ai_generated"
            # Confidence scales with how far above threshold
            confidence = min(1.0, 0.7 + (score - self.ai_threshold) * 3)
            if is_collaborative:
                label = "collaborative_ai_dominant"
                confidence *= 0.85  # Reduce confidence for collaborative

        elif score <= self.human_threshold:
            label = "human_written"
            confidence = min(1.0, 0.7 + (self.human_threshold - score) * 3)
            if is_collaborative:
                label = "collaborative_human_dominant"
                confidence *= 0.85

        else:
            label = "uncertain"
            # Lower confidence in the uncertain zone
            mid = (self.ai_threshold + self.human_threshold) / 2
            distance_from_mid = abs(score - mid)
            range_width = (self.ai_threshold - self.human_threshold) / 2
            confidence = max(0.3, 0.5 - distance_from_mid / range_width * 0.2)

        return label, confidence
