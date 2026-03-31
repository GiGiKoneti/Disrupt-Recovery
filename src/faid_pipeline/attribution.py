"""
k-NN Attribution Module

Given an embedding, finds k-nearest neighbours in the FAISS index
and attributes the text to a model family (or human).

Author: SynthDetect Team
"""

import numpy as np
from collections import Counter
from typing import Dict, Optional
from dataclasses import dataclass

from src.faid_pipeline.vector_db import VectorDB
from src.utils.logger import get_logger

logger = get_logger("faid_pipeline.attribution")


@dataclass
class AttributionResult:
    """Result from k-NN attribution."""

    predicted_source: str           # "human", "gpt4", "claude", "gemini", etc.
    confidence: float               # [0, 1]
    neighbor_labels: list           # Labels of k nearest neighbours
    neighbor_distances: list        # Distances to k nearest neighbours
    label_distribution: Dict        # Vote counts per label


class AttributionEngine:
    """
    Attributes text to a source (human or specific AI model)
    using k-nearest neighbour voting in embedding space.
    """

    def __init__(
        self,
        vector_db: VectorDB = None,
        k: int = 5,
        max_distance: float = 0.8,
    ):
        """
        Args:
            vector_db: Pre-configured VectorDB. Created if None.
            k: Number of neighbours for voting.
            max_distance: Maximum distance for valid attribution.
        """
        self.vector_db = vector_db or VectorDB()
        self.k = k
        self.max_distance = max_distance

        logger.info(f"AttributionEngine initialized: k={k}")

    def attribute(self, embedding: np.ndarray) -> AttributionResult:
        """
        Attribute an embedding to a source via k-NN.

        Args:
            embedding: L2-normalised embedding vector.

        Returns:
            AttributionResult with predicted source and confidence.
        """
        if self.vector_db.size == 0:
            logger.warning("VectorDB is empty — cannot attribute")
            return AttributionResult(
                predicted_source="unknown",
                confidence=0.0,
                neighbor_labels=[],
                neighbor_distances=[],
                label_distribution={},
            )

        # Search k nearest neighbours
        distances, indices, metadata_list = self.vector_db.search(
            embedding, k=self.k
        )

        # Extract labels from metadata
        labels = [m.get("source", "unknown") for m in metadata_list]
        dist_list = distances.tolist()

        # Weighted voting: closer neighbours get higher weight
        # Weight = 1 / (1 + distance) for inner product
        label_weights: Dict[str, float] = {}
        for label, dist in zip(labels, dist_list):
            weight = max(0, dist)  # Inner product similarity (higher = closer)
            label_weights[label] = label_weights.get(label, 0) + weight

        # Majority vote with confidence
        if label_weights:
            predicted = max(label_weights, key=label_weights.get)
            total_weight = sum(label_weights.values())
            confidence = label_weights[predicted] / total_weight if total_weight > 0 else 0
        else:
            predicted = "unknown"
            confidence = 0.0

        # Label distribution (counts)
        label_counts = dict(Counter(labels))

        result = AttributionResult(
            predicted_source=predicted,
            confidence=round(float(confidence), 4),
            neighbor_labels=labels,
            neighbor_distances=dist_list,
            label_distribution=label_counts,
        )

        logger.debug(
            f"Attribution: {predicted} (confidence={result.confidence}, "
            f"distribution={label_counts})"
        )

        return result
