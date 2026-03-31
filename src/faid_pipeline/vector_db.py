"""
FAISS Vector Database Wrapper

Manages FAISS indices for FAID's k-NN attribution.
Supports per-model-family indices for fine-grained attribution.

Author: SynthDetect Team
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("faid_pipeline.vector_db")


class VectorDB:
    """
    FAISS-based vector database for embedding similarity search.

    Stores embeddings with metadata (label, model_family, text_id)
    for k-NN attribution in the FAID pipeline.
    """

    def __init__(
        self,
        dim: int = 256,
        index_type: str = "flat",
        db_dir: str = "data/vector_db",
    ):
        """
        Args:
            dim: Embedding dimension.
            index_type: FAISS index type ("flat" for exact search).
            db_dir: Directory to store/load index files.
        """
        self.dim = dim
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Create FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine on L2-normed)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Metadata storage
        self.metadata: List[Dict] = []

        logger.info(f"VectorDB initialized: dim={dim}, index_type={index_type}")

    def add(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict],
    ) -> None:
        """
        Add embeddings with metadata to the index.

        Args:
            embeddings: Normalised embeddings [n, dim].
            metadata_list: List of metadata dicts for each embedding.
        """
        assert len(embeddings) == len(metadata_list), "Embeddings and metadata must match"
        assert embeddings.shape[1] == self.dim, f"Expected dim={self.dim}, got {embeddings.shape[1]}"

        # Ensure float32 and L2-normalised
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

        logger.info(f"Added {len(embeddings)} vectors (total: {self.index.ntotal})")

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search for k nearest neighbours.

        Args:
            query: Query embedding [1, dim] or [dim].
            k: Number of neighbours.

        Returns:
            Tuple of (distances, indices, metadata_list).
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = query.astype(np.float32)
        faiss.normalize_L2(query)

        k = min(k, self.index.ntotal)  # Can't find more than we have
        distances, indices = self.index.search(query, k)

        # Gather metadata
        result_metadata = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                result_metadata.append(self.metadata[idx])
            else:
                result_metadata.append({"error": "index out of range"})

        return distances[0], indices[0], result_metadata

    def save(self, name: str = "main") -> None:
        """Save index and metadata to disk."""
        index_path = self.db_dir / f"{name}_index.faiss"
        meta_path = self.db_dir / f"{name}_metadata.json"

        faiss.write_index(self.index, str(index_path))

        with open(meta_path, "w") as f:
            json.dump(self.metadata, f)

        logger.info(f"VectorDB saved: {index_path} ({self.index.ntotal} vectors)")

    def load(self, name: str = "main") -> bool:
        """Load index and metadata from disk."""
        index_path = self.db_dir / f"{name}_index.faiss"
        meta_path = self.db_dir / f"{name}_metadata.json"

        if not index_path.exists() or not meta_path.exists():
            logger.warning(f"VectorDB files not found for '{name}'")
            return False

        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        logger.info(f"VectorDB loaded: {index_path} ({self.index.ntotal} vectors)")
        return True

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal
