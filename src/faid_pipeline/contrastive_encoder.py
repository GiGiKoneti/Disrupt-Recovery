"""
FAID Contrastive Encoder (Inference Wrapper)

Wraps the trained PyTorch encoder for inference:
- Loads trained checkpoint
- Extracts features → produces embeddings
- Used by the FAID detector for attribution

Author: SynthDetect Team
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional

from src.models.encoder_network import ContrastiveEncoder
from src.models.model_utils import load_checkpoint
from src.faid_pipeline.feature_extraction import FeatureExtractor
from src.utils.logger import get_logger

logger = get_logger("faid_pipeline.contrastive_encoder")


class FAIDEncoder:
    """
    Inference wrapper for the contrastive encoder.

    Combines feature extraction + model inference into
    a single call that produces embeddings from raw text.
    """

    def __init__(
        self,
        checkpoint_path: str = "data/models/contrastive_encoder_v1.pth",
        input_dim: int = 195,
        embedding_dim: int = 256,
        device: str = "cpu",
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint.
            input_dim: Feature vector dimension.
            embedding_dim: Output embedding dimension.
            device: PyTorch device (cpu for M1 local).
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Model
        self.model = ContrastiveEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
        )
        self.model.to(device)
        self.model.eval()

        # Load weights if checkpoint exists
        self._loaded = False
        if self.checkpoint_path.exists():
            try:
                load_checkpoint(self.model, str(self.checkpoint_path), device=device)
                self._loaded = True
                logger.info(f"Encoder loaded from {self.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        else:
            logger.warning(
                f"No checkpoint found at {self.checkpoint_path}. "
                "Using random weights (train model first)."
            )

    @property
    def is_trained(self) -> bool:
        """Whether a trained checkpoint was loaded."""
        return self._loaded

    def encode_text(self, text: str) -> np.ndarray:
        """
        Extract features and produce embedding for a single text.

        Args:
            text: Input text.

        Returns:
            L2-normalised embedding vector [embedding_dim].
        """
        # Extract features
        feature_vector = self.feature_extractor.extract_flat_vector(text)

        # Convert to tensor
        x = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            embedding = self.model(x)

        return embedding.cpu().numpy().flatten()

    def encode_batch(self, texts: list) -> np.ndarray:
        """
        Encode a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            Embeddings array [n_texts, embedding_dim].
        """
        # Extract features for all texts
        feature_vectors = []
        for text in texts:
            fv = self.feature_extractor.extract_flat_vector(text)
            feature_vectors.append(fv)

        # Stack and convert
        x = torch.FloatTensor(np.array(feature_vectors)).to(self.device)

        # Inference
        with torch.no_grad():
            embeddings = self.model(x)

        return embeddings.cpu().numpy()
