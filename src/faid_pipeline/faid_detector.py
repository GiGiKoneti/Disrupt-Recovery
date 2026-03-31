"""
FAID Detector — Main Orchestrator

Chains the Fine-Grained AI Detection pipeline:
  Text → Feature Extraction → Contrastive Encoding → FAISS Search → Attribution

Produces:
- Binary detection (AI vs human)
- Model family attribution (GPT, Claude, Gemini, etc.)
- Confidence score

Author: SynthDetect Team
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.faid_pipeline.contrastive_encoder import FAIDEncoder
from src.faid_pipeline.attribution import AttributionEngine, AttributionResult
from src.faid_pipeline.vector_db import VectorDB
from src.utils.logger import get_logger

logger = get_logger("faid_pipeline.detector")


@dataclass
class FAIDResult:
    """Result from FAID detection pipeline."""

    is_ai: bool                    # Binary: is this AI-generated?
    confidence: float              # Confidence in the classification [0, 1]
    predicted_source: str          # Model family or "human"
    attribution: AttributionResult = None
    metadata: Dict = field(default_factory=dict)


class FAIDDetector:
    """
    Main FAID pipeline orchestrator.

    Pipeline:
    1. Extract linguistic features (lexical, syntactic, semantic, stylometric)
    2. Encode via contrastive encoder → 256-dim embedding
    3. Search FAISS index for k nearest neighbours
    4. Attribute via weighted k-NN voting
    """

    def __init__(
        self,
        encoder: FAIDEncoder = None,
        vector_db: VectorDB = None,
        attribution_engine: AttributionEngine = None,
        confidence_threshold: float = 0.65,
        checkpoint_path: str = "data/models/contrastive_encoder_v1.pth",
        vector_db_name: str = "main",
    ):
        """
        Args:
            encoder: Pre-configured FAIDEncoder.
            vector_db: Pre-configured VectorDB.
            attribution_engine: Pre-configured AttributionEngine.
            confidence_threshold: Min confidence for AI classification.
            checkpoint_path: Path to encoder checkpoint.
            vector_db_name: Name of FAISS index to load.
        """
        self.confidence_threshold = confidence_threshold

        # Initialize components
        self.encoder = encoder or FAIDEncoder(checkpoint_path=checkpoint_path)

        self.vector_db = vector_db or VectorDB()
        if self.vector_db.size == 0:
            self.vector_db.load(vector_db_name)

        self.attribution_engine = attribution_engine or AttributionEngine(
            vector_db=self.vector_db,
        )

        logger.info(
            f"FAIDDetector initialized: "
            f"encoder_trained={self.encoder.is_trained}, "
            f"vector_db_size={self.vector_db.size}, "
            f"threshold={confidence_threshold}"
        )

    def detect(self, text: str) -> FAIDResult:
        """
        Run full FAID detection on input text.

        Args:
            text: Input text to analyse.

        Returns:
            FAIDResult with classification and attribution.
        """
        logger.info(f"Starting FAID detection on {len(text.split())} words")

        # Step 1-2: Feature extraction + encoding
        embedding = self.encoder.encode_text(text)

        # Step 3-4: FAISS search + attribution
        attribution = self.attribution_engine.attribute(embedding)

        # Classification
        is_ai = (
            attribution.predicted_source != "human"
            and attribution.confidence >= self.confidence_threshold
        )

        result = FAIDResult(
            is_ai=is_ai,
            confidence=attribution.confidence,
            predicted_source=attribution.predicted_source,
            attribution=attribution,
            metadata={
                "encoder_trained": self.encoder.is_trained,
                "vector_db_size": self.vector_db.size,
            },
        )

        logger.info(
            f"FAID result: is_ai={result.is_ai}, "
            f"source={result.predicted_source}, "
            f"confidence={result.confidence}"
        )

        return result
