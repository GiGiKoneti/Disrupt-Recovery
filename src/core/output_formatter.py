"""
Output Formatting Module

Structures detection results into clean, serializable output
for research analysis, API responses, and logging.

Author: SynthDetect Team
"""

import json
from dataclasses import asdict
from typing import Dict

from src.core.fusion_layer import FusionResult, Classification
from src.utils.logger import get_logger

logger = get_logger("core.output_formatter")


class OutputFormatter:
    """Formats detection results for various output targets."""

    @staticmethod
    def to_dict(result: FusionResult) -> Dict:
        """Convert FusionResult to a clean dictionary."""
        return {
            "classification": result.classification.value,
            "confidence": result.confidence,
            "fusion_score": result.fusion_score,
            "is_ai_generated": result.classification in (
                Classification.FULLY_AI,
            ),
            "is_collaborative": result.classification == Classification.COLLABORATIVE,
            "scores": {
                "dr_score": result.dr_score,
                "faid_confidence": result.faid_confidence,
                "collaboration_coefficient": result.collaboration_coefficient,
            },
            "attribution": {
                "predicted_source": result.predicted_source,
            },
            "explanation": result.explanation,
        }

    @staticmethod
    def to_json(result: FusionResult, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(OutputFormatter.to_dict(result), indent=indent)

    @staticmethod
    def to_summary(result: FusionResult) -> str:
        """Generate a concise one-line summary."""
        emoji = {
            Classification.FULLY_AI: "🤖",
            Classification.FULLY_HUMAN: "👤",
            Classification.COLLABORATIVE: "🤝",
            Classification.UNCERTAIN: "❓",
        }
        e = emoji.get(result.classification, "❓")
        return (
            f"{e} {result.classification.value.upper()} "
            f"(confidence: {result.confidence:.0%}, "
            f"score: {result.fusion_score:.3f}, "
            f"source: {result.predicted_source})"
        )
