"""
Fusion Layer — Signal Reconciliation

Combines D&R and FAID pipeline outputs into a final classification:
  - fully_ai: High D&R score + FAID confirms AI
  - collaborative: Mixed signals or high D&R variance
  - fully_human: Low D&R score + FAID confirms human

Fusion formula:
  score_final = w_dr * S_DR + w_faid * S_FAID
  (default: w_dr=0.6, w_faid=0.4)

Author: SynthDetect Team
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum

from src.dr_pipeline.dr_detector import DRResult
from src.faid_pipeline.faid_detector import FAIDResult
from src.utils.logger import get_logger

logger = get_logger("core.fusion_layer")


class Classification(str, Enum):
    FULLY_AI = "fully_ai"
    COLLABORATIVE = "collaborative"
    FULLY_HUMAN = "fully_human"
    UNCERTAIN = "uncertain"


@dataclass
class FusionResult:
    """Final detection result from fusion of D&R and FAID."""

    classification: Classification
    confidence: float                    # [0, 1]
    fusion_score: float                  # Combined score [0, 1]
    dr_score: float                      # D&R pipeline score
    faid_confidence: float               # FAID pipeline confidence
    predicted_source: str                # Model attribution (from FAID)
    collaboration_coefficient: float     # 0=pure, 1=highly collaborative
    explanation: str                     # Human-readable explanation
    details: Dict = field(default_factory=dict)


class FusionLayer:
    """
    Reconciles signals from D&R and FAID pipelines.

    Handles cases where pipelines agree, disagree, or
    when only one pipeline produces confident results.
    """

    def __init__(
        self,
        weight_dr: float = 0.6,
        weight_faid: float = 0.4,
        ai_threshold: float = 0.80,
        human_threshold: float = 0.40,
        min_confidence: float = 0.50,
        variance_threshold: float = 0.05,
    ):
        self.weight_dr = weight_dr
        self.weight_faid = weight_faid
        self.ai_threshold = ai_threshold
        self.human_threshold = human_threshold
        self.min_confidence = min_confidence
        self.variance_threshold = variance_threshold

        logger.info(
            f"FusionLayer initialized: w_dr={weight_dr}, w_faid={weight_faid}, "
            f"ai_threshold={ai_threshold}, human_threshold={human_threshold}"
        )

    def fuse(
        self,
        dr_result: DRResult,
        faid_result: FAIDResult,
    ) -> FusionResult:
        """
        Fuse D&R and FAID results into final classification.

        Args:
            dr_result: Result from D&R pipeline.
            faid_result: Result from FAID pipeline.

        Returns:
            FusionResult with unified classification and explanation.
        """
        # Normalise FAID confidence to [0, 1] score
        # If FAID says AI, use confidence directly; if human, invert
        faid_score = faid_result.confidence if faid_result.is_ai else (1 - faid_result.confidence)

        # Weighted fusion
        fusion_score = (
            self.weight_dr * dr_result.score
            + self.weight_faid * faid_score
        )

        # Collaboration coefficient (from D&R variance)
        collab_coeff = min(1.0, dr_result.variance / self.variance_threshold)

        # Classification
        classification, confidence = self._classify(
            fusion_score, dr_result, faid_result, collab_coeff
        )

        # Generate explanation
        explanation = self._generate_explanation(
            classification, fusion_score, dr_result, faid_result, collab_coeff
        )

        result = FusionResult(
            classification=classification,
            confidence=round(confidence, 4),
            fusion_score=round(fusion_score, 4),
            dr_score=dr_result.score,
            faid_confidence=faid_result.confidence,
            predicted_source=faid_result.predicted_source,
            collaboration_coefficient=round(collab_coeff, 4),
            explanation=explanation,
            details={
                "dr_label": dr_result.label,
                "dr_variance": dr_result.variance,
                "faid_is_ai": faid_result.is_ai,
                "faid_source": faid_result.predicted_source,
                "weight_dr": self.weight_dr,
                "weight_faid": self.weight_faid,
            },
        )

        logger.info(
            f"Fusion result: {classification.value} "
            f"(confidence={confidence:.3f}, score={fusion_score:.3f})"
        )

        return result

    def _classify(
        self,
        fusion_score: float,
        dr_result: DRResult,
        faid_result: FAIDResult,
        collab_coeff: float,
    ) -> tuple:
        """Determine classification and confidence."""

        # High collaboration → collaborative
        if collab_coeff > 0.7:
            return Classification.COLLABORATIVE, 0.5 + collab_coeff * 0.3

        # Strong AI signal
        if fusion_score >= self.ai_threshold:
            confidence = min(1.0, 0.7 + (fusion_score - self.ai_threshold) * 2)

            # Check for agreement between pipelines
            if faid_result.is_ai:
                confidence = min(1.0, confidence + 0.1)  # Agreement bonus
            else:
                confidence *= 0.85  # Disagreement penalty

            # Moderate collaboration → collaborative AI-dominant
            if collab_coeff > 0.3:
                return Classification.COLLABORATIVE, confidence * 0.9

            return Classification.FULLY_AI, confidence

        # Strong human signal
        elif fusion_score <= self.human_threshold:
            confidence = min(1.0, 0.7 + (self.human_threshold - fusion_score) * 2)

            if not faid_result.is_ai:
                confidence = min(1.0, confidence + 0.1)
            else:
                confidence *= 0.85

            if collab_coeff > 0.3:
                return Classification.COLLABORATIVE, confidence * 0.9

            return Classification.FULLY_HUMAN, confidence

        # Middle zone → uncertain or collaborative
        else:
            if collab_coeff > 0.3:
                return Classification.COLLABORATIVE, 0.5 + collab_coeff * 0.2

            return Classification.UNCERTAIN, 0.4

    def _generate_explanation(
        self,
        classification: Classification,
        fusion_score: float,
        dr_result: DRResult,
        faid_result: FAIDResult,
        collab_coeff: float,
    ) -> str:
        """Generate human-readable explanation of the detection result."""

        parts = []

        # Overall verdict
        verdict_map = {
            Classification.FULLY_AI: "This text appears to be AI-generated.",
            Classification.FULLY_HUMAN: "This text appears to be human-written.",
            Classification.COLLABORATIVE: "This text shows signs of human-AI collaboration.",
            Classification.UNCERTAIN: "The analysis is inconclusive for this text.",
        }
        parts.append(verdict_map[classification])

        # D&R signal
        if dr_result.score >= 0.85:
            parts.append(
                f"D&R analysis shows high recovery fidelity (score: {dr_result.score:.2f}), "
                "indicating the text structure closely matches LLM output patterns."
            )
        elif dr_result.score <= 0.65:
            parts.append(
                f"D&R analysis shows low recovery fidelity (score: {dr_result.score:.2f}), "
                "suggesting idiosyncratic human writing patterns."
            )
        else:
            parts.append(
                f"D&R analysis shows moderate recovery fidelity (score: {dr_result.score:.2f})."
            )

        # FAID attribution
        if faid_result.is_ai and faid_result.predicted_source != "unknown":
            parts.append(
                f"Stylometric analysis attributes this text to {faid_result.predicted_source} "
                f"with {faid_result.confidence:.0%} confidence."
            )
        elif not faid_result.is_ai:
            parts.append(
                "Stylometric analysis is consistent with human authorship."
            )

        # Collaboration indicator
        if collab_coeff > 0.3:
            parts.append(
                f"Variance analysis (coefficient: {collab_coeff:.2f}) suggests "
                "mixed authorship across different sections of the text."
            )

        return " ".join(parts)
