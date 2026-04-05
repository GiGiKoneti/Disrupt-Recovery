"""
Explanation Generator
=====================

Produces human-readable explanations for SynthDetect detection results.
Supports multiple output formats: plain text, structured JSON, and
HTML-formatted reports.

The explainability module serves two purposes:
  1. Research: Understanding what signals drive the detection
  2. Production: Providing transparency to end-users

Author: SynthDetect Team
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExplanationDetail(Enum):
    """Level of detail in the explanation."""
    BRIEF = "brief"       # One-line verdict
    STANDARD = "standard" # Verdict + key evidence
    DETAILED = "detailed" # Full pipeline breakdown


@dataclass
class Evidence:
    """A single piece of evidence supporting the detection."""
    signal: str          # e.g., "D&R Structural Similarity"
    value: float         # e.g., 0.9334
    interpretation: str  # e.g., "High recovery fidelity suggests AI origin"
    weight: float = 1.0  # Contribution to final score


@dataclass
class Explanation:
    """Full explanation of a detection result."""
    verdict: str                      # "AI-Generated", "Human-Written", etc.
    confidence: float                 # 0.0 - 1.0
    summary: str                      # Human-readable summary
    evidence: list[Evidence] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "evidence": [
                {
                    "signal": e.signal,
                    "value": round(e.value, 4),
                    "interpretation": e.interpretation,
                    "weight": e.weight,
                }
                for e in self.evidence
            ],
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        lines = []
        lines.append(f"═══ SynthDetect Analysis ═══")
        lines.append(f"Verdict     : {self.verdict}")
        lines.append(f"Confidence  : {self.confidence:.1%}")
        lines.append(f"Summary     : {self.summary}")
        if self.evidence:
            lines.append(f"\nKey Evidence:")
            for e in sorted(self.evidence, key=lambda x: -x.weight):
                lines.append(f"  • {e.signal}: {e.value:.4f}")
                lines.append(f"    → {e.interpretation}")
        if self.recommendations:
            lines.append(f"\nRecommendations:")
            for r in self.recommendations:
                lines.append(f"  • {r}")
        return "\n".join(lines)


class ExplanationGenerator:
    """
    Generates human-readable explanations from detection pipeline results.
    """

    # Thresholds for interpretation
    HIGH_SIM = 0.90
    LOW_SIM = 0.70
    HIGH_VARIANCE = 0.05
    HIGH_CONFIDENCE = 0.80

    def __init__(self, detail: ExplanationDetail = ExplanationDetail.STANDARD):
        self.detail = detail

    def explain_dr_result(self, dr_result: dict) -> list[Evidence]:
        """Generate evidence from D&R pipeline results."""
        evidence = []

        score = dr_result.get("score", 0.5)
        variance = dr_result.get("variance", 0.0)
        semantic = dr_result.get("semantic_similarity", score)
        structural = dr_result.get("structural_similarity", score)

        # Overall recovery score
        if score >= self.HIGH_SIM:
            interp = (
                "The LLM recovered the shuffled text with high fidelity, "
                "suggesting the text follows predictable patterns typical of AI generation."
            )
        elif score <= self.LOW_SIM:
            interp = (
                "The LLM struggled to recover the original text structure, "
                "indicating idiosyncratic human writing patterns."
            )
        else:
            interp = (
                "Moderate recovery fidelity — the text shows some AI-like patterns "
                "but also contains human-like structural variability."
            )

        evidence.append(Evidence(
            signal="D&R Combined Score",
            value=score,
            interpretation=interp,
            weight=0.6,
        ))

        # Structural vs semantic gap
        if self.detail != ExplanationDetail.BRIEF:
            if structural > semantic:
                struct_interp = (
                    "Sentence ordering was recovered very accurately — "
                    "a strong indicator of AI generation, as LLMs produce "
                    "text with highly predictable discourse flow."
                )
            else:
                struct_interp = (
                    "Sentence ordering was harder to recover than meaning, "
                    "suggesting non-standard narrative structure typical of human writing."
                )

            evidence.append(Evidence(
                signal="Structural Similarity",
                value=structural,
                interpretation=struct_interp,
                weight=0.3,
            ))

        # Variance (collaborative detection)
        if self.detail == ExplanationDetail.DETAILED and variance > 0:
            if variance > self.HIGH_VARIANCE:
                var_interp = (
                    "High variance across chunks suggests mixed authorship — "
                    "some sections appear AI-generated while others look human-written."
                )
            else:
                var_interp = (
                    "Low variance indicates consistent authorship throughout the document."
                )

            evidence.append(Evidence(
                signal="Cross-Chunk Variance",
                value=variance,
                interpretation=var_interp,
                weight=0.1,
            ))

        return evidence

    def explain_faid_result(self, faid_result: dict) -> list[Evidence]:
        """Generate evidence from FAID pipeline results."""
        evidence = []

        is_ai = faid_result.get("is_ai", False)
        confidence = faid_result.get("confidence", 0.0)
        source = faid_result.get("attributed_source", "unknown")
        top_sources = faid_result.get("top_sources", [])

        if is_ai:
            if confidence >= self.HIGH_CONFIDENCE:
                interp = (
                    f"Stylometric analysis strongly matches known AI writing patterns. "
                    f"Most likely source: {source} (confidence: {confidence:.1%})."
                )
            else:
                interp = (
                    f"Stylometric features show some AI-like characteristics "
                    f"but attribution confidence is moderate ({confidence:.1%})."
                )
        else:
            interp = (
                "Stylometric features are consistent with human writing patterns — "
                "no strong match to any known AI model family."
            )

        evidence.append(Evidence(
            signal="FAID Attribution",
            value=confidence,
            interpretation=interp,
            weight=0.4,
        ))

        # Model family details
        if self.detail == ExplanationDetail.DETAILED and top_sources:
            for src_info in top_sources[:3]:
                src_name = src_info.get("source", "unknown")
                src_conf = src_info.get("confidence", 0.0)
                evidence.append(Evidence(
                    signal=f"FAID Source Match: {src_name}",
                    value=src_conf,
                    interpretation=f"Similarity to {src_name} writing patterns.",
                    weight=0.05,
                ))

        return evidence

    def generate(
        self,
        fusion_result: dict | None = None,
        dr_result: dict | None = None,
        faid_result: dict | None = None,
    ) -> Explanation:
        """
        Generate a complete explanation from pipeline results.

        Args:
            fusion_result: Output from FusionLayer.fuse()
            dr_result: Output from DRDetector
            faid_result: Output from FAIDDetector
        """
        evidence = []

        # Collect evidence from both pipelines
        if dr_result:
            evidence.extend(self.explain_dr_result(dr_result))

        if faid_result:
            evidence.extend(self.explain_faid_result(faid_result))

        # Determine verdict and confidence
        if fusion_result:
            classification = fusion_result.get("classification", "uncertain")
            confidence = fusion_result.get("confidence", 0.5)
            fusion_score = fusion_result.get("fusion_score", 0.5)
        elif dr_result:
            score = dr_result.get("score", 0.5)
            if score >= 0.85:
                classification = "fully_ai"
            elif score <= 0.40:
                classification = "fully_human"
            else:
                classification = "uncertain"
            confidence = abs(score - 0.5) * 2
            fusion_score = score
        else:
            classification = "uncertain"
            confidence = 0.0
            fusion_score = 0.5

        # Map to human-readable verdict
        verdict_map = {
            "fully_ai": "AI-Generated",
            "fully_human": "Human-Written",
            "collaborative": "Collaborative (Human + AI)",
            "uncertain": "Uncertain — Needs Manual Review",
        }
        verdict = verdict_map.get(classification, "Unknown")

        # Generate summary
        summary = self._build_summary(classification, confidence, evidence)

        # Generate recommendations
        recommendations = self._build_recommendations(classification, confidence, evidence)

        return Explanation(
            verdict=verdict,
            confidence=confidence,
            summary=summary,
            evidence=evidence,
            recommendations=recommendations,
            metadata={
                "classification_raw": classification,
                "fusion_score": round(fusion_score, 4),
                "detail_level": self.detail.value,
                "num_evidence": len(evidence),
            },
        )

    def _build_summary(
        self,
        classification: str,
        confidence: float,
        evidence: list[Evidence],
    ) -> str:
        """Build a human-readable summary paragraph."""
        if classification == "fully_ai":
            if confidence >= 0.8:
                return (
                    "This text is very likely AI-generated. Both the structural recovery "
                    "analysis and stylometric fingerprinting strongly indicate machine authorship."
                )
            return (
                "This text appears to be AI-generated based on recovery pattern analysis, "
                "though confidence is moderate."
            )
        elif classification == "fully_human":
            return (
                "This text appears to be human-written. The D&R pipeline shows the "
                "distinctive structural unpredictability characteristic of human composition."
            )
        elif classification == "collaborative":
            return (
                "This text shows signs of mixed authorship — some sections exhibit "
                "AI-like patterns while others are consistent with human writing. "
                "This may indicate AI-assisted writing or human editing of AI output."
            )
        else:
            return (
                "The detection signals are inconclusive. The text falls in the "
                "uncertain zone between human and AI patterns. Additional context "
                "or a longer text sample may improve detection accuracy."
            )

    def _build_recommendations(
        self,
        classification: str,
        confidence: float,
        evidence: list[Evidence],
    ) -> list[str]:
        """Build actionable recommendations."""
        recs = []

        if confidence < 0.6:
            recs.append("Consider providing a longer text sample (>300 words) for more reliable detection.")

        if classification == "collaborative":
            recs.append("Review individual sections separately to identify which parts may be AI-generated.")

        if classification == "uncertain":
            recs.append("Try rerunning with different chunking parameters for a second opinion.")
            recs.append("Cross-reference with other detection tools for consensus.")

        if classification == "fully_ai":
            recs.append("Verify by checking for other AI indicators (e.g., repetitive phrasing, over-hedging).")

        return recs
