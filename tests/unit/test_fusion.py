"""
Unit Tests for Fusion Layer
"""

import pytest
from src.core.fusion_layer import FusionLayer, Classification
from src.dr_pipeline.dr_detector import DRResult
from src.faid_pipeline.faid_detector import FAIDResult
from src.faid_pipeline.attribution import AttributionResult


class TestFusionLayer:

    @pytest.fixture
    def fusion(self):
        return FusionLayer()

    def _make_dr_result(self, score=0.9, variance=0.01, label="ai_generated"):
        return DRResult(
            score=score, variance=variance, chunk_scores=[],
            num_chunks=3, label=label, confidence=0.9,
        )

    def _make_faid_result(self, is_ai=True, confidence=0.8, source="gpt4"):
        return FAIDResult(
            is_ai=is_ai, confidence=confidence, predicted_source=source,
            attribution=AttributionResult(
                predicted_source=source, confidence=confidence,
                neighbor_labels=[], neighbor_distances=[], label_distribution={},
            ),
        )

    def test_strong_ai_signal(self, fusion):
        dr = self._make_dr_result(score=0.92, variance=0.01)
        faid = self._make_faid_result(is_ai=True, confidence=0.85)
        result = fusion.fuse(dr, faid)
        assert result.classification == Classification.FULLY_AI
        assert result.confidence > 0.7

    def test_strong_human_signal(self, fusion):
        dr = self._make_dr_result(score=0.30, variance=0.01, label="human_written")
        faid = self._make_faid_result(is_ai=False, confidence=0.9, source="human")
        result = fusion.fuse(dr, faid)
        assert result.classification == Classification.FULLY_HUMAN

    def test_collaborative_high_variance(self, fusion):
        dr = self._make_dr_result(score=0.75, variance=0.08)  # High variance
        faid = self._make_faid_result(is_ai=True, confidence=0.6)
        result = fusion.fuse(dr, faid)
        assert result.classification == Classification.COLLABORATIVE

    def test_explanation_not_empty(self, fusion):
        dr = self._make_dr_result()
        faid = self._make_faid_result()
        result = fusion.fuse(dr, faid)
        assert len(result.explanation) > 0

    def test_fusion_score_range(self, fusion):
        dr = self._make_dr_result(score=0.5)
        faid = self._make_faid_result(confidence=0.5)
        result = fusion.fuse(dr, faid)
        assert 0.0 <= result.fusion_score <= 1.0
