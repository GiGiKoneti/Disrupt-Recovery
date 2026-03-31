"""
Unit Tests for Feature Extraction
"""

import pytest
import numpy as np
from src.faid_pipeline.feature_extraction import FeatureExtractor


class TestFeatureExtractor:
    """Tests for multi-level feature extraction."""

    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()

    def test_extract_features_returns_all_categories(self, extractor, sample_human_text):
        features = extractor.extract_features(sample_human_text)
        assert "lexical" in features
        assert "syntactic" in features
        assert "semantic" in features
        assert "stylometric" in features

    def test_lexical_features_dimension(self, extractor, sample_human_text):
        features = extractor.extract_features(sample_human_text)
        assert features["lexical"].shape == (10,)

    def test_stylometric_features_dimension(self, extractor, sample_human_text):
        features = extractor.extract_features(sample_human_text)
        assert features["stylometric"].shape == (15,)

    def test_flat_vector(self, extractor, sample_human_text):
        vec = extractor.extract_flat_vector(sample_human_text)
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(vec) > 0

    def test_ttr_reasonable(self, extractor, sample_human_text):
        features = extractor.extract_features(sample_human_text)
        ttr = features["lexical"][0]
        assert 0.0 < ttr <= 1.0  # TTR must be in (0, 1]

    def test_different_texts_different_features(self, extractor, sample_human_text, sample_ai_text):
        f1 = extractor.extract_flat_vector(sample_human_text)
        f2 = extractor.extract_flat_vector(sample_ai_text)
        # Features should differ for different texts
        assert not np.allclose(f1, f2)
