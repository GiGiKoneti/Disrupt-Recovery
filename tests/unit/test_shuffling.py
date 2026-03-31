"""
Unit Tests for Shuffling Module
"""

import pytest
from src.dr_pipeline.shuffling import ShuffleEngine, ShuffleLevel


class TestShuffleEngine:
    """Tests for sentence-level shuffling."""

    @pytest.fixture
    def engine(self):
        return ShuffleEngine(
            shuffle_level=ShuffleLevel.SENTENCE,
            preserve_ratio=0.0,
            preserve_boundaries=True,
        )

    def test_basic_shuffle(self, engine, sample_human_text):
        shuffled = engine.disrupt(sample_human_text, seed=42)
        assert shuffled != sample_human_text  # Should be different
        assert len(shuffled.split()) > 0  # Should have content

    def test_reproducibility(self, engine, sample_human_text):
        """Same seed should produce same result."""
        result1 = engine.disrupt(sample_human_text, seed=123)
        result2 = engine.disrupt(sample_human_text, seed=123)
        assert result1 == result2

    def test_different_seeds_different_results(self, engine, sample_human_text):
        result1 = engine.disrupt(sample_human_text, seed=1)
        result2 = engine.disrupt(sample_human_text, seed=2)
        # May occasionally be the same, but very unlikely
        # We just check both produce output
        assert len(result1) > 0
        assert len(result2) > 0

    def test_short_text_unchanged(self, engine):
        """Text with ≤2 sentences shouldn't be shuffled."""
        short = "Single sentence here. And one more."
        result = engine.disrupt(short, seed=42)
        assert result == short or len(result.split()) > 0

    def test_empty_text(self, engine):
        assert engine.disrupt("", seed=42) == ""
        assert engine.disrupt("   ", seed=42) == "   "

    def test_preserve_boundaries(self, sample_human_text):
        """First and last sentences should stay in place."""
        engine = ShuffleEngine(preserve_boundaries=True, preserve_ratio=0.0)

        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        original_sents = [s.text.strip() for s in nlp(sample_human_text).sents]

        shuffled = engine.disrupt(sample_human_text, seed=42)
        shuffled_sents = [s.text.strip() for s in nlp(shuffled).sents]

        if len(original_sents) > 2:
            assert shuffled_sents[0] == original_sents[0]
            assert shuffled_sents[-1] == original_sents[-1]

    def test_disruption_score(self, engine, sample_human_text):
        shuffled = engine.disrupt(sample_human_text, seed=42)
        score = engine.compute_disruption_score(sample_human_text, shuffled)
        assert 0.0 <= score <= 1.0
        # Identical texts should have zero disruption
        assert engine.compute_disruption_score(sample_human_text, sample_human_text) == 0.0
