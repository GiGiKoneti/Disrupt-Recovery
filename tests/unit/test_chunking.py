"""
Unit Tests for Chunking Module

Tests all three chunking strategies: fixed, semantic, sliding window.
"""

import pytest
from src.dr_pipeline.chunking import ChunkingEngine, ChunkStrategy


class TestFixedChunking:
    """Tests for fixed-length chunking."""

    def test_basic_chunking(self):
        engine = ChunkingEngine(chunk_size=5, strategy=ChunkStrategy.FIXED)
        text = "one two three four five six seven eight nine ten"
        chunks = engine.chunk_text(text)
        assert len(chunks) == 2
        assert chunks[0] == "one two three four five"
        assert chunks[1] == "six seven eight nine ten"

    def test_uneven_split(self):
        engine = ChunkingEngine(chunk_size=3, strategy=ChunkStrategy.FIXED)
        text = "a b c d e f g"
        chunks = engine.chunk_text(text)
        assert len(chunks) == 3
        assert chunks[2] == "g"

    def test_empty_text(self):
        engine = ChunkingEngine(chunk_size=10, strategy=ChunkStrategy.FIXED)
        assert engine.chunk_text("") == []
        assert engine.chunk_text("   ") == []

    def test_single_chunk(self):
        engine = ChunkingEngine(chunk_size=100, strategy=ChunkStrategy.FIXED)
        text = "short text here"
        chunks = engine.chunk_text(text)
        assert len(chunks) == 1


class TestSemanticChunking:
    """Tests for semantic (sentence-aware) chunking."""

    def test_respects_sentences(self):
        engine = ChunkingEngine(chunk_size=10, strategy=ChunkStrategy.SEMANTIC)
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = engine.chunk_text(text)
        # Each chunk should contain complete sentences
        for chunk in chunks:
            assert chunk.rstrip().endswith(".")

    def test_short_text_single_chunk(self):
        engine = ChunkingEngine(chunk_size=100, strategy=ChunkStrategy.SEMANTIC)
        text = "This is a short text. It has two sentences."
        chunks = engine.chunk_text(text)
        assert len(chunks) == 1

    @pytest.mark.slow
    def test_long_text(self, sample_human_text):
        engine = ChunkingEngine(chunk_size=50, strategy=ChunkStrategy.SEMANTIC)
        chunks = engine.chunk_text(sample_human_text)
        assert len(chunks) >= 2
        # Verify no content is lost
        total_words = sum(len(c.split()) for c in chunks)
        original_words = len(sample_human_text.split())
        assert abs(total_words - original_words) <= 5  # Small tolerance


class TestSlidingWindowChunking:
    """Tests for sliding window chunking."""

    def test_overlap_exists(self):
        engine = ChunkingEngine(
            chunk_size=5, overlap=2, strategy=ChunkStrategy.SLIDING
        )
        text = "one two three four five six seven eight nine ten"
        chunks = engine.chunk_text(text)
        assert len(chunks) >= 2
        # Check overlap: last 2 words of chunk 0 should appear in chunk 1
        words_0 = chunks[0].split()
        words_1 = chunks[1].split()
        # Step size = 5-2 = 3, so chunk 1 starts at word 3
        assert words_0[3] == words_1[0]

    def test_small_final_chunk_skipped(self):
        engine = ChunkingEngine(
            chunk_size=10, overlap=2, strategy=ChunkStrategy.SLIDING
        )
        text = " ".join([f"word{i}" for i in range(12)])
        chunks = engine.chunk_text(text)
        # The last chunk would be very small and should be skipped
        for chunk in chunks:
            assert len(chunk.split()) >= 5  # At least chunk_size/2
