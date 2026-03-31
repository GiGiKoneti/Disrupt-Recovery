"""
LLM Recovery Module for D&R Pipeline

Orchestrates calls to Google Gemini to recover shuffled text.
This is the "Recover" step in D&R.

Key Insight: AI-generated text, when shuffled and recovered by an LLM,
will be reconstructed with higher fidelity than human-written text
due to posterior concentration.

Author: SynthDetect Team
"""

from typing import Optional

from src.llm_integration.gemini_client import GeminiClient
from src.llm_integration.cache_manager import CacheManager
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("dr_pipeline.recovery")


class RecoveryEngine:
    """
    Handles LLM-based text recovery with caching and error handling.

    Given shuffled text, asks Gemini to reconstruct the logical order.
    Caches results to avoid redundant API calls.
    """

    RECOVERY_PROMPT_TEMPLATE = """The following text has been shuffled. Your task is to reconstruct the most logical original order.

Rules:
- Rearrange sentences/paragraphs to restore natural flow
- Do NOT add or remove content
- Do NOT paraphrase or modify wording
- Do NOT explain your reasoning
- Provide ONLY the reconstructed text

Shuffled text:
{shuffled_text}

Reconstructed text:"""

    def __init__(
        self,
        gemini_client: GeminiClient = None,
        cache_manager: CacheManager = None,
        enable_cache: bool = None,
    ):
        """
        Args:
            gemini_client: Pre-configured GeminiClient. Created if None.
            cache_manager: Pre-configured CacheManager. Created if None.
            enable_cache: Override cache setting.
        """
        self.gemini_client = gemini_client or GeminiClient()

        enable_cache = enable_cache if enable_cache is not None else settings.ENABLE_CACHE
        if enable_cache:
            self.cache_manager = cache_manager or CacheManager(
                cache_dir=str(settings.cache_path)
            )
        else:
            self.cache_manager = None

        logger.info(
            f"RecoveryEngine initialized: cache={'enabled' if self.cache_manager else 'disabled'}"
        )

    def recover(self, shuffled_text: str) -> str:
        """
        Recover (reconstruct) shuffled text using LLM.

        Args:
            shuffled_text: Shuffled text chunk.

        Returns:
            Recovered (reconstructed) text.

        Raises:
            RuntimeError: If LLM call fails after retries.
        """
        if not shuffled_text or not shuffled_text.strip():
            return shuffled_text

        # Check cache first
        cache_key = None
        if self.cache_manager:
            cache_key = CacheManager.generate_key(
                provider="google",
                model=self.gemini_client.model_name,
                text=shuffled_text,
            )
            cached_result = self.cache_manager.get(cache_key)

            if cached_result:
                logger.info("Recovery result found in cache")
                return cached_result

        # Make LLM call
        prompt = self.RECOVERY_PROMPT_TEMPLATE.format(shuffled_text=shuffled_text)

        recovered_text = self.gemini_client.generate(
            prompt=prompt,
            temperature=0.0,  # Deterministic for reproducibility
            max_output_tokens=2000,
        )

        # Cache result
        if self.cache_manager and cache_key:
            self.cache_manager.set(
                cache_key, recovered_text, ttl=settings.CACHE_TTL_SECONDS
            )

        logger.info(
            f"Recovery complete: {len(shuffled_text)} chars → {len(recovered_text)} chars"
        )

        return recovered_text

    def recover_batch(self, shuffled_chunks: list) -> list:
        """
        Recover multiple chunks sequentially.

        Args:
            shuffled_chunks: List of shuffled text chunks.

        Returns:
            List of recovered text chunks.
        """
        recovered = []
        for i, chunk in enumerate(shuffled_chunks):
            logger.info(f"Recovering chunk {i + 1}/{len(shuffled_chunks)}")
            recovered.append(self.recover(chunk))
        return recovered

    def usage_stats(self) -> dict:
        """Return combined LLM and cache usage statistics."""
        stats = {"llm": self.gemini_client.usage_stats()}
        if self.cache_manager:
            stats["cache"] = self.cache_manager.stats()
        return stats
