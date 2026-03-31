"""
Google Gemini Client

Wrapper around the google-genai SDK for Gemini 2.5 Flash.
Handles API calls, error handling, token tracking, and retries.

Author: SynthDetect Team
"""

import time
from typing import Optional

from google import genai
from google.genai.types import GenerateContentConfig

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("llm_integration.gemini_client")


class GeminiClient:
    """
    Client for Google Gemini API.

    Provides text generation with retry logic,
    token usage tracking, and cost estimation.
    """

    # Approximate pricing for Gemini 2.5 Flash (per 1M tokens)
    COST_PER_1M_INPUT_TOKENS = 0.15   # $0.15
    COST_PER_1M_OUTPUT_TOKENS = 0.60  # $0.60

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        max_retries: int = 3,
        timeout: int = None,
    ):
        """
        Args:
            api_key: Google API key. Falls back to settings.
            model_name: Model name (default: gemini-2.5-flash).
            max_retries: Max retry attempts on failure.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or settings.GOOGLE_API_KEY
        self.model_name = model_name or settings.DEFAULT_MODEL
        self.max_retries = max_retries
        self.timeout = timeout or settings.TIMEOUT_SECONDS

        # Initialize client
        self.client = genai.Client(api_key=self.api_key)

        # Usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0

        logger.info(f"GeminiClient initialized: model={self.model_name}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 2000,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Generate text using Gemini.

        Args:
            prompt: User prompt text.
            temperature: Sampling temperature (0.0 = deterministic).
            max_output_tokens: Maximum tokens in response.
            system_instruction: Optional system-level instruction.

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If all retries fail.
        """
        config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )

                # Track usage
                self._total_calls += 1
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    self._total_input_tokens += getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    )
                    self._total_output_tokens += getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    )

                result = response.text.strip()
                logger.debug(
                    f"Gemini call successful (attempt {attempt}): "
                    f"{len(result)} chars returned"
                )
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Gemini API call failed (attempt {attempt}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** (attempt - 1)
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        raise RuntimeError(
            f"Gemini API call failed after {self.max_retries} attempts: {last_error}"
        )

    def usage_stats(self) -> dict:
        """Return cumulative token usage and estimated cost."""
        input_cost = (self._total_input_tokens / 1_000_000) * self.COST_PER_1M_INPUT_TOKENS
        output_cost = (self._total_output_tokens / 1_000_000) * self.COST_PER_1M_OUTPUT_TOKENS

        return {
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 6),
        }
