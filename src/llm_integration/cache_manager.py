"""
File-Based Cache Manager

Lightweight caching for LLM responses using local filesystem.
Designed for M1 8GB environments — no Redis dependency.

Falls back gracefully if cache dir is unavailable.

Author: SynthDetect Team
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("llm_integration.cache_manager")


class CacheManager:
    """
    File-based caching for LLM responses.

    Stores cached responses as JSON files in a local directory.
    Each cache entry includes the response, timestamp, and TTL.
    """

    def __init__(self, cache_dir: str = "data/cache", default_ttl: int = 86400):
        """
        Args:
            cache_dir: Directory to store cache files.
            default_ttl: Default time-to-live in seconds (24h).
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.enabled = True
            logger.info(f"Cache initialized at {self.cache_dir}")
        except OSError as e:
            logger.warning(f"Cache directory creation failed, caching disabled: {e}")
            self.enabled = False

    def get(self, key: str) -> Optional[str]:
        """
        Retrieve cached value.

        Args:
            key: Cache key (hash of input).

        Returns:
            Cached value string, or None if not found/expired.
        """
        if not self.enabled:
            return None

        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            self._misses += 1
            return None

        try:
            with open(cache_file, "r") as f:
                entry = json.load(f)

            # Check TTL
            if time.time() - entry["timestamp"] > entry["ttl"]:
                # Expired — remove and return None
                cache_file.unlink(missing_ok=True)
                self._misses += 1
                logger.debug(f"Cache expired for key {key[:16]}...")
                return None

            self._hits += 1
            logger.debug(f"Cache hit for key {key[:16]}...")
            return entry["value"]

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Cache read error for key {key[:16]}...: {e}")
            self._misses += 1
            return None

    def set(self, key: str, value: str, ttl: int = None) -> bool:
        """
        Store value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds. Uses default if not specified.

        Returns:
            True if successfully cached.
        """
        if not self.enabled:
            return False

        cache_file = self.cache_dir / f"{key}.json"

        try:
            entry = {
                "key": key,
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl or self.default_ttl,
            }

            with open(cache_file, "w") as f:
                json.dump(entry, f)

            logger.debug(f"Cached value for key {key[:16]}...")
            return True

        except OSError as e:
            logger.warning(f"Cache write error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a cached entry."""
        if not self.enabled:
            return False

        cache_file = self.cache_dir / f"{key}.json"
        try:
            cache_file.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def clear_all(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries cleared.
        """
        if not self.enabled:
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass

        logger.info(f"Cleared {count} cache entries")
        return count

    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "total_requests": total,
            "cache_size": sum(1 for _ in self.cache_dir.glob("*.json")) if self.enabled else 0,
        }

    @staticmethod
    def generate_key(provider: str, model: str, text: str) -> str:
        """
        Generate unique cache key from provider, model, and input text.

        Args:
            provider: LLM provider name.
            model: Model name.
            text: Input text.

        Returns:
            SHA-256 hash string.
        """
        key_material = f"{provider}:{model}:{text}"
        return hashlib.sha256(key_material.encode()).hexdigest()
