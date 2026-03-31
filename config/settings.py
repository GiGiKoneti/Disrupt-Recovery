"""
Global Settings Module

Loads configuration from .env file using Pydantic BaseSettings.
All configurable parameters are centralized here.

Author: SynthDetect Team
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- API Keys ---
    GOOGLE_API_KEY: str = Field(default="", description="Google Gemini API key")

    # --- LLM Configuration ---
    DEFAULT_LLM_PROVIDER: str = Field(default="google", description="LLM provider")
    DEFAULT_MODEL: str = Field(default="gemini-2.5-flash", description="Default model name")

    # --- Detection Thresholds ---
    DR_SIMILARITY_THRESHOLD: float = Field(
        default=0.85, description="D&R similarity threshold for AI detection"
    )
    FAID_CONFIDENCE_THRESHOLD: float = Field(
        default=0.65, description="FAID confidence threshold"
    )

    # --- Storage ---
    DATABASE_URL: str = Field(
        default="sqlite:///data/synthdetect.db", description="Database connection URL"
    )
    CACHE_DIR: str = Field(default="data/cache", description="File-based cache directory")
    ENABLE_CACHE: bool = Field(default=True, description="Enable LLM response caching")
    CACHE_TTL_SECONDS: int = Field(default=86400, description="Cache TTL in seconds (24h)")

    # --- Logging ---
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: str = Field(default="logs/synthdetect.log", description="Log file path")

    # --- Performance ---
    MAX_TEXT_LENGTH: int = Field(default=5000, description="Max text length in words")
    BATCH_SIZE: int = Field(default=8, description="Batch size for processing")
    TIMEOUT_SECONDS: int = Field(default=30, description="API timeout in seconds")

    # --- Development ---
    DEBUG: bool = Field(default=False, description="Debug mode")
    TESTING: bool = Field(default=False, description="Testing mode")
    SEED: int = Field(default=42, description="Random seed for reproducibility")

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }

    @property
    def cache_path(self) -> Path:
        """Resolved cache directory path."""
        path = PROJECT_ROOT / self.CACHE_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_path(self) -> Path:
        """Resolved log file path."""
        path = PROJECT_ROOT / Path(self.LOG_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# Singleton settings instance
settings = Settings()
