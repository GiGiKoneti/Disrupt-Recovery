"""
Logging Setup Module

Configures structured logging from logging_config.yaml.

Author: SynthDetect Team
"""

import logging
import logging.config
from pathlib import Path

import yaml


_LOGGING_CONFIGURED = False


def setup_logging(config_path: str = None, level: str = None) -> None:
    """
    Configure logging from YAML config file.

    Args:
        config_path: Path to logging_config.yaml. If None, uses default.
        level: Override log level (DEBUG, INFO, WARNING, ERROR).
    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        return

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "logging_config.yaml"
    else:
        config_path = Path(config_path)

    # Ensure log directory exists
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Override level if specified
        if level:
            for handler in config.get("handlers", {}).values():
                handler["level"] = level
            for logger in config.get("loggers", {}).values():
                logger["level"] = level

        logging.config.dictConfig(config)
    else:
        # Fallback to basic config
        logging.basicConfig(
            level=getattr(logging, level or "INFO"),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with proper namespace.

    Args:
        name: Logger name (e.g., 'dr_pipeline.chunking')

    Returns:
        Configured logger instance.
    """
    setup_logging()
    return logging.getLogger(f"synthdetect.{name}")
