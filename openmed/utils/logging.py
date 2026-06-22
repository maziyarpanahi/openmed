"""Logging utilities for OpenMed."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
) -> None:
    """Set up logging for OpenMed.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_string: Custom format string for log messages.
        include_timestamp: Whether to include timestamp in log messages.
    """
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set specific logger levels
    logging.getLogger("openmed").setLevel(getattr(logging, level.upper()))
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for OpenMed modules.

    Args:
        name: Name of the logger (usually __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"openmed.{name}")


class OpenMedLogger:
    """Custom logger class with medical text processing context."""

    def __init__(self, name: str):
        """Initialize logger.

        Args:
            name: Logger name.
        """
        self.logger = get_logger(name)

    def log_model_loading(self, model_name: str, status: str = "started") -> None:
        """Log model loading events.

        Args:
            model_name: Name of the model being loaded.
            status: Status of the loading (started, completed, failed).
        """
        if status == "started":
            self.logger.info(f"Loading model: {model_name}")
        elif status == "completed":
            self.logger.info(f"Successfully loaded model: {model_name}")
        elif status == "failed":
            self.logger.error(f"Failed to load model: {model_name}")

    def log_processing(self, text_length: int, processing_time: float) -> None:
        """Log text processing metrics.

        Args:
            text_length: Length of processed text.
            processing_time: Time taken for processing.
        """
        self.logger.debug(
            f"Processed text of length {text_length} in {processing_time:.3f}s"
        )

    def log_predictions(self, num_entities: int, model_name: str) -> None:
        """Log prediction results.

        Args:
            num_entities: Number of entities predicted.
            model_name: Name of the model used.
        """
        self.logger.info(f"Model {model_name} predicted {num_entities} entities")
