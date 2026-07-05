"""Utility functions for OpenMed."""

from .gateway import (
    GatewayLimits,
    InputValidationError,
    NormalizedInput,
    ensure_valid_encoding,
    get_default_limits,
    normalize_input,
    normalize_text,
    validate_language,
)
from .logging import get_logger, setup_logging
from .profiling import (
    BatchMetrics,
    InferenceMetrics,
    Profiler,
    ProfileReport,
    Timer,
    TimingResult,
    disable_profiling,
    enable_profiling,
    get_profile_report,
    get_profiler,
    profile,
    timed,
)
from .validation import validate_input, validate_model_name

__all__ = [
    "setup_logging",
    "get_logger",
    "validate_input",
    "validate_model_name",
    # Shared input gateway
    "InputValidationError",
    "GatewayLimits",
    "NormalizedInput",
    "ensure_valid_encoding",
    "get_default_limits",
    "normalize_text",
    "validate_language",
    "normalize_input",
    # Profiling utilities
    "Profiler",
    "ProfileReport",
    "TimingResult",
    "InferenceMetrics",
    "BatchMetrics",
    "Timer",
    "get_profiler",
    "enable_profiling",
    "disable_profiling",
    "get_profile_report",
    "profile",
    "timed",
]
