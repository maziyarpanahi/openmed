"""Utility functions for OpenMed."""

from .deprecation import deprecated
from .logging import get_logger, setup_logging
from .profiling import (
    BatchMetrics,
    InferenceMetrics,
    PeakRSSMeasurement,
    Profiler,
    ProfileReport,
    Timer,
    TimingResult,
    disable_profiling,
    enable_profiling,
    get_peak_rss_bytes,
    get_profile_report,
    get_profiler,
    measure_peak_rss,
    profile,
    timed,
)
from .validation import (
    InputValidationError,
    validate_input,
    validate_language,
    validate_model_name,
    validate_text_input,
)

__all__ = [
    "deprecated",
    "setup_logging",
    "get_logger",
    "InputValidationError",
    "validate_input",
    "validate_language",
    "validate_model_name",
    "validate_text_input",
    # Profiling utilities
    "Profiler",
    "ProfileReport",
    "TimingResult",
    "InferenceMetrics",
    "PeakRSSMeasurement",
    "BatchMetrics",
    "Timer",
    "get_profiler",
    "enable_profiling",
    "get_peak_rss_bytes",
    "disable_profiling",
    "get_profile_report",
    "measure_peak_rss",
    "profile",
    "timed",
]
