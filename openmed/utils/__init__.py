"""Utility functions for OpenMed."""

from .logging import setup_logging, get_logger
from .validation import validate_input, validate_model_name
from .profiling import (
    Profiler,
    ProfileReport,
    TimingResult,
    InferenceMetrics,
    BatchMetrics,
    Timer,
    get_profiler,
    enable_profiling,
    disable_profiling,
    get_profile_report,
    profile,
    timed,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "validate_input",
    "validate_model_name",
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
