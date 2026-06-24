"""Core functionality for OpenMed package."""

from .audit import AuditReport, AuditSignature, AuditSpan, DetectorInfo
from .config import (
    PROFILE_PRESETS,
    OpenMedConfig,
    delete_profile,
    get_profile,
    list_profiles,
    load_config_with_profile,
    save_profile,
)
from .models import ModelLoader, load_model
from .script_detect import (
    SCRIPT_LANGUAGE_HINTS,
    SUPPORTED_SCRIPTS,
    UNKNOWN_SCRIPT,
    candidate_languages_for_script,
    detect_script,
    segment_by_script,
)

__all__ = [
    "ModelLoader",
    "load_model",
    "OpenMedConfig",
    "AuditReport",
    "AuditSignature",
    "AuditSpan",
    "DetectorInfo",
    "PROFILE_PRESETS",
    "list_profiles",
    "get_profile",
    "save_profile",
    "delete_profile",
    "load_config_with_profile",
    "SCRIPT_LANGUAGE_HINTS",
    "SUPPORTED_SCRIPTS",
    "UNKNOWN_SCRIPT",
    "candidate_languages_for_script",
    "detect_script",
    "segment_by_script",
]
