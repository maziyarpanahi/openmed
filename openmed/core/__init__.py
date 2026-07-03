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
from .custom_recognizer import CustomRecognizer
from .model_search import ModelQuery, ModelSearchResult, search_models
from .models import ModelLoader, load_model
from .offline import OfflineModeError
from .redaction_preview import redaction_preview, render_redaction_preview
from .script_detect import (
    SCRIPT_LANGUAGE_HINTS,
    SUPPORTED_SCRIPTS,
    UNKNOWN_SCRIPT,
    candidate_languages_for_script,
    detect_script,
    segment_by_script,
)
from .surrogate_vault import (
    InMemorySurrogateStore,
    JsonFileSurrogateStore,
    SurrogateEntry,
    SurrogateKey,
    SurrogateVault,
)

__all__ = [
    "ModelLoader",
    "load_model",
    "ModelQuery",
    "ModelSearchResult",
    "search_models",
    "OpenMedConfig",
    "CustomRecognizer",
    "AuditReport",
    "AuditSignature",
    "AuditSpan",
    "DetectorInfo",
    "redaction_preview",
    "render_redaction_preview",
    "SurrogateVault",
    "SurrogateKey",
    "SurrogateEntry",
    "InMemorySurrogateStore",
    "JsonFileSurrogateStore",
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
    "OfflineModeError",
]
