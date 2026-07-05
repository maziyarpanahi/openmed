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
    ZERO_WIDTH_CHARS,
    DetectionNormalization,
    candidate_languages_for_script,
    detect_script,
    normalize_for_pii_detection,
    segment_by_script,
)
from .surrogate_vault import (
    ENCRYPTION_SCHEME,
    InMemorySurrogateStore,
    JsonFileSurrogateStore,
    SurrogateEntry,
    SurrogateKey,
    SurrogateSource,
    SurrogateVault,
    VaultConsistencyReport,
    VaultRotationResult,
)
from .telemetry import (
    PipelineTelemetry,
    StageMetrics,
    otel_available,
    telemetry_enabled_from_env,
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
    "SurrogateSource",
    "VaultConsistencyReport",
    "VaultRotationResult",
    "InMemorySurrogateStore",
    "JsonFileSurrogateStore",
    "ENCRYPTION_SCHEME",
    "PROFILE_PRESETS",
    "list_profiles",
    "get_profile",
    "save_profile",
    "delete_profile",
    "load_config_with_profile",
    "SCRIPT_LANGUAGE_HINTS",
    "SUPPORTED_SCRIPTS",
    "UNKNOWN_SCRIPT",
    "ZERO_WIDTH_CHARS",
    "DetectionNormalization",
    "candidate_languages_for_script",
    "detect_script",
    "normalize_for_pii_detection",
    "segment_by_script",
    "OfflineModeError",
    "PipelineTelemetry",
    "StageMetrics",
    "otel_available",
    "telemetry_enabled_from_env",
]
