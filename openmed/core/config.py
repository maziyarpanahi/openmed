"""Configuration management for OpenMed."""

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from .offline import (
    OFFLINE_ENV_VAR,
    configure_offline_mode,
    env_flag_enabled,
)

# Environment variable used to override the config file location
CONFIG_ENV_VAR = "OPENMED_CONFIG"

# Environment variable for active profile
PROFILE_ENV_VAR = "OPENMED_PROFILE"

# Official CPU-oriented INT8 artifact used by the low-resource profile.
LOW_RESOURCE_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-onnx-android"
LOW_RESOURCE_PII_REVISION = "82f57fcab68125b05f1aa9fdd41319732358311b"

# Environment variable for the PyTorch/Transformers attention backend.
TORCH_ATTENTION_BACKEND_ENV_VAR = "OPENMED_TORCH_ATTENTION_BACKEND"

CHINESE_SEGMENTATION_BACKEND_ENV_VAR = "OPENMED_CHINESE_SEGMENTATION_BACKEND"
CHINESE_USER_DICT_ENV_VAR = "OPENMED_CHINESE_USER_DICT"
CHINESE_PKUSEG_DOMAIN_ENV_VAR = "OPENMED_CHINESE_PKUSEG_DOMAIN"

_xdg_config = os.getenv("XDG_CONFIG_HOME")
if _xdg_config:
    _default_config_root = Path(_xdg_config)
else:
    _default_config_root = Path.home() / ".config"

DEFAULT_CONFIG_DIR = _default_config_root / "openmed"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"
PROFILES_DIR = DEFAULT_CONFIG_DIR / "profiles"


class ConfigValidationError(ValueError):
    """Aggregate one or more schema violations without echoing config values."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("Invalid OpenMed configuration: " + "; ".join(errors))


def config_schema_path() -> Path:
    """Return the installed Draft 2020-12 configuration schema path."""

    return Path(__file__).with_name("config.schema.json")


@lru_cache(maxsize=1)
def _config_schema() -> Dict[str, Any]:
    """Load the bundled configuration schema once per process."""

    with config_schema_path().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError("OpenMed configuration schema must be a JSON object")
    return payload


def _validate_config_mapping(
    data: Mapping[str, Any],
    *,
    profile: bool = False,
) -> None:
    """Validate a mapping against the controlled bundled-schema subset."""

    schema = _config_schema()
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise RuntimeError("OpenMed configuration schema has no properties object")

    if profile:
        declared_profile_keys = schema.get("x-profile-keys")
        if not isinstance(declared_profile_keys, list) or not all(
            isinstance(key, str) for key in declared_profile_keys
        ):
            raise RuntimeError(
                "OpenMed configuration schema has no valid x-profile-keys list"
            )
        allowed_keys = set(declared_profile_keys)
    else:
        allowed_keys = set(properties)

    errors: List[str] = []
    for key in sorted(data, key=str):
        if not isinstance(key, str) or key not in allowed_keys:
            errors.append(f"{key}: unknown configuration key")
            continue
        property_schema = properties.get(key)
        if not isinstance(property_schema, dict):
            errors.append(f"{key}: schema definition is invalid")
            continue
        errors.extend(_property_validation_errors(key, data[key], property_schema))

    if errors:
        raise ConfigValidationError(errors)


def _config_validation_view(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the values as ``OpenMedConfig.__post_init__`` normalizes them.

    Validation must run before dataclass construction so malformed mappings do
    not reach field-specific operations or value-bearing legacy exceptions.
    Preserve the established whitespace, case, and numeric-string coercions in
    the validation-only copy; the constructor remains responsible for applying
    those normalizations to the actual instance.
    """

    normalized = dict(data)
    for key in ("chinese_segmentation_backend", "remote_inference_protocol"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = value.strip().lower()

    chinese_domain = normalized.get("chinese_pkuseg_domain")
    if isinstance(chinese_domain, str):
        normalized["chinese_pkuseg_domain"] = chinese_domain.strip()

    for key in (
        "indic_name_similarity_threshold",
        "remote_inference_timeout_seconds",
    ):
        value = normalized.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError, OverflowError):
            pass
    return normalized


def _property_validation_errors(
    path: str,
    value: Any,
    schema: Mapping[str, Any],
) -> List[str]:
    errors: List[str] = []
    declared_types = schema.get("type")
    if isinstance(declared_types, str):
        allowed_types = (declared_types,)
    elif isinstance(declared_types, list) and all(
        isinstance(item, str) for item in declared_types
    ):
        allowed_types = tuple(declared_types)
    else:
        return [f"{path}: schema type declaration is invalid"]

    if not any(_matches_json_type(value, name) for name in allowed_types):
        expected = " or ".join(allowed_types)
        return [f"{path}: expected {expected}, got {_json_type_name(value)}"]

    if value is None:
        return errors

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path}: value is not in the allowed set")

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            errors.append(f"{path}: string is shorter than the minimum length")

    if _matches_json_type(value, "number"):
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return [f"{path}: number must be finite"]
        if not math.isfinite(numeric):
            return [f"{path}: number must be finite"]
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and numeric < float(minimum):
            errors.append(f"{path}: number is below the minimum")
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and numeric > float(maximum):
            errors.append(f"{path}: number exceeds the maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, (int, float)) and numeric <= float(
            exclusive_minimum
        ):
            errors.append(f"{path}: number must exceed the exclusive minimum")

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(
                    _property_validation_errors(
                        f"{path}[{index}]",
                        item,
                        item_schema,
                    )
                )
    return errors


def _matches_json_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, Mapping)
    return False


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


# Built-in profile presets
PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "dev": {
        "log_level": "DEBUG",
        "timeout": 600,
        "use_medical_tokenizer": True,
    },
    "prod": {
        "log_level": "WARNING",
        "timeout": 300,
        "use_medical_tokenizer": True,
    },
    "test": {
        "log_level": "DEBUG",
        "timeout": 60,
        "use_medical_tokenizer": False,
    },
    "fast": {
        "log_level": "WARNING",
        "timeout": 120,
        "use_medical_tokenizer": False,
    },
    "low_resource": {
        "backend": "onnx",
        "batch_size": 1,
        "device": "cpu",
        "lazy_model_loading": True,
        "log_level": "WARNING",
        "num_workers": 1,
        "onnx_intra_op_num_threads": 2,
        "onnx_variant": "int8",
        "pii_model": LOW_RESOURCE_PII_MODEL,
        "pii_model_revision": LOW_RESOURCE_PII_REVISION,
        "timeout": 900,
        "use_medical_tokenizer": False,
    },
}


@dataclass
class OpenMedConfig:
    """Configuration class for OpenMed package."""

    # Default organization on HuggingFace Hub
    default_org: str = "OpenMed"

    # Model cache directory
    cache_dir: Optional[str] = None

    # Device preference
    device: Optional[str] = None

    # Token for private models (if needed)
    hf_token: Optional[str] = None

    # Logging level
    log_level: str = "INFO"

    # Model loading timeout
    timeout: int = 300

    # Medical-aware tokenizer toggle (output remapping only; does not change model tokenization)
    use_medical_tokenizer: bool = True

    # Optional list of terms to keep intact when remapping output onto medical tokens
    medical_tokenizer_exceptions: Optional[List[str]] = None

    # Chinese word segmentation. jieba is lightweight and bundled; the other
    # backends are optional and load only when explicitly selected.
    chinese_segmentation_backend: str = "jieba"
    chinese_user_dict_path: Optional[str] = None
    chinese_pkuseg_domain: str = "medicine"

    # Protect common clinical vocabulary from PERSON/LOCATION/ORGANIZATION over-redaction
    clinical_protect_enabled: bool = True
    clinical_protect_terms: Optional[List[str]] = None
    clinical_protect_use_builtin: bool = True

    # Inference backend: None (auto-detect), "hf", "mlx", "onnx", or "remote"
    backend: Optional[str] = None

    # Runtime resource controls. None preserves backend-specific defaults.
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    lazy_model_loading: bool = True

    # Optional profile-selected PII model and ONNX Runtime tuning.
    pii_model: Optional[str] = None
    pii_model_revision: Optional[str] = None
    onnx_variant: str = "auto"
    onnx_intra_op_num_threads: Optional[int] = None

    # PyTorch/Transformers attention backend: auto, flash_attention_2, sdpa, or eager
    torch_attention_backend: str = "auto"

    # Optional load-time bitsandbytes 4-bit quantization for CUDA torch loads
    load_in_4bit: bool = False
    bnb_4bit_use_double_quant: bool = True

    # Cache-only, no-egress mode for inference and de-identification
    local_only: bool = False

    # CJK width normalization convention: "cjk" (Latin/digits/symbols to
    # half-width, Han left as-is) or "nfkc" (strict per-character NFKC).
    cjk_width_convention: str = "cjk"

    # Optional OpenCC pre-pass for Chinese text. None disables conversion;
    # otherwise mixed variants are canonicalized before model inference.
    chinese_target_script: Optional[str] = None

    # Link Indic personal-name spellings through a transliteration-safe vault
    # key. Disabled by default to preserve existing pseudonymization behavior.
    transliteration_aware_name_matching: bool = False
    indic_name_similarity_threshold: float = 0.80

    # Active profile name (if any)
    profile: Optional[str] = None

    # Explicit KServe V2 / Triton inference settings. These are appended to
    # preserve the positional order of the established public constructor.
    # Tokenization and decoding stay local; the endpoint receives tensors.
    remote_inference_endpoint: Optional[str] = None
    remote_inference_protocol: str = "http"
    remote_inference_model_name: Optional[str] = None
    remote_inference_model_version: Optional[str] = None
    remote_inference_tokenizer: Optional[str] = None
    remote_inference_timeout_seconds: float = 30.0
    remote_inference_verify_tls: bool = True

    def __post_init__(self):
        """Post-initialization to set default values."""
        if self.cache_dir is None:
            self.cache_dir = os.path.expanduser("~/.cache/openmed")

        # An environment-selected built-in profile must affect runtime settings,
        # not merely record its name. Apply it before field-specific environment
        # overrides so those overrides retain their existing precedence.
        env_profile = os.getenv(PROFILE_ENV_VAR)
        if env_profile and self.profile is None:
            profile_data = PROFILE_PRESETS.get(env_profile)
            if profile_data is not None:
                for key, value in profile_data.items():
                    setattr(self, key, value)
            self.profile = env_profile

        if not isinstance(self.transliteration_aware_name_matching, bool):
            raise TypeError("transliteration_aware_name_matching must be a boolean")
        if isinstance(self.indic_name_similarity_threshold, bool):
            raise TypeError("indic_name_similarity_threshold must be a real number")
        self.indic_name_similarity_threshold = float(
            self.indic_name_similarity_threshold
        )

        if self.cjk_width_convention not in {"cjk", "nfkc"}:
            raise ValueError(
                "cjk_width_convention must be 'cjk' or 'nfkc', got "
                f"{self.cjk_width_convention!r}"
            )

        if not isinstance(self.remote_inference_protocol, str):
            raise TypeError("remote_inference_protocol must be a string")
        self.remote_inference_protocol = self.remote_inference_protocol.strip().lower()
        if self.remote_inference_protocol not in {"http", "grpc"}:
            raise ValueError(
                "remote_inference_protocol must be 'http' or 'grpc', got "
                f"{self.remote_inference_protocol!r}"
            )
        if isinstance(self.remote_inference_timeout_seconds, bool):
            raise TypeError("remote_inference_timeout_seconds must be a real number")
        self.remote_inference_timeout_seconds = float(
            self.remote_inference_timeout_seconds
        )
        if (
            not math.isfinite(self.remote_inference_timeout_seconds)
            or self.remote_inference_timeout_seconds <= 0
        ):
            raise ValueError(
                "remote_inference_timeout_seconds must be positive and finite"
            )
        if not isinstance(self.remote_inference_verify_tls, bool):
            raise TypeError("remote_inference_verify_tls must be a boolean")

        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.onnx_variant not in {"auto", "int8", "fp16", "fp32"}:
            raise ValueError("onnx_variant must be auto, int8, fp16, or fp32")
        if (
            self.onnx_intra_op_num_threads is not None
            and self.onnx_intra_op_num_threads <= 0
        ):
            raise ValueError("onnx_intra_op_num_threads must be positive")
        if self.chinese_target_script not in {None, "simplified", "traditional"}:
            raise ValueError(
                "chinese_target_script must be None, 'simplified', or "
                f"'traditional', got {self.chinese_target_script!r}"
            )
        if not 0.5 <= self.indic_name_similarity_threshold <= 1.0:
            raise ValueError(
                "indic_name_similarity_threshold must be between 0.5 and 1.0"
            )

        if self.hf_token is None:
            self.hf_token = os.getenv("HF_TOKEN")

        env_use_med_tok = os.getenv("OPENMED_USE_MEDICAL_TOKENIZER")
        if env_use_med_tok is not None:
            self.use_medical_tokenizer = env_use_med_tok.lower() not in {
                "0",
                "false",
                "no",
            }

        env_exceptions = os.getenv("OPENMED_MEDICAL_TOKENIZER_EXCEPTIONS")
        if env_exceptions:
            self.medical_tokenizer_exceptions = [
                item.strip() for item in env_exceptions.split(",") if item.strip()
            ]

        env_chinese_backend = os.getenv(CHINESE_SEGMENTATION_BACKEND_ENV_VAR)
        if env_chinese_backend:
            self.chinese_segmentation_backend = env_chinese_backend
        self.chinese_segmentation_backend = (
            self.chinese_segmentation_backend.strip().lower()
        )
        if self.chinese_segmentation_backend not in {"jieba", "pkuseg", "hanlp"}:
            raise ValueError(
                "chinese_segmentation_backend must be jieba, pkuseg, or hanlp"
            )

        env_chinese_user_dict = os.getenv(CHINESE_USER_DICT_ENV_VAR)
        if env_chinese_user_dict:
            self.chinese_user_dict_path = env_chinese_user_dict

        env_pkuseg_domain = os.getenv(CHINESE_PKUSEG_DOMAIN_ENV_VAR)
        if env_pkuseg_domain:
            self.chinese_pkuseg_domain = env_pkuseg_domain
        self.chinese_pkuseg_domain = self.chinese_pkuseg_domain.strip()
        if not self.chinese_pkuseg_domain:
            raise ValueError("chinese_pkuseg_domain must not be empty")

        env_protect = os.getenv("OPENMED_CLINICAL_PROTECT")
        if env_protect is not None:
            self.clinical_protect_enabled = env_protect.lower() not in {
                "0",
                "false",
                "no",
            }

        env_protect_terms = os.getenv("OPENMED_CLINICAL_PROTECT_TERMS")
        if env_protect_terms:
            self.clinical_protect_terms = [
                item.strip() for item in env_protect_terms.split(",") if item.strip()
            ]

        env_protect_builtin = os.getenv("OPENMED_CLINICAL_PROTECT_USE_BUILTIN")
        if env_protect_builtin is not None:
            self.clinical_protect_use_builtin = env_protect_builtin.lower() not in {
                "0",
                "false",
                "no",
            }

        env_attention_backend = os.getenv(TORCH_ATTENTION_BACKEND_ENV_VAR)
        if env_attention_backend is not None:
            self.torch_attention_backend = env_attention_backend

        env_load_in_4bit = os.getenv("OPENMED_LOAD_IN_4BIT")
        if env_load_in_4bit is not None:
            self.load_in_4bit = env_load_in_4bit.lower() not in {
                "0",
                "false",
                "no",
            }

        env_bnb_double_quant = os.getenv("OPENMED_BNB_4BIT_USE_DOUBLE_QUANT")
        if env_bnb_double_quant is not None:
            self.bnb_4bit_use_double_quant = env_bnb_double_quant.lower() not in {
                "0",
                "false",
                "no",
            }

        env_indic_matching = os.getenv("OPENMED_TRANSLITERATION_AWARE_NAME_MATCHING")
        if env_indic_matching is not None:
            self.transliteration_aware_name_matching = env_flag_enabled(
                env_indic_matching
            )

        env_indic_threshold = os.getenv("OPENMED_INDIC_NAME_SIMILARITY_THRESHOLD")
        if env_indic_threshold is not None:
            self.indic_name_similarity_threshold = float(env_indic_threshold)
            if not 0.5 <= self.indic_name_similarity_threshold <= 1.0:
                raise ValueError(
                    "OPENMED_INDIC_NAME_SIMILARITY_THRESHOLD must be between "
                    "0.5 and 1.0"
                )

        env_offline = os.getenv(OFFLINE_ENV_VAR)
        if env_offline is not None:
            self.local_only = self.local_only or env_flag_enabled(env_offline)

        configure_offline_mode(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OpenMedConfig":
        """Create config from a schema-validated dictionary."""

        _validate_config_mapping(_config_validation_view(config_dict))
        config = cls(**config_dict)
        config.validate()
        return config

    @classmethod
    def from_profile(cls, profile_name: str, **overrides: Any) -> "OpenMedConfig":
        """Create config from a named profile.

        Args:
            profile_name: Name of a built-in or custom profile.
            **overrides: Additional config values to override.

        Returns:
            OpenMedConfig instance with profile settings applied.

        Raises:
            ValueError: If the profile doesn't exist.
        """
        # First check built-in presets
        if profile_name in PROFILE_PRESETS:
            profile_data = dict(PROFILE_PRESETS[profile_name])
        else:
            # Try to load from profile file
            profile_path = PROFILES_DIR / f"{profile_name}.toml"
            if profile_path.exists():
                profile_data = _load_toml(profile_path)
            else:
                available = list(PROFILE_PRESETS.keys())
                # Add custom profiles
                if PROFILES_DIR.exists():
                    for p in PROFILES_DIR.glob("*.toml"):
                        available.append(p.stem)
                raise ValueError(
                    f"Unknown profile: {profile_name}. "
                    f"Available profiles: {', '.join(sorted(available))}"
                )

        _validate_config_mapping(profile_data, profile=True)
        _validate_config_mapping(overrides, profile=True)
        profile_data.update(overrides)
        profile_data["profile"] = profile_name
        return cls.from_dict(profile_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "default_org": self.default_org,
            "cache_dir": self.cache_dir,
            "device": self.device,
            "hf_token": self.hf_token,
            "log_level": self.log_level,
            "timeout": self.timeout,
            "use_medical_tokenizer": self.use_medical_tokenizer,
            "medical_tokenizer_exceptions": self.medical_tokenizer_exceptions,
            "chinese_segmentation_backend": self.chinese_segmentation_backend,
            "chinese_user_dict_path": self.chinese_user_dict_path,
            "chinese_pkuseg_domain": self.chinese_pkuseg_domain,
            "clinical_protect_enabled": self.clinical_protect_enabled,
            "clinical_protect_terms": self.clinical_protect_terms,
            "clinical_protect_use_builtin": self.clinical_protect_use_builtin,
            "backend": self.backend,
            "remote_inference_endpoint": self.remote_inference_endpoint,
            "remote_inference_protocol": self.remote_inference_protocol,
            "remote_inference_model_name": self.remote_inference_model_name,
            "remote_inference_model_version": self.remote_inference_model_version,
            "remote_inference_tokenizer": self.remote_inference_tokenizer,
            "remote_inference_timeout_seconds": self.remote_inference_timeout_seconds,
            "remote_inference_verify_tls": self.remote_inference_verify_tls,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "lazy_model_loading": self.lazy_model_loading,
            "pii_model": self.pii_model,
            "pii_model_revision": self.pii_model_revision,
            "onnx_variant": self.onnx_variant,
            "onnx_intra_op_num_threads": self.onnx_intra_op_num_threads,
            "torch_attention_backend": self.torch_attention_backend,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "local_only": self.local_only,
            "cjk_width_convention": self.cjk_width_convention,
            "chinese_target_script": self.chinese_target_script,
            "transliteration_aware_name_matching": (
                self.transliteration_aware_name_matching
            ),
            "indic_name_similarity_threshold": self.indic_name_similarity_threshold,
            "profile": self.profile,
        }

    def validate(self) -> None:
        """Validate this configuration against the bundled JSON Schema.

        Raises:
            ConfigValidationError: If one or more fields violate the schema.
        """

        _validate_config_mapping(self.to_dict())

    def with_profile(self, profile_name: str) -> "OpenMedConfig":
        """Return a new config with profile settings applied.

        Args:
            profile_name: Name of the profile to apply.

        Returns:
            New OpenMedConfig with profile settings merged.
        """
        # Start with current values
        current = self.to_dict()

        # Get profile settings
        if profile_name in PROFILE_PRESETS:
            profile_data = dict(PROFILE_PRESETS[profile_name])
        else:
            profile_path = PROFILES_DIR / f"{profile_name}.toml"
            if profile_path.exists():
                profile_data = _load_toml(profile_path)
            else:
                raise ValueError(f"Unknown profile: {profile_name}")

        # Merge profile into current (profile values override)
        current.update(profile_data)
        current["profile"] = profile_name
        return OpenMedConfig.from_dict(current)


# Global configuration instance
_config = OpenMedConfig()


def get_config() -> OpenMedConfig:
    """Get the global configuration instance."""
    return _config


def set_config(config: OpenMedConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def resolve_config_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the configuration file path, applying environment overrides."""
    if path:
        return Path(path).expanduser()

    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()

    return DEFAULT_CONFIG_PATH


def ensure_config_directory(path: Path) -> None:
    """Ensure that the configuration directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    # Quoted string (double or single)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # Fallback to raw string
    return value


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return f'"{value}"'


def _load_toml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            data[key] = _parse_value(value)
    return data


def _dump_toml(data: Dict[str, Any]) -> str:
    lines = [
        "# OpenMed configuration file",
        "# Generated automatically. Edit with care.",
        "",
    ]
    for key, value in data.items():
        lines.append(f"{key} = {_format_value(value)}")
    return "\n".join(lines) + "\n"


def load_config_from_file(path: Optional[Union[str, Path]] = None) -> OpenMedConfig:
    """Load configuration from a TOML file, merging with current defaults."""
    config_path = resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    file_data = _load_toml(config_path)
    _validate_config_mapping(file_data)
    merged = get_config().to_dict()

    merged.update(file_data)

    return OpenMedConfig.from_dict(merged)


def save_config_to_file(
    config: OpenMedConfig, path: Optional[Union[str, Path]] = None
) -> Path:
    """Persist configuration to a TOML file."""
    config.validate()
    config_path = resolve_config_path(path)
    ensure_config_directory(config_path)
    toml_content = _dump_toml(config.to_dict())
    config_path.write_text(toml_content, encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Profile Management Functions
# ---------------------------------------------------------------------------


def list_profiles() -> List[str]:
    """List all available profiles (built-in and custom).

    Returns:
        List of profile names.
    """
    profiles = list(PROFILE_PRESETS.keys())

    # Add custom profiles from profiles directory
    if PROFILES_DIR.exists():
        for profile_path in PROFILES_DIR.glob("*.toml"):
            profile_name = profile_path.stem
            if profile_name not in profiles:
                profiles.append(profile_name)

    return sorted(profiles)


def get_profile(profile_name: str) -> Dict[str, Any]:
    """Get the settings for a specific profile.

    Args:
        profile_name: Name of the profile.

    Returns:
        Dictionary of profile settings.

    Raises:
        ValueError: If the profile doesn't exist.
    """
    if profile_name in PROFILE_PRESETS:
        return dict(PROFILE_PRESETS[profile_name])

    profile_path = PROFILES_DIR / f"{profile_name}.toml"
    if profile_path.exists():
        profile_data = _load_toml(profile_path)
        _validate_config_mapping(profile_data, profile=True)
        return profile_data

    raise ValueError(f"Unknown profile: {profile_name}")


def save_profile(profile_name: str, settings: Dict[str, Any]) -> Path:
    """Save a custom profile to the profiles directory.

    Args:
        profile_name: Name for the profile.
        settings: Profile settings to save.

    Returns:
        Path to the saved profile file.
    """
    _validate_config_mapping(settings, profile=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{profile_name}.toml"

    lines = [
        f"# OpenMed profile: {profile_name}",
        "# Custom profile configuration",
        "",
    ]
    for key, value in settings.items():
        lines.append(f"{key} = {_format_value(value)}")

    profile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return profile_path


def delete_profile(profile_name: str) -> bool:
    """Delete a custom profile.

    Args:
        profile_name: Name of the profile to delete.

    Returns:
        True if deleted, False if not found.

    Raises:
        ValueError: If trying to delete a built-in profile.
    """
    if profile_name in PROFILE_PRESETS:
        raise ValueError(f"Cannot delete built-in profile: {profile_name}")

    profile_path = PROFILES_DIR / f"{profile_name}.toml"
    if profile_path.exists():
        profile_path.unlink()
        return True
    return False


def load_config_with_profile(
    profile_name: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> OpenMedConfig:
    """Load configuration with an optional profile applied.

    This function provides a convenient way to load configuration
    with profile settings merged in. The profile can be specified
    directly or via the OPENMED_PROFILE environment variable.

    Args:
        profile_name: Optional profile name to apply.
        config_path: Optional path to config file.

    Returns:
        OpenMedConfig instance.
    """
    # Start with base config
    try:
        config = load_config_from_file(config_path)
    except FileNotFoundError:
        config = get_config()

    # Determine profile (explicit > env > config file)
    effective_profile = profile_name or os.getenv(PROFILE_ENV_VAR) or config.profile

    if effective_profile:
        config = config.with_profile(effective_profile)

    return config
