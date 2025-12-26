"""Configuration management for OpenMed."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import os

# Environment variable used to override the config file location
CONFIG_ENV_VAR = "OPENMED_CONFIG"

# Environment variable for active profile
PROFILE_ENV_VAR = "OPENMED_PROFILE"

_xdg_config = os.getenv("XDG_CONFIG_HOME")
if _xdg_config:
    _default_config_root = Path(_xdg_config)
else:
    _default_config_root = Path.home() / ".config"

DEFAULT_CONFIG_DIR = _default_config_root / "openmed"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"
PROFILES_DIR = DEFAULT_CONFIG_DIR / "profiles"

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

    # Active profile name (if any)
    profile: Optional[str] = None

    def __post_init__(self):
        """Post-initialization to set default values."""
        if self.cache_dir is None:
            self.cache_dir = os.path.expanduser("~/.cache/openmed")

        if self.hf_token is None:
            self.hf_token = os.getenv("HF_TOKEN")

        env_use_med_tok = os.getenv("OPENMED_USE_MEDICAL_TOKENIZER")
        if env_use_med_tok is not None:
            self.use_medical_tokenizer = env_use_med_tok.lower() not in {"0", "false", "no"}

        env_exceptions = os.getenv("OPENMED_MEDICAL_TOKENIZER_EXCEPTIONS")
        if env_exceptions:
            self.medical_tokenizer_exceptions = [item.strip() for item in env_exceptions.split(",") if item.strip()]

        # Check for profile environment variable
        env_profile = os.getenv(PROFILE_ENV_VAR)
        if env_profile and self.profile is None:
            self.profile = env_profile

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OpenMedConfig":
        """Create config from dictionary."""
        # Filter out unknown keys
        valid_keys = {
            "default_org", "cache_dir", "device", "hf_token",
            "log_level", "timeout", "use_medical_tokenizer",
            "medical_tokenizer_exceptions", "profile"
        }
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_profile(cls, profile_name: str, **overrides: Any) -> "OpenMedConfig":
        """Create config from a named profile.

        Args:
            profile_name: Name of the profile (dev, prod, test, fast, or custom).
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

        profile_data["profile"] = profile_name
        profile_data.update(overrides)
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
            "profile": self.profile,
        }

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

    # Fallback to raw string
    return value


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
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
    merged = get_config().to_dict()

    for key, value in file_data.items():
        if key in merged:
            merged[key] = value

    return OpenMedConfig.from_dict(merged)


def save_config_to_file(
    config: OpenMedConfig, path: Optional[Union[str, Path]] = None
) -> Path:
    """Persist configuration to a TOML file."""
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
        return _load_toml(profile_path)

    raise ValueError(f"Unknown profile: {profile_name}")


def save_profile(profile_name: str, settings: Dict[str, Any]) -> Path:
    """Save a custom profile to the profiles directory.

    Args:
        profile_name: Name for the profile.
        settings: Profile settings to save.

    Returns:
        Path to the saved profile file.
    """
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
