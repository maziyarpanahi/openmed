"""Configuration management for OpenMed."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Union
import os

# Environment variable used to override the config file location
CONFIG_ENV_VAR = "OPENMED_CONFIG"

_xdg_config = os.getenv("XDG_CONFIG_HOME")
if _xdg_config:
    _default_config_root = Path(_xdg_config)
else:
    _default_config_root = Path.home() / ".config"

DEFAULT_CONFIG_DIR = _default_config_root / "openmed"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"


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

    def __post_init__(self):
        """Post-initialization to set default values."""
        if self.cache_dir is None:
            self.cache_dir = os.path.expanduser("~/.cache/openmed")

        if self.hf_token is None:
            self.hf_token = os.getenv("HF_TOKEN")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OpenMedConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "default_org": self.default_org,
            "cache_dir": self.cache_dir,
            "device": self.device,
            "hf_token": self.hf_token,
            "log_level": self.log_level,
            "timeout": self.timeout,
        }


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
