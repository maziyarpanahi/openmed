"""Configuration management for OpenMed."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os


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
