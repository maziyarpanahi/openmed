"""Local-only PHI replacement primitives for the Health Universe agent."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
MODEL_REVISION = "364fd803fc7830dc655619c8d345508202e3868b66f7357123bb713828eefc9e"

OFFLINE_ENVIRONMENT = {
    "DO_NOT_TRACK": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "OPENMED_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


@dataclass(frozen=True)
class PhiReplacementSettings:
    """Runtime settings for local PHI replacement."""

    model_path: str = MODEL_ID
    model_cache_dir: str = str(Path.home() / ".cache" / "openmed")
    confidence_threshold: float = 0.5
    base_seed: int = 42
    language: str = "en"
    extraction_wait_seconds: int = 1800
    maximum_documents: int = 2000
    minimum_alphanumeric_characters: int = 10

    def __post_init__(self) -> None:
        """Reject unsafe or nonsensical settings without echoing input values."""

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        if not 30 <= self.extraction_wait_seconds <= 14400:
            raise ValueError("extraction_wait_seconds must be between 30 and 14400")
        if not 1 <= self.maximum_documents <= 10000:
            raise ValueError("maximum_documents must be between 1 and 10000")
        if self.minimum_alphanumeric_characters < 1:
            raise ValueError("minimum_alphanumeric_characters must be positive")

    @classmethod
    def from_environment(cls) -> PhiReplacementSettings:
        """Load settings from ``PHI_AGENT_*`` environment variables."""

        return cls(
            model_path=os.getenv("PHI_AGENT_MODEL_PATH", MODEL_ID),
            model_cache_dir=os.getenv(
                "PHI_AGENT_MODEL_CACHE_DIR",
                str(Path.home() / ".cache" / "openmed"),
            ),
            confidence_threshold=float(
                os.getenv("PHI_AGENT_CONFIDENCE_THRESHOLD", "0.5")
            ),
            base_seed=int(os.getenv("PHI_AGENT_BASE_SEED", "42")),
            language=os.getenv("PHI_AGENT_LANGUAGE", "en"),
            extraction_wait_seconds=int(
                os.getenv("PHI_AGENT_EXTRACTION_WAIT_SECONDS", "1800")
            ),
            maximum_documents=int(os.getenv("PHI_AGENT_MAXIMUM_DOCUMENTS", "2000")),
            minimum_alphanumeric_characters=int(
                os.getenv("PHI_AGENT_MINIMUM_ALPHANUMERIC_CHARACTERS", "10")
            ),
        )


@dataclass(frozen=True)
class ReplacementResult:
    """Replacement Markdown plus evidence without identifier values."""

    markdown: str
    entity_count: int
    action_counts: dict[str, int]
    entity_label_counts: dict[str, int]
    replacement_collision_count: int
    replacement_collision_label_counts: dict[str, int]
    ip_surrogates_checked: int
    invalid_ip_surrogates: int


def force_offline_environment() -> None:
    """Disable model-network access and common telemetry paths."""

    for environment_name, environment_value in OFFLINE_ENVIRONMENT.items():
        os.environ[environment_name] = environment_value


def normalized_surface(value: str | None) -> str:
    """Normalize an in-memory surface for replacement collision checks."""

    return re.sub(r"[^A-Za-z0-9]", "", value or "").casefold()


def invalid_ip_surrogate(label: str, value: str | None) -> bool | None:
    """Return IP validity without returning or retaining the IP value."""

    normalized_label = label.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized_label not in {"ip", "ip_address", "ipv4", "ipv6"}:
        return None
    try:
        parsed = ipaddress.ip_address((value or "").strip())
    except ValueError:
        return True
    if normalized_label == "ipv4":
        return parsed.version != 4
    if normalized_label == "ipv6":
        return parsed.version != 6
    return False


def seed_for_scope(base_seed: int, scope: str) -> int:
    """Derive a stable per-thread seed without persisting the thread ID."""

    material = f"{base_seed}:{scope}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")


class OpenMedReplacementEngine:
    """Lazily load and reuse one local OpenMed replacement model."""

    def __init__(self, settings: PhiReplacementSettings):
        self.settings = settings
        self._deidentify: Any = None
        self._config: Any = None
        self._loader: Any = None
        force_offline_environment()

    def _ensure_runtime(self) -> None:
        if self._loader is not None:
            return
        from openmed import deidentify
        from openmed.core.config import OpenMedConfig
        from openmed.core.models import ModelLoader

        self._deidentify = deidentify
        self._config = OpenMedConfig(
            cache_dir=self.settings.model_cache_dir,
            local_only=True,
            backend="hf",
            log_level="WARNING",
        )
        self._loader = ModelLoader(config=self._config)

    def replace(self, markdown: str, *, seed_scope: str) -> ReplacementResult:
        """Replace detected identifiers without saving a re-identification map."""

        self._ensure_runtime()
        result = self._deidentify(
            markdown,
            method="replace",
            model_name=self.settings.model_path,
            confidence_threshold=self.settings.confidence_threshold,
            config=self._config,
            loader=self._loader,
            lang=self.settings.language,
            policy=None,
            use_safety_sweep=True,
            keep_mapping=False,
            consistent=True,
            seed=seed_for_scope(self.settings.base_seed, seed_scope),
            cache_results=False,
        )

        actions: Counter[str] = Counter()
        labels: Counter[str] = Counter()
        collision_labels: Counter[str] = Counter()
        ip_checked = 0
        invalid_ips = 0
        for entity in result.pii_entities:
            actions[entity.action or "unknown"] += 1
            labels[entity.label] += 1
            source = normalized_surface(entity.original_text)
            replacement = normalized_surface(entity.redacted_text)
            if source and source == replacement:
                collision_labels[entity.label] += 1
            invalid_ip = invalid_ip_surrogate(entity.label, entity.redacted_text)
            if invalid_ip is not None:
                ip_checked += 1
                invalid_ips += int(invalid_ip)

        return ReplacementResult(
            markdown=result.deidentified_text,
            entity_count=len(result.pii_entities),
            action_counts=dict(actions),
            entity_label_counts=dict(labels),
            replacement_collision_count=sum(collision_labels.values()),
            replacement_collision_label_counts=dict(collision_labels),
            ip_surrogates_checked=ip_checked,
            invalid_ip_surrogates=invalid_ips,
        )
