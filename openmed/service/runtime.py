"""Runtime helpers for the OpenMed REST service."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from openmed.core.config import OpenMedConfig, PROFILE_ENV_VAR
from openmed.core.models import ModelLoader
from openmed.utils.validation import validate_model_name

SERVICE_PRELOAD_ENV_VAR = "OPENMED_SERVICE_PRELOAD_MODELS"


def parse_preload_models(raw_value: Optional[str]) -> Tuple[str, ...]:
    """Parse and validate the preload-model environment variable."""
    if raw_value is None:
        return ()

    models = []
    seen = set()
    for item in raw_value.split(","):
        model_name = item.strip()
        if not model_name:
            continue

        validated = validate_model_name(model_name)
        if validated in seen:
            continue

        models.append(validated)
        seen.add(validated)

    return tuple(models)


@dataclass
class ServiceRuntime:
    """Shared runtime state for the REST service."""

    profile: str
    config: OpenMedConfig
    preload_models: Tuple[str, ...] = ()
    _loader_factory: Optional[Callable[[OpenMedConfig], ModelLoader]] = None
    _loader: Optional[ModelLoader] = None
    _loader_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls) -> "ServiceRuntime":
        """Create a runtime using the current process environment."""
        profile = os.getenv(PROFILE_ENV_VAR, "prod")
        config = OpenMedConfig.from_profile(profile)
        preload_models = parse_preload_models(os.getenv(SERVICE_PRELOAD_ENV_VAR))
        return cls(
            profile=profile,
            config=config,
            preload_models=preload_models,
            _loader_factory=ModelLoader,
        )

    def get_loader(self) -> ModelLoader:
        """Return the shared loader, creating it on first use."""
        if self._loader is None:
            with self._loader_lock:
                if self._loader is None:
                    factory = self._loader_factory or ModelLoader
                    self._loader = factory(self.config)
        return self._loader

    def preload(self) -> None:
        """Warm configured model pipelines during service startup."""
        if not self.preload_models:
            return

        loader = self.get_loader()
        for model_name in self.preload_models:
            loader.create_pipeline(
                model_name,
                task="token-classification",
                aggregation_strategy="simple",
                use_fast_tokenizer=True,
            )
