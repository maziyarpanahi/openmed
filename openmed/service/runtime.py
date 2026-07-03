"""Runtime helpers for the OpenMed REST service."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from openmed.core.config import PROFILE_ENV_VAR, OpenMedConfig
from openmed.core.models import ModelLoader
from openmed.utils.validation import validate_batch_size, validate_model_name

from .keep_alive import parse_keep_alive
from .resilience import ResilienceManager, ServiceResilienceConfig
from .warm_pool import WarmPool, parse_max_resident_models

SERVICE_PRELOAD_ENV_VAR = "OPENMED_SERVICE_PRELOAD_MODELS"
SERVICE_KEEP_ALIVE_ENV_VAR = "OPENMED_SERVICE_KEEP_ALIVE"
SERVICE_MAX_RESIDENT_ENV_VAR = "OPENMED_SERVICE_MAX_RESIDENT_MODELS"
SERVICE_BATCHING_ENABLED_ENV_VAR = "OPENMED_SERVICE_BATCHING_ENABLED"
SERVICE_BATCH_MAX_SIZE_ENV_VAR = "OPENMED_SERVICE_BATCH_MAX_SIZE"
SERVICE_BATCH_MAX_WAIT_MS_ENV_VAR = "OPENMED_SERVICE_BATCH_MAX_WAIT_MS"
SERVICE_COALESCING_ENABLED_ENV_VAR = "OPENMED_SERVICE_COALESCING_ENABLED"
SERVICE_SHUTDOWN_DRAIN_ENV_VAR = "OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS"
SERVICE_RATE_LIMIT_RPS_ENV_VAR = "OPENMED_SERVICE_RATE_LIMIT_RPS"
SERVICE_RATE_LIMIT_BURST_ENV_VAR = "OPENMED_SERVICE_RATE_LIMIT_BURST"
SERVICE_MAX_CONCURRENCY_ENV_VAR = "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY"
SERVICE_THROTTLE_KEY_ENV_VAR = "OPENMED_SERVICE_THROTTLE_KEY"
SERVICE_CONCURRENCY_WAIT_ENV_VAR = "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS"
SERVICE_RESILIENCE_ENABLED_ENV_VAR = "OPENMED_SERVICE_RESILIENCE_ENABLED"
SERVICE_RETRY_MAX_ATTEMPTS_ENV_VAR = "OPENMED_SERVICE_RETRY_MAX_ATTEMPTS"
SERVICE_RETRY_BACKOFF_INITIAL_ENV_VAR = "OPENMED_SERVICE_RETRY_BACKOFF_INITIAL_SECONDS"
SERVICE_RETRY_BACKOFF_MULTIPLIER_ENV_VAR = "OPENMED_SERVICE_RETRY_BACKOFF_MULTIPLIER"
SERVICE_RETRY_BACKOFF_MAX_ENV_VAR = "OPENMED_SERVICE_RETRY_BACKOFF_MAX_SECONDS"
SERVICE_RETRY_BACKOFF_JITTER_ENV_VAR = "OPENMED_SERVICE_RETRY_BACKOFF_JITTER_SECONDS"
SERVICE_BREAKER_FAILURE_THRESHOLD_ENV_VAR = (
    "OPENMED_SERVICE_CIRCUIT_BREAKER_FAILURE_THRESHOLD"
)
SERVICE_BREAKER_RECOVERY_TIMEOUT_ENV_VAR = (
    "OPENMED_SERVICE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS"
)
DEFAULT_SERVICE_BATCH_MAX_SIZE = 8
DEFAULT_SERVICE_BATCH_MAX_WAIT_MS = 5.0
DEFAULT_SERVICE_SHUTDOWN_DRAIN_SECONDS = 30.0
DEFAULT_SERVICE_CONCURRENCY_WAIT_SECONDS = 0.05
DEFAULT_SERVICE_RETRY_MAX_ATTEMPTS = 3
DEFAULT_SERVICE_RETRY_BACKOFF_INITIAL_SECONDS = 0.05
DEFAULT_SERVICE_RETRY_BACKOFF_MULTIPLIER = 2.0
DEFAULT_SERVICE_RETRY_BACKOFF_MAX_SECONDS = 1.0
DEFAULT_SERVICE_RETRY_BACKOFF_JITTER_SECONDS = 0.01
DEFAULT_SERVICE_BREAKER_FAILURE_THRESHOLD = 3
DEFAULT_SERVICE_BREAKER_RECOVERY_TIMEOUT_SECONDS = 30.0

_BATCHING_ENABLED_VALUES = {"1", "true", "yes", "on", "enabled"}
_BATCHING_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}
_THROTTLE_KEY_ALIASES = {
    "global": "global",
    "process": "global",
    "xff": "x-forwarded-for",
    "x-forwarded-for": "x-forwarded-for",
    "peer": "peer",
    "client": "peer",
    "remote": "peer",
}


@dataclass(frozen=True)
class ServiceBatchingConfig:
    """Dynamic batching settings for REST model-backed endpoints."""

    enabled: bool = False
    max_batch_size: int = DEFAULT_SERVICE_BATCH_MAX_SIZE
    max_wait_ms: float = DEFAULT_SERVICE_BATCH_MAX_WAIT_MS


@dataclass(frozen=True)
class ServiceCoalescingConfig:
    """Request coalescing settings for REST model-backed endpoints."""

    enabled: bool = False


@dataclass(frozen=True)
class ServiceThrottleConfig:
    """Rate and concurrency limits for REST model-backed endpoints."""

    rate_limit_rps: float = 0.0
    rate_limit_burst: int = 0
    max_concurrency: int = 0
    concurrency_wait_seconds: float = DEFAULT_SERVICE_CONCURRENCY_WAIT_SECONDS
    key_by: str = "global"

    @property
    def rate_limit_enabled(self) -> bool:
        """Return whether token-bucket rate limiting is active."""
        return self.rate_limit_rps > 0 and self.rate_limit_burst > 0

    @property
    def concurrency_enabled(self) -> bool:
        """Return whether in-flight concurrency limiting is active."""
        return self.max_concurrency > 0

    @property
    def enabled(self) -> bool:
        """Return whether any throttling gate is active."""
        return self.rate_limit_enabled or self.concurrency_enabled


def _parse_non_negative_float(
    raw_value: Optional[str],
    *,
    env_var: str,
    default: float,
) -> float:
    if raw_value is None or not raw_value.strip():
        return default

    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be a non-negative number") from exc
    if parsed < 0:
        raise ValueError(f"{env_var} must be greater than or equal to 0")
    return parsed


def _parse_non_negative_int(
    raw_value: Optional[str],
    *,
    env_var: str,
    default: int,
) -> int:
    if raw_value is None or not raw_value.strip():
        return default

    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{env_var} must be greater than or equal to 0")
    return parsed


def parse_service_batching_enabled(raw_value: Optional[str]) -> bool:
    """Parse the dynamic-batching feature flag."""
    if raw_value is None:
        return False

    normalized = raw_value.strip().lower()
    if not normalized:
        return False
    if normalized in _BATCHING_ENABLED_VALUES:
        return True
    if normalized in _BATCHING_DISABLED_VALUES:
        return False
    raise ValueError(
        f"{SERVICE_BATCHING_ENABLED_ENV_VAR} must be a boolean value like "
        "'true' or 'false'"
    )


def parse_service_coalescing_enabled(raw_value: Optional[str]) -> bool:
    """Parse the request-coalescing feature flag."""
    if raw_value is None:
        return False

    normalized = raw_value.strip().lower()
    if not normalized:
        return False
    if normalized in _BATCHING_ENABLED_VALUES:
        return True
    if normalized in _BATCHING_DISABLED_VALUES:
        return False
    raise ValueError(
        f"{SERVICE_COALESCING_ENABLED_ENV_VAR} must be a boolean value like "
        "'true' or 'false'"
    )


def parse_service_resilience_enabled(raw_value: Optional[str]) -> bool:
    """Parse the resilience feature flag."""
    if raw_value is None:
        return True

    normalized = raw_value.strip().lower()
    if not normalized:
        return True
    if normalized in _BATCHING_ENABLED_VALUES:
        return True
    if normalized in _BATCHING_DISABLED_VALUES:
        return False
    raise ValueError(
        f"{SERVICE_RESILIENCE_ENABLED_ENV_VAR} must be a boolean value like "
        "'true' or 'false'"
    )


def parse_service_batch_max_size(raw_value: Optional[str]) -> int:
    """Parse the configured dynamic-batching maximum size."""
    if raw_value is None or not raw_value.strip():
        return DEFAULT_SERVICE_BATCH_MAX_SIZE

    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{SERVICE_BATCH_MAX_SIZE_ENV_VAR} must be a positive integer"
        ) from exc
    return validate_batch_size(parsed)


def parse_service_batch_max_wait_ms(raw_value: Optional[str]) -> float:
    """Parse the configured dynamic-batching wait window in milliseconds."""
    if raw_value is None or not raw_value.strip():
        return DEFAULT_SERVICE_BATCH_MAX_WAIT_MS

    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{SERVICE_BATCH_MAX_WAIT_MS_ENV_VAR} must be a non-negative number"
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"{SERVICE_BATCH_MAX_WAIT_MS_ENV_VAR} must be greater than or equal to 0"
        )
    return parsed


def parse_shutdown_drain_seconds(raw_value: Optional[str]) -> float:
    """Parse the configured graceful-shutdown drain window in seconds."""
    if raw_value is None or not raw_value.strip():
        return DEFAULT_SERVICE_SHUTDOWN_DRAIN_SECONDS

    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{SERVICE_SHUTDOWN_DRAIN_ENV_VAR} must be a non-negative number"
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"{SERVICE_SHUTDOWN_DRAIN_ENV_VAR} must be greater than or equal to 0"
        )
    return parsed


def parse_service_rate_limit_rps(raw_value: Optional[str]) -> float:
    """Parse the configured token-bucket refill rate in requests per second."""
    return _parse_non_negative_float(
        raw_value,
        env_var=SERVICE_RATE_LIMIT_RPS_ENV_VAR,
        default=0.0,
    )


def parse_service_rate_limit_burst(raw_value: Optional[str]) -> int:
    """Parse the configured token-bucket burst capacity."""
    return _parse_non_negative_int(
        raw_value,
        env_var=SERVICE_RATE_LIMIT_BURST_ENV_VAR,
        default=0,
    )


def parse_service_max_concurrency(raw_value: Optional[str]) -> int:
    """Parse the configured maximum number of in-flight model requests."""
    return _parse_non_negative_int(
        raw_value,
        env_var=SERVICE_MAX_CONCURRENCY_ENV_VAR,
        default=0,
    )


def parse_service_concurrency_wait_seconds(raw_value: Optional[str]) -> float:
    """Parse the bounded wait before rejecting saturated requests."""
    return _parse_non_negative_float(
        raw_value,
        env_var=SERVICE_CONCURRENCY_WAIT_ENV_VAR,
        default=DEFAULT_SERVICE_CONCURRENCY_WAIT_SECONDS,
    )


def parse_service_retry_max_attempts(raw_value: Optional[str]) -> int:
    """Parse the configured retry attempt cap."""
    if raw_value is None or not raw_value.strip():
        return DEFAULT_SERVICE_RETRY_MAX_ATTEMPTS

    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{SERVICE_RETRY_MAX_ATTEMPTS_ENV_VAR} must be a positive integer"
        ) from exc
    if parsed < 1:
        raise ValueError(
            f"{SERVICE_RETRY_MAX_ATTEMPTS_ENV_VAR} must be greater than or equal to 1"
        )
    return parsed


def parse_service_retry_backoff_multiplier(raw_value: Optional[str]) -> float:
    """Parse the exponential backoff multiplier."""
    if raw_value is None or not raw_value.strip():
        return DEFAULT_SERVICE_RETRY_BACKOFF_MULTIPLIER

    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{SERVICE_RETRY_BACKOFF_MULTIPLIER_ENV_VAR} must be a number"
        ) from exc
    if parsed < 1:
        raise ValueError(
            f"{SERVICE_RETRY_BACKOFF_MULTIPLIER_ENV_VAR} must be greater than "
            "or equal to 1"
        )
    return parsed


def parse_service_breaker_failure_threshold(raw_value: Optional[str]) -> int:
    """Parse the circuit-breaker consecutive failure threshold."""
    if raw_value is None or not raw_value.strip():
        return DEFAULT_SERVICE_BREAKER_FAILURE_THRESHOLD

    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{SERVICE_BREAKER_FAILURE_THRESHOLD_ENV_VAR} must be a positive integer"
        ) from exc
    if parsed < 1:
        raise ValueError(
            f"{SERVICE_BREAKER_FAILURE_THRESHOLD_ENV_VAR} must be greater than "
            "or equal to 1"
        )
    return parsed


def parse_service_throttle_key(raw_value: Optional[str]) -> str:
    """Parse how throttle buckets are keyed."""
    if raw_value is None or not raw_value.strip():
        return "global"

    normalized = raw_value.strip().lower()
    try:
        return _THROTTLE_KEY_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"{SERVICE_THROTTLE_KEY_ENV_VAR} must be one of: global, "
            "x-forwarded-for, or peer"
        ) from exc


def parse_service_throttle_config() -> ServiceThrottleConfig:
    """Read throttling settings from the current process environment."""
    return ServiceThrottleConfig(
        rate_limit_rps=parse_service_rate_limit_rps(
            os.getenv(SERVICE_RATE_LIMIT_RPS_ENV_VAR)
        ),
        rate_limit_burst=parse_service_rate_limit_burst(
            os.getenv(SERVICE_RATE_LIMIT_BURST_ENV_VAR)
        ),
        max_concurrency=parse_service_max_concurrency(
            os.getenv(SERVICE_MAX_CONCURRENCY_ENV_VAR)
        ),
        concurrency_wait_seconds=parse_service_concurrency_wait_seconds(
            os.getenv(SERVICE_CONCURRENCY_WAIT_ENV_VAR)
        ),
        key_by=parse_service_throttle_key(os.getenv(SERVICE_THROTTLE_KEY_ENV_VAR)),
    )


def parse_service_batching_config() -> ServiceBatchingConfig:
    """Read dynamic-batching settings from the current process environment."""
    return ServiceBatchingConfig(
        enabled=parse_service_batching_enabled(
            os.getenv(SERVICE_BATCHING_ENABLED_ENV_VAR)
        ),
        max_batch_size=parse_service_batch_max_size(
            os.getenv(SERVICE_BATCH_MAX_SIZE_ENV_VAR)
        ),
        max_wait_ms=parse_service_batch_max_wait_ms(
            os.getenv(SERVICE_BATCH_MAX_WAIT_MS_ENV_VAR)
        ),
    )


def parse_service_coalescing_config() -> ServiceCoalescingConfig:
    """Read request-coalescing settings from the current process environment."""
    return ServiceCoalescingConfig(
        enabled=parse_service_coalescing_enabled(
            os.getenv(SERVICE_COALESCING_ENABLED_ENV_VAR)
        )
    )


def parse_service_resilience_config() -> ServiceResilienceConfig:
    """Read retry and circuit-breaker settings from the environment."""
    return ServiceResilienceConfig(
        enabled=parse_service_resilience_enabled(
            os.getenv(SERVICE_RESILIENCE_ENABLED_ENV_VAR)
        ),
        max_attempts=parse_service_retry_max_attempts(
            os.getenv(SERVICE_RETRY_MAX_ATTEMPTS_ENV_VAR)
        ),
        backoff_initial_seconds=_parse_non_negative_float(
            os.getenv(SERVICE_RETRY_BACKOFF_INITIAL_ENV_VAR),
            env_var=SERVICE_RETRY_BACKOFF_INITIAL_ENV_VAR,
            default=DEFAULT_SERVICE_RETRY_BACKOFF_INITIAL_SECONDS,
        ),
        backoff_multiplier=parse_service_retry_backoff_multiplier(
            os.getenv(SERVICE_RETRY_BACKOFF_MULTIPLIER_ENV_VAR)
        ),
        backoff_max_seconds=_parse_non_negative_float(
            os.getenv(SERVICE_RETRY_BACKOFF_MAX_ENV_VAR),
            env_var=SERVICE_RETRY_BACKOFF_MAX_ENV_VAR,
            default=DEFAULT_SERVICE_RETRY_BACKOFF_MAX_SECONDS,
        ),
        backoff_jitter_seconds=_parse_non_negative_float(
            os.getenv(SERVICE_RETRY_BACKOFF_JITTER_ENV_VAR),
            env_var=SERVICE_RETRY_BACKOFF_JITTER_ENV_VAR,
            default=DEFAULT_SERVICE_RETRY_BACKOFF_JITTER_SECONDS,
        ),
        failure_threshold=parse_service_breaker_failure_threshold(
            os.getenv(SERVICE_BREAKER_FAILURE_THRESHOLD_ENV_VAR)
        ),
        recovery_timeout_seconds=_parse_non_negative_float(
            os.getenv(SERVICE_BREAKER_RECOVERY_TIMEOUT_ENV_VAR),
            env_var=SERVICE_BREAKER_RECOVERY_TIMEOUT_ENV_VAR,
            default=DEFAULT_SERVICE_BREAKER_RECOVERY_TIMEOUT_SECONDS,
        ),
    )


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
    max_resident_models: Optional[int] = None
    default_keep_alive_seconds: Optional[float] = None
    shutdown_drain_seconds: float = DEFAULT_SERVICE_SHUTDOWN_DRAIN_SECONDS
    batching: ServiceBatchingConfig = field(default_factory=ServiceBatchingConfig)
    coalescing: ServiceCoalescingConfig = field(default_factory=ServiceCoalescingConfig)
    throttle: ServiceThrottleConfig = field(default_factory=ServiceThrottleConfig)
    resilience_config: ServiceResilienceConfig = field(
        default_factory=ServiceResilienceConfig
    )
    _loader_factory: Optional[Callable[[OpenMedConfig], ModelLoader]] = None
    _loader: Optional[ModelLoader] = None
    _warm_pool: Optional[WarmPool] = None
    _resilience: Optional[ResilienceManager] = None
    _loader_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    metrics: Optional[Any] = None

    @classmethod
    def from_env(cls, *, metrics: Optional[Any] = None) -> "ServiceRuntime":
        """Create a runtime using the current process environment."""
        profile = os.getenv(PROFILE_ENV_VAR, "prod")
        config = OpenMedConfig.from_profile(profile)
        preload_models = parse_preload_models(os.getenv(SERVICE_PRELOAD_ENV_VAR))
        max_resident_models = parse_max_resident_models(
            os.getenv(SERVICE_MAX_RESIDENT_ENV_VAR)
        )
        keep_alive = parse_keep_alive(os.getenv(SERVICE_KEEP_ALIVE_ENV_VAR))
        batching = parse_service_batching_config()
        coalescing = parse_service_coalescing_config()
        throttle = parse_service_throttle_config()
        resilience_config = parse_service_resilience_config()
        return cls(
            profile=profile,
            config=config,
            preload_models=preload_models,
            max_resident_models=max_resident_models,
            default_keep_alive_seconds=keep_alive,
            shutdown_drain_seconds=parse_shutdown_drain_seconds(
                os.getenv(SERVICE_SHUTDOWN_DRAIN_ENV_VAR)
            ),
            batching=batching,
            coalescing=coalescing,
            throttle=throttle,
            resilience_config=resilience_config,
            _loader_factory=ModelLoader,
            metrics=metrics,
        )

    def get_model_loader(self) -> ModelLoader:
        """Return the wrapped model loader, creating it on first use."""
        if self._loader is None:
            with self._loader_lock:
                if self._loader is None:
                    factory = self._loader_factory or ModelLoader
                    self._loader = factory(self.config)
        return self._loader

    def get_loader(self) -> WarmPool:
        """Return the shared warm-pool loader proxy."""
        if self._warm_pool is None:
            with self._loader_lock:
                if self._warm_pool is None:
                    self._warm_pool = WarmPool(
                        self.get_model_loader,
                        warm_models=self.preload_models,
                        max_resident_models=self.max_resident_models,
                        default_keep_alive_seconds=self.default_keep_alive_seconds,
                        metrics=self.metrics,
                    )
        return self._warm_pool

    def get_resilience(self) -> ResilienceManager:
        """Return the shared retry and circuit-breaker manager."""
        if self._resilience is None:
            with self._loader_lock:
                if self._resilience is None:
                    self._resilience = ResilienceManager(self.resilience_config)
        return self._resilience

    def preload(self) -> None:
        """Warm configured model pipelines during service startup."""
        if not self.preload_models:
            return

        self.get_loader().preload()

    def run_model_request(
        self,
        model_name: str,
        keep_alive: Any,
        operation: Callable[[], Any],
    ) -> Any:
        """Run one model-backed operation with retry and breaker protection."""
        model_key = self._resolve_model_name(model_name)
        return self.get_resilience().execute(
            model_key,
            lambda: self._run_model_request_once(model_key, keep_alive, operation),
        )

    def _run_model_request_once(
        self,
        model_key: str,
        keep_alive: Any,
        operation: Callable[[], Any],
    ) -> Any:
        """Run one model-backed attempt and update idle-unload bookkeeping."""
        pool = self.get_loader()
        active_model_key = pool.begin_request(model_key)
        try:
            return operation()
        finally:
            pool.finish_request(active_model_key, keep_alive)

    def begin_model_request(self, model_name: str) -> str:
        """Mark a resolved model as active and cancel pending idle unload."""
        model_key = self._resolve_model_name(model_name)
        resilience = self.get_resilience()
        resilience.check_available(model_key)
        try:
            return self.get_loader().begin_request(model_key)
        except Exception as exc:
            resilience.record_error(model_key, exc)
            raise

    def finish_model_request(
        self,
        model_key: str,
        keep_alive: Any,
        error: Optional[BaseException] = None,
    ) -> None:
        """Mark a model request as complete and schedule idle unloading."""
        finish_error: Optional[BaseException] = None
        try:
            self.get_loader().finish_request(model_key, keep_alive)
        except Exception as exc:
            finish_error = exc
        if error is None and finish_error is None:
            self.get_resilience().record_success(model_key)
        elif finish_error is not None:
            self.get_resilience().record_error(model_key, finish_error)
        else:
            self.get_resilience().record_error(model_key, error)
        if finish_error is not None:
            raise finish_error

    def circuit_breaker_state_counts(self) -> dict[str, int]:
        """Return aggregate circuit-breaker states without model/backend labels."""
        return self.get_resilience().state_counts()

    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload one inactive model from the shared loader cache."""
        return self.get_loader().unload_model(model_name)

    def unload_all_models(self) -> Dict[str, Any]:
        """Unload every inactive model from the shared loader cache."""
        return self.get_loader().unload_all_models()

    def loaded_models(self) -> Dict[str, Any]:
        """Return cache and keep-alive status for the service runtime."""
        if self._warm_pool is None and self._loader is None:
            return {
                "default_keep_alive_seconds": self.default_keep_alive_seconds,
                "max_resident_models": self.max_resident_models,
                "warm_models": list(self.preload_models),
                "models": {},
            }
        return self.get_loader().loaded_models()

    def record_speculative_decode(self, metrics: Any) -> None:
        """Forward aggregate speculative decode metrics to the metrics registry."""
        if self.metrics is None:
            return
        recorder = getattr(self.metrics, "record_speculative_decode", None)
        if callable(recorder):
            recorder(metrics)

    def _resolve_keep_alive_seconds(self, keep_alive: Any) -> Optional[float]:
        if keep_alive is None:
            return self.default_keep_alive_seconds
        return parse_keep_alive(keep_alive)

    def _resolve_model_name(self, model_name: str) -> str:
        validated = validate_model_name(model_name)
        loader = self.get_model_loader()
        resolver = getattr(loader, "resolve_model_name", None)
        if callable(resolver):
            return resolver(validated)
        return validated
