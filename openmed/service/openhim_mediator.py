"""OpenHIM mediator registration, heartbeat, and response helpers.

The integration is deliberately opt-in.  Importing this module or creating the
REST application does not contact OpenHIM; network activity only begins when
``OPENMED_OPENHIM_MEDIATOR_ENABLED`` is true and the service lifespan starts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import quote, urlsplit

import httpx

from openmed.__about__ import __version__

OPENHIM_MEDIA_TYPE = "application/json+openhim"
OPENHIM_MEDIATOR_PATH = "/openhim/deidentify"
OPENHIM_HEARTBEAT_PATH = "/openhim/heartbeat"
DEFAULT_MEDIATOR_URN = "urn:openhim-mediator:openmed-deidentification"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_RESPONSE_HEADER_ALLOWLIST = frozenset(
    {
        "cache-control",
        "content-encoding",
        "content-language",
        "content-type",
        "x-request-id",
    }
)

Deidentifier = Callable[..., Any]


class OpenHIMMediatorConfigurationError(ValueError):
    """Raised when opt-in OpenHIM mediator settings are invalid."""


class OpenHIMMediatorProtocolError(RuntimeError):
    """Raised when OpenHIM core rejects registration or a heartbeat."""


def _parse_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise OpenHIMMediatorConfigurationError(
        f"{name} must be one of: 1, 0, true, false, yes, no, on, off"
    )


def _parse_positive_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise OpenHIMMediatorConfigurationError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise OpenHIMMediatorConfigurationError(f"{name} must be greater than zero")
    return parsed


def _parse_choice(name: str, default: str, choices: frozenset[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        raise OpenHIMMediatorConfigurationError(
            f"{name} must be one of: {', '.join(sorted(choices))}"
        )
    return value


def default_mediator_registration(
    *,
    host: str = "openmed-mediator",
    port: int = 8080,
) -> dict[str, Any]:
    """Return the built-in OpenMed registration payload.

    Args:
        host: Hostname OpenHIM core should use to reach the mediator.
        port: TCP port exposed by the mediator.

    Returns:
        An OpenHIM mediator registration mapping.
    """

    route = {
        "name": "OpenMed de-identification route",
        "host": host,
        "path": OPENHIM_MEDIATOR_PATH,
        "port": port,
        "primary": True,
        "type": "http",
    }
    return {
        "urn": DEFAULT_MEDIATOR_URN,
        "version": __version__,
        "name": "OpenMed De-identification Mediator",
        "description": (
            "Local-first FHIR and clinical free-text de-identification for OpenHIM"
        ),
        "defaultChannelConfig": [
            {
                "name": "OpenMed De-identification",
                "description": "De-identify clinical payloads inside the HIE boundary",
                "urlPattern": "^/openmed/deidentify$",
                "routes": [dict(route)],
                "allow": ["admin"],
                "methods": ["POST"],
                "type": "http",
            }
        ],
        "endpoints": [{**route, "name": "OpenMed de-identification endpoint"}],
    }


def load_mediator_registration(path: Optional[str]) -> dict[str, Any]:
    """Load and validate a mediator registration payload.

    Args:
        path: Optional JSON path.  When omitted, the built-in payload is used.

    Returns:
        A validated, mutable registration mapping.
    """

    if path:
        config_path = Path(path)
        try:
            raw = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OpenHIMMediatorConfigurationError(
                f"Unable to read OpenHIM mediator config: {config_path}"
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenHIMMediatorConfigurationError(
                f"OpenHIM mediator config is not valid JSON: {config_path}"
            ) from exc
    else:
        host = os.environ.get("OPENMED_OPENHIM_MEDIATOR_HOST", "openmed-mediator")
        raw_port = os.environ.get("OPENMED_OPENHIM_MEDIATOR_PORT", "8080")
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise OpenHIMMediatorConfigurationError(
                "OPENMED_OPENHIM_MEDIATOR_PORT must be an integer"
            ) from exc
        if not 1 <= port <= 65535:
            raise OpenHIMMediatorConfigurationError(
                "OPENMED_OPENHIM_MEDIATOR_PORT must be between 1 and 65535"
            )
        payload = default_mediator_registration(host=host, port=port)

    if not isinstance(payload, dict):
        raise OpenHIMMediatorConfigurationError(
            "OpenHIM mediator config must be a JSON object"
        )
    for key in ("urn", "name", "version"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise OpenHIMMediatorConfigurationError(
                f"OpenHIM mediator config requires a non-empty {key}"
            )
    for key in ("defaultChannelConfig", "endpoints"):
        if not isinstance(payload.get(key), list) or not payload[key]:
            raise OpenHIMMediatorConfigurationError(
                f"OpenHIM mediator config requires a non-empty {key} array"
            )
    return dict(payload)


@dataclass(frozen=True)
class OpenHIMMediatorSettings:
    """Environment-derived configuration for the OpenHIM lifecycle client."""

    enabled: bool = False
    api_url: str = ""
    username: str = ""
    password: str = field(default="", repr=False)
    auth_mode: str = "basic"
    verify_tls: bool = True
    allow_insecure_http: bool = False
    heartbeat_interval_seconds: float = 10.0
    request_timeout_seconds: float = 10.0
    policy: str = "hipaa_safe_harbor"
    method: str = "replace"
    registration: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "OpenHIMMediatorSettings":
        """Parse mediator settings without contacting OpenHIM core."""

        enabled = _parse_bool("OPENMED_OPENHIM_MEDIATOR_ENABLED", False)
        if not enabled:
            return cls()

        api_url = os.environ.get("OPENMED_OPENHIM_CORE_URL", "").strip().rstrip("/")
        username = os.environ.get("OPENMED_OPENHIM_USERNAME", "").strip()
        password = os.environ.get("OPENMED_OPENHIM_PASSWORD", "")
        missing = [
            name
            for name, value in (
                ("OPENMED_OPENHIM_CORE_URL", api_url),
                ("OPENMED_OPENHIM_USERNAME", username),
                ("OPENMED_OPENHIM_PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise OpenHIMMediatorConfigurationError(
                "OpenHIM mediator is enabled but required settings are missing: "
                + ", ".join(missing)
            )
        parsed_api_url = urlsplit(api_url)
        if parsed_api_url.scheme not in {"http", "https"} or not parsed_api_url.netloc:
            raise OpenHIMMediatorConfigurationError(
                "OPENMED_OPENHIM_CORE_URL must be an absolute HTTP(S) URL"
            )
        if parsed_api_url.username is not None or parsed_api_url.password is not None:
            raise OpenHIMMediatorConfigurationError(
                "OPENMED_OPENHIM_CORE_URL must not contain credentials"
            )
        allow_insecure_http = _parse_bool("OPENMED_OPENHIM_ALLOW_INSECURE_HTTP", False)
        if parsed_api_url.scheme == "http" and not allow_insecure_http:
            raise OpenHIMMediatorConfigurationError(
                "OPENMED_OPENHIM_CORE_URL must use https:// unless "
                "OPENMED_OPENHIM_ALLOW_INSECURE_HTTP is explicitly enabled"
            )

        heartbeat_interval = _parse_positive_float(
            "OPENMED_OPENHIM_HEARTBEAT_INTERVAL_SECONDS", 10.0
        )
        if heartbeat_interval > 30:
            raise OpenHIMMediatorConfigurationError(
                "OPENMED_OPENHIM_HEARTBEAT_INTERVAL_SECONDS must not exceed 30"
            )

        registration = load_mediator_registration(
            os.environ.get("OPENMED_OPENHIM_CONFIG_PATH")
        )
        policy = os.environ.get("OPENMED_OPENHIM_POLICY", "hipaa_safe_harbor").strip()
        if not policy:
            raise OpenHIMMediatorConfigurationError(
                "OPENMED_OPENHIM_POLICY must be non-empty"
            )
        try:
            from openmed.core.policy import load_policy

            policy = load_policy(policy).name
        except ValueError as exc:
            raise OpenHIMMediatorConfigurationError(
                "OPENMED_OPENHIM_POLICY must name a bundled policy"
            ) from exc
        return cls(
            enabled=True,
            api_url=api_url,
            username=username,
            password=password,
            auth_mode=_parse_choice(
                "OPENMED_OPENHIM_AUTH_MODE",
                "basic",
                frozenset({"basic", "token"}),
            ),
            verify_tls=_parse_bool("OPENMED_OPENHIM_VERIFY_TLS", True),
            allow_insecure_http=allow_insecure_http,
            heartbeat_interval_seconds=heartbeat_interval,
            request_timeout_seconds=_parse_positive_float(
                "OPENMED_OPENHIM_REQUEST_TIMEOUT_SECONDS", 10.0
            ),
            policy=policy,
            method=_parse_choice(
                "OPENMED_OPENHIM_METHOD",
                "replace",
                frozenset(
                    {
                        "aadhaar_mask",
                        "format_preserve",
                        "hash",
                        "mask",
                        "remove",
                        "replace",
                        "shift_dates",
                    }
                ),
            ),
            registration=registration,
        )


class OpenHIMMediatorClient:
    """Register the mediator and maintain its OpenHIM heartbeat."""

    def __init__(
        self,
        settings: OpenHIMMediatorSettings,
        *,
        client: Optional[httpx.AsyncClient] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not settings.enabled:
            raise OpenHIMMediatorConfigurationError(
                "Cannot create an OpenHIM client while the integration is disabled"
            )
        self.settings = settings
        self._monotonic = monotonic
        self._started_at = monotonic()
        self._client = client or httpx.AsyncClient(
            timeout=settings.request_timeout_seconds,
            verify=settings.verify_tls,
        )
        self._owns_client = client is None
        self._registered = False
        self._last_heartbeat_at: Optional[str] = None
        self._last_error: Optional[str] = None
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._register_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()

    @property
    def urn(self) -> str:
        """Return the configured mediator URN."""

        return str(self.settings.registration["urn"])

    async def _authentication_kwargs(self) -> dict[str, Any]:
        if self.settings.auth_mode == "basic":
            return {
                "auth": httpx.BasicAuth(
                    self.settings.username,
                    self.settings.password,
                )
            }

        username = quote(self.settings.username, safe="")
        response = await self._client.get(
            f"{self.settings.api_url}/authenticate/{username}"
        )
        if response.status_code != 200:
            raise OpenHIMMediatorProtocolError(
                f"OpenHIM authentication preflight returned {response.status_code}"
            )
        try:
            details = response.json()
        except ValueError as exc:
            raise OpenHIMMediatorProtocolError(
                "OpenHIM authentication preflight returned invalid JSON"
            ) from exc
        if not isinstance(details, Mapping):
            raise OpenHIMMediatorProtocolError(
                "OpenHIM authentication preflight returned an invalid payload"
            )
        salt = details.get("salt")
        timestamp = details.get("ts")
        if (
            not isinstance(salt, str)
            or not salt
            or not isinstance(timestamp, str)
            or not timestamp
        ):
            raise OpenHIMMediatorProtocolError(
                "OpenHIM authentication preflight omitted salt or timestamp"
            )

        password_hash = hashlib.sha512(
            f"{salt}{self.settings.password}".encode("utf-8")
        ).hexdigest()
        token = hashlib.sha512(
            f"{password_hash}{salt}{timestamp}".encode("utf-8")
        ).hexdigest()
        return {
            "headers": {
                "auth-username": self.settings.username,
                "auth-ts": timestamp,
                "auth-salt": salt,
                "auth-token": token,
            }
        }

    async def register(self) -> None:
        """Register once per client instance, even when called concurrently."""

        if self._registered:
            return
        async with self._register_lock:
            if self._registered:
                return
            authentication = await self._authentication_kwargs()
            response = await self._client.post(
                f"{self.settings.api_url}/mediators",
                json=dict(self.settings.registration),
                **authentication,
            )
            if response.status_code != 201:
                raise OpenHIMMediatorProtocolError(
                    f"OpenHIM mediator registration returned {response.status_code}"
                )
            self._registered = True
            self._last_error = None

    async def heartbeat(self, *, force_config: bool = False) -> Optional[Any]:
        """Send one heartbeat and return any updated OpenHIM configuration."""

        await self.register()
        payload: dict[str, Any] = {
            "uptime": max(0.0, self._monotonic() - self._started_at)
        }
        if force_config:
            payload["config"] = True
        encoded_urn = quote(self.urn, safe="")
        authentication = await self._authentication_kwargs()
        response = await self._client.post(
            f"{self.settings.api_url}/mediators/{encoded_urn}/heartbeat",
            json=payload,
            **authentication,
        )
        if response.status_code != 200:
            raise OpenHIMMediatorProtocolError(
                f"OpenHIM mediator heartbeat returned {response.status_code}"
            )
        self._last_heartbeat_at = _utc_timestamp()
        self._last_error = None
        if not response.content or response.text == "OK":
            return None
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text

    async def start(self) -> None:
        """Register, fetch initial config, and start periodic heartbeats."""

        async with self._start_lock:
            if self._heartbeat_task is not None:
                return
            await self.register()
            await self.heartbeat(force_config=True)
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="openhim-mediator-heartbeat"
            )

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.heartbeat_interval_seconds)
            try:
                await self.heartbeat()
            except asyncio.CancelledError:
                raise
            except (httpx.HTTPError, OpenHIMMediatorProtocolError) as exc:
                self._last_error = type(exc).__name__

    async def stop(self) -> None:
        """Stop periodic heartbeats and close the owned HTTP client."""

        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if self._owns_client:
            await self._client.aclose()

    def status(self) -> dict[str, Any]:
        """Return non-secret local heartbeat state for health checks."""

        return {
            "enabled": True,
            "registered": self._registered,
            "urn": self.urn,
            "last_heartbeat_at": self._last_heartbeat_at,
            "last_error": self._last_error,
        }


def create_openhim_mediator(
    settings: OpenHIMMediatorSettings,
) -> OpenHIMMediatorClient:
    """Create the lifecycle client used by the REST application."""

    return OpenHIMMediatorClient(settings)


@dataclass(frozen=True)
class MediatorTransformResult:
    """A transformed or byte-preserved mediator response."""

    body: bytes
    headers: Mapping[str, str]
    status_code: int
    transformed: bool
    operation: str


def mediator_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Copy response-safe representation and correlation headers only."""

    return {
        str(key): str(value)
        for key, value in headers.items()
        if str(key).lower() in _RESPONSE_HEADER_ALLOWLIST
    }


def transform_mediator_payload(
    body: bytes,
    headers: Mapping[str, str],
    status_code: int,
    *,
    deidentifier: Optional[Deidentifier] = None,
    policy: str = "hipaa_safe_harbor",
    method: str = "replace",
) -> MediatorTransformResult:
    """De-identify supported text while preserving opaque payloads exactly.

    FHIR JSON resources and Bundles use ``openmed.interop.fhir_operations``.
    Plain-text media types use the same core privacy pipeline.  Any other body
    is returned byte-for-byte with the supplied headers and status code.
    """

    copied_headers = dict(headers)
    media_type = _media_type(copied_headers)

    if media_type == "application/fhir+json" or media_type in {
        "application/json",
        "application/json+fhir",
    }:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if media_type == "application/fhir+json":
                raise ValueError("FHIR payload must be valid UTF-8 JSON") from None
            return MediatorTransformResult(
                body=body,
                headers=copied_headers,
                status_code=status_code,
                transformed=False,
                operation="pass-through",
            )

        resource_type = (
            payload.get("resourceType") if isinstance(payload, dict) else None
        )
        if resource_type is None:
            if media_type == "application/fhir+json":
                raise ValueError("FHIR payload requires resourceType")
            return MediatorTransformResult(
                body=body,
                headers=copied_headers,
                status_code=status_code,
                transformed=False,
                operation="pass-through",
            )

        from openmed.interop.fhir_operations import (
            de_identify,
            de_identify_bundle,
            de_identify_resource,
        )

        if resource_type == "Bundle":
            transformed = de_identify_bundle(
                payload,
                policy=policy,
                method=method,
                deidentifier=deidentifier,
            )
            operation = "fhir-bundle-de-identify"
        elif resource_type == "Parameters":
            transformed = de_identify(
                _configured_fhir_parameters(
                    payload,
                    policy=policy,
                    method=method,
                ),
                deidentifier=deidentifier,
            )
            operation = "fhir-operation-de-identify"
        else:
            transformed = de_identify_resource(
                payload,
                policy=policy,
                method=method,
                deidentifier=deidentifier,
            )
            operation = "fhir-resource-de-identify"
        return MediatorTransformResult(
            body=json.dumps(
                transformed, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"),
            headers=copied_headers,
            status_code=status_code,
            transformed=True,
            operation=operation,
        )

    if media_type.startswith("text/"):
        try:
            text = body.decode(_charset(copied_headers))
        except (LookupError, UnicodeDecodeError) as exc:
            raise ValueError("Text payload encoding is invalid") from exc
        result = _deidentify_text(
            text,
            deidentifier=deidentifier,
            policy=policy,
            method=method,
        )
        return MediatorTransformResult(
            body=result.encode(_charset(copied_headers)),
            headers=copied_headers,
            status_code=status_code,
            transformed=True,
            operation="text-de-identify",
        )

    return MediatorTransformResult(
        body=body,
        headers=copied_headers,
        status_code=status_code,
        transformed=False,
        operation="pass-through",
    )


def build_openhim_envelope(
    result: MediatorTransformResult,
    *,
    urn: str,
    request_method: str = "POST",
    request_path: str = OPENHIM_MEDIATOR_PATH,
) -> dict[str, Any]:
    """Build the structured response consumed by OpenHIM core."""

    timestamp = _utc_timestamp()
    response = {
        "status": result.status_code,
        "headers": dict(result.headers),
        "body": result.body.decode(_charset(result.headers)),
        "timestamp": timestamp,
    }
    if 200 <= result.status_code < 300:
        transaction_status = "Successful"
    elif result.status_code >= 500:
        transaction_status = "Failed"
    else:
        transaction_status = "Completed with error(s)"
    envelope: dict[str, Any] = {
        "x-mediator-urn": urn,
        "status": transaction_status,
        "response": response,
        "orchestrations": [
            {
                "name": "OpenMed local de-identification",
                "request": {
                    "method": request_method,
                    "path": request_path,
                    "headers": {},
                    "timestamp": timestamp,
                },
                "response": dict(response),
            }
        ],
        "properties": {
            "openmed.operation": result.operation,
            "openmed.local-first": True,
        },
    }
    if result.status_code >= 500:
        envelope["error"] = {"message": "Mediator request failed"}
    return envelope


def failed_openhim_result(status_code: int, code: str) -> MediatorTransformResult:
    """Return a PHI-free structured error body for the mediator envelope."""

    body = json.dumps(
        {"error": {"code": code, "message": "Mediator request failed"}},
        separators=(",", ":"),
    ).encode("utf-8")
    return MediatorTransformResult(
        body=body,
        headers={"content-type": "application/json"},
        status_code=status_code,
        transformed=True,
        operation="de-identification-error",
    )


def _deidentify_text(
    text: str,
    *,
    deidentifier: Optional[Deidentifier],
    policy: str,
    method: str,
) -> str:
    if deidentifier is None:
        from openmed.core.pii import deidentify

        deidentifier = deidentify
    result = deidentifier(text, policy=policy, method=method)
    clean_text = getattr(result, "deidentified_text", None)
    if not isinstance(clean_text, str):
        raise TypeError("deidentifier must return an object with deidentified_text")
    return clean_text


def _configured_fhir_parameters(
    payload: Mapping[str, Any],
    *,
    policy: str,
    method: str,
) -> dict[str, Any]:
    parameters = payload.get("parameter")
    retained = (
        [
            item
            for item in parameters
            if not (
                isinstance(item, Mapping) and item.get("name") in {"policy", "method"}
            )
        ]
        if isinstance(parameters, list)
        else []
    )
    return {
        **dict(payload),
        "parameter": [
            *retained,
            {"name": "policy", "valueString": policy},
            {"name": "method", "valueCode": method},
        ],
    }


def _media_type(headers: Mapping[str, str]) -> str:
    for key, value in headers.items():
        if key.lower() == "content-type":
            return value.split(";", 1)[0].strip().lower()
    return "application/octet-stream"


def _charset(headers: Mapping[str, str]) -> str:
    for key, value in headers.items():
        if key.lower() != "content-type":
            continue
        for parameter in value.split(";")[1:]:
            name, separator, charset = parameter.partition("=")
            if separator and name.strip().lower() == "charset":
                return charset.strip().strip('"')
    return "utf-8"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
