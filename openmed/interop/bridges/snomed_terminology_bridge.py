"""Out-of-process SNOMED CT terminology-server grounding.

This module deliberately contains no SNOMED CT release data.  It sends only
the caller-provided span (or an explicit concept id) to a terminology server
that the caller operates and licenses.  Responses are converted immediately
to the shared :class:`~openmed.clinical.grounding.matcher.ConceptMatch` shape;
they are not cached, persisted, or included in OpenMed artifacts.
"""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from openmed.clinical.grounding.matcher import ConceptMatch, normalize_term

SNOMED_SYSTEM_URI = "http://snomed.info/sct"
SNOMED_TERMINOLOGY_SYSTEM = SNOMED_SYSTEM_URI
DEFAULT_SCTID_PATTERN = r"[0-9]{6,18}"

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_LIMIT = 5
_MAX_LIMIT = 50
_MAX_QUERY_CHARS = 1_024
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_SCTID_RE = re.compile(rf"^{DEFAULT_SCTID_PATTERN}$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_NESTED_SEARCH_KEYS = frozenset(
    {
        "bundle",
        "children",
        "concept",
        "contains",
        "data",
        "descriptions",
        "entry",
        "expansion",
        "items",
        "match",
        "matches",
        "resource",
        "response",
        "result",
        "results",
    }
)
_DISPLAY_KEYS = (
    "display",
    "term",
    "preferredTerm",
    "preferred_term",
    "fullySpecifiedName",
    "fsn",
)
_CODE_KEYS = ("code", "conceptId", "concept_id", "sctid", "id")

__all__ = [
    "DEFAULT_SCTID_PATTERN",
    "SNOMED_SYSTEM_URI",
    "SNOMED_TERMINOLOGY_SYSTEM",
    "SNOMEDTerminologyBridge",
    "SNOMEDTerminologyBridgeError",
    "SNOMEDTerminologyConfig",
    "SNOMEDTerminologyConfigurationError",
    "SNOMEDTerminologyServerError",
    "SnomedTerminologyBridge",
]


class SNOMEDTerminologyBridgeError(RuntimeError):
    """Base error for a configured SNOMED terminology-server request."""


class SNOMEDTerminologyConfigurationError(SNOMEDTerminologyBridgeError, ValueError):
    """Raised when the caller has not supplied a usable bridge configuration."""


class SNOMEDTerminologyServerError(SNOMEDTerminologyBridgeError):
    """Raised when the terminology server response cannot be used."""


@dataclass(frozen=True)
class SNOMEDTerminologyConfig:
    """Connection settings for one caller-operated terminology server.

    ``endpoint`` is the server's FHIR base URL, such as
    ``https://terminology.example/fhir``.  Credentials are optional because a
    user-operated local server may not require them.  When credentials are
    needed, provide a bearer token, API key, basic-auth pair, or explicit
    credential headers.  Secret fields and headers are excluded from the
    representation so they are not accidentally printed in diagnostics.
    """

    endpoint: str
    timeout: float = _DEFAULT_TIMEOUT
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    bearer_token: str | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)
    api_key_header: str = "X-API-Key"
    username: str | None = field(default=None, repr=False)
    password: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        object.__setattr__(self, "timeout", _validate_timeout(self.timeout))
        _validate_header_name(self.api_key_header)
        normalized_headers = _validate_headers(self.headers)
        object.__setattr__(self, "headers", normalized_headers)

        bearer_token = _optional_secret(self.bearer_token, "bearer_token")
        api_key = _optional_secret(self.api_key, "api_key")
        username = _optional_secret(self.username, "username")
        password = _optional_secret(self.password, "password")
        if (username is None) != (password is None):
            raise SNOMEDTerminologyConfigurationError(
                "SNOMED terminology basic authentication requires both username "
                "and password."
            )
        if sum(value is not None for value in (bearer_token, api_key, username)) > 1:
            raise SNOMEDTerminologyConfigurationError(
                "Configure one SNOMED terminology credential mechanism; use "
                "headers for a server-specific scheme."
            )
        if bearer_token is not None and _has_header(
            normalized_headers, "authorization"
        ):
            raise SNOMEDTerminologyConfigurationError(
                "Do not configure both bearer_token and an Authorization header."
            )
        if api_key is not None and _has_header(normalized_headers, self.api_key_header):
            raise SNOMEDTerminologyConfigurationError(
                "Do not configure both api_key and its API-key header."
            )
        if username is not None and _has_header(normalized_headers, "authorization"):
            raise SNOMEDTerminologyConfigurationError(
                "Do not configure both basic authentication and an Authorization "
                "header."
            )
        object.__setattr__(self, "bearer_token", bearer_token)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "username", username)
        object.__setattr__(self, "password", password)


@dataclass(frozen=True)
class _RemoteConcept:
    code: str
    display: str
    terms: tuple[str, ...]
    remote_score: float | None = None


class SNOMEDTerminologyBridge:
    """Resolve SNOMED terms through a user-supplied FHIR terminology server.

    The bridge has no bundled terminology content and does not write a cache.
    Text lookup uses ``GET CodeSystem`` with the FHIR ``filter`` parameter;
    numeric SNOMED concept ids use ``GET CodeSystem/$lookup``.  An injected
    ``opener`` or HTTPX-compatible ``client`` is useful for synthetic offline
    tests.  The caller owns an injected client and its lifecycle.

    Args:
        endpoint: FHIR base URL for the caller-operated terminology server.
        config: Optional pre-built :class:`SNOMEDTerminologyConfig`.
        base_url: Alias for ``endpoint`` for client compatibility.
        headers: Additional request headers, including server-specific
            credentials when needed.
        bearer_token: Optional bearer credential kept only in memory.
        token: Alias for ``bearer_token``.
        api_key: Optional API-key credential kept only in memory.
        username: Optional basic-auth username.
        password: Optional basic-auth password.
        client: Optional object exposing ``get(url, params=, headers=)``.
        opener: Optional urllib-compatible callable accepting a Request and
            timeout.  ``client`` and ``opener`` are mutually exclusive.

    Raises:
        SNOMEDTerminologyConfigurationError: If an endpoint is absent or
            connection settings are invalid.
    """

    system_uri = SNOMED_SYSTEM_URI

    def __init__(
        self,
        endpoint: str | SNOMEDTerminologyConfig | None = None,
        *,
        config: SNOMEDTerminologyConfig | None = None,
        base_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        headers: Mapping[str, str] | None = None,
        bearer_token: str | None = None,
        token: str | None = None,
        api_key: str | None = None,
        api_key_header: str = "X-API-Key",
        username: str | None = None,
        password: str | None = None,
        client: Any | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if client is not None and opener is not None:
            raise SNOMEDTerminologyConfigurationError(
                "Provide either an HTTP client or an opener, not both."
            )
        if config is not None:
            if endpoint is not None or base_url is not None:
                raise SNOMEDTerminologyConfigurationError(
                    "Provide either config or endpoint/base_url, not both."
                )
            if (
                any(
                    value is not None
                    for value in (
                        headers,
                        bearer_token,
                        token,
                        api_key,
                        username,
                        password,
                    )
                )
                or timeout != _DEFAULT_TIMEOUT
                or api_key_header != "X-API-Key"
            ):
                raise SNOMEDTerminologyConfigurationError(
                    "Connection options must be supplied through config when "
                    "config is provided."
                )
            resolved_config = config
        else:
            if isinstance(endpoint, SNOMEDTerminologyConfig):
                if base_url is not None:
                    raise SNOMEDTerminologyConfigurationError(
                        "Provide either a config object or base_url, not both."
                    )
                resolved_config = endpoint
            else:
                resolved_endpoint = endpoint if endpoint is not None else base_url
                if resolved_endpoint is None:
                    raise SNOMEDTerminologyConfigurationError(
                        "A user-supplied SNOMED terminology endpoint is required; "
                        "OpenMed does not bundle SNOMED CT or silently fall back "
                        "to an in-process vocabulary."
                    )
                if token is not None and bearer_token is not None:
                    raise SNOMEDTerminologyConfigurationError(
                        "Provide either token or bearer_token, not both."
                    )
                resolved_config = SNOMEDTerminologyConfig(
                    endpoint=str(resolved_endpoint),
                    timeout=timeout,
                    headers=headers or {},
                    bearer_token=bearer_token if bearer_token is not None else token,
                    api_key=api_key,
                    api_key_header=api_key_header,
                    username=username,
                    password=password,
                )
        if not isinstance(resolved_config, SNOMEDTerminologyConfig):
            raise SNOMEDTerminologyConfigurationError(
                "config must be a SNOMEDTerminologyConfig instance."
            )
        if client is not None and not callable(getattr(client, "get", None)):
            raise SNOMEDTerminologyConfigurationError(
                "client must expose a callable get(url, params=, headers=) method."
            )
        if opener is not None and not callable(opener):
            raise SNOMEDTerminologyConfigurationError("opener must be callable.")

        self.config = resolved_config
        self.endpoint = resolved_config.endpoint
        self.base_url = self.endpoint
        self._client = client
        self._opener = opener

    def __enter__(self) -> SNOMEDTerminologyBridge:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release bridge-owned resources.

        The default urllib path has no persistent client.  An injected client
        remains owned by its caller and is intentionally not closed here.
        """

    def lookup(
        self,
        term: str,
        *,
        limit: int = _DEFAULT_LIMIT,
        language: str | None = None,
    ) -> tuple[ConceptMatch, ...]:
        """Resolve a span term to ranked SNOMED :class:`ConceptMatch` values.

        Numeric SCTIDs use FHIR ``CodeSystem/$lookup``.  Other terms use FHIR
        ``CodeSystem`` search.  An empty result is an ordinary terminology
        abstention; missing configuration and invalid server responses raise
        explicit errors.
        """

        query = _validate_query(term, field_name="term")
        normalized_limit = _validate_limit(limit)
        if _SCTID_RE.fullmatch(query):
            return self.lookup_code(query, language=language)
        return self.search(query, limit=normalized_limit, language=language)

    def match(
        self,
        term: str,
        *,
        limit: int = _DEFAULT_LIMIT,
        language: str | None = None,
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`lookup`, matching the shared matcher interface."""

        return self.lookup(term, limit=limit, language=language)

    def link(
        self,
        term: str,
        *,
        limit: int = _DEFAULT_LIMIT,
        language: str | None = None,
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`lookup` for linker-oriented call sites."""

        return self.lookup(term, limit=limit, language=language)

    def resolve(
        self,
        term: str,
        *,
        limit: int = _DEFAULT_LIMIT,
        language: str | None = None,
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`lookup` for terminology-resolution call sites."""

        return self.lookup(term, limit=limit, language=language)

    def search(
        self,
        term: str,
        *,
        limit: int = _DEFAULT_LIMIT,
        language: str | None = None,
    ) -> tuple[ConceptMatch, ...]:
        """Search the SNOMED CodeSystem for one caller-provided span term."""

        query = _validate_query(term, field_name="term")
        normalized_limit = _validate_limit(limit)
        params: dict[str, str] = {
            "url": self.system_uri,
            "filter": query,
            "_count": str(normalized_limit),
        }
        language_value = _validate_language(language)
        if language_value is not None:
            params["displayLanguage"] = language_value
        payload = self._get_json("CodeSystem", params)
        records = _iter_search_concepts(payload)
        return _to_matches(
            records,
            query=query,
            limit=normalized_limit,
            source="fhir-code-system-search",
        )

    def lookup_code(
        self,
        code: str,
        *,
        language: str | None = None,
    ) -> tuple[ConceptMatch, ...]:
        """Resolve one SCTID with FHIR ``CodeSystem/$lookup``."""

        normalized_code = _validate_query(code, field_name="code")
        params = {"system": self.system_uri, "code": normalized_code}
        language_value = _validate_language(language)
        if language_value is not None:
            params["displayLanguage"] = language_value
        payload = self._get_json("CodeSystem/$lookup", params)
        records = _iter_lookup_concepts(payload, requested_code=normalized_code)
        return _to_matches(
            records,
            query=normalized_code,
            limit=1,
            source="fhir-code-system-lookup",
            force_exact=True,
        )

    def _get_json(self, path: str, params: Mapping[str, str]) -> dict[str, Any]:
        headers = self._request_headers()
        if self._client is not None:
            try:
                response = self._client.get(
                    f"{self.endpoint.rstrip('/')}/{path.lstrip('/')}",
                    params=dict(params),
                    headers=headers,
                )
            except (TimeoutError, OSError) as exc:
                raise SNOMEDTerminologyServerError(
                    "SNOMED terminology server request failed."
                ) from exc
            status_code = _response_status(response)
            if status_code >= 300:
                raise SNOMEDTerminologyServerError(
                    f"SNOMED terminology server returned HTTP {status_code}."
                )
            return _decode_client_response(response)

        request = urlrequest.Request(
            _request_url(self.endpoint, path, params),
            method="GET",
            headers=headers,
        )
        opener = self._opener or _open_without_redirects
        try:
            with opener(request, timeout=self.config.timeout) as response:
                status_code = int(getattr(response, "status", 200))
                if status_code >= 300:
                    raise SNOMEDTerminologyServerError(
                        f"SNOMED terminology server returned HTTP {status_code}."
                    )
                content_length = _content_length(getattr(response, "headers", {}))
                if content_length is not None and content_length > _MAX_RESPONSE_BYTES:
                    raise SNOMEDTerminologyServerError(
                        "SNOMED terminology server response is too large."
                    )
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except SNOMEDTerminologyServerError:
            raise
        except urlerror.HTTPError as exc:
            raise SNOMEDTerminologyServerError(
                f"SNOMED terminology server returned HTTP {exc.code}."
            ) from exc
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            raise SNOMEDTerminologyServerError(
                "SNOMED terminology server request failed."
            ) from exc
        return _decode_json_bytes(raw)

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/fhir+json, application/json",
            **dict(self.config.headers),
        }
        if self.config.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"
        elif self.config.api_key is not None:
            headers[self.config.api_key_header] = self.config.api_key
        elif self.config.username is not None and self.config.password is not None:
            credentials = f"{self.config.username}:{self.config.password}".encode(
                "utf-8"
            )
            headers["Authorization"] = "Basic " + base64.b64encode(credentials).decode(
                "ascii"
            )
        return headers


SnomedTerminologyBridge = SNOMEDTerminologyBridge


def _open_without_redirects(
    request: urlrequest.Request,
    *,
    timeout: float,
) -> Any:
    class _NoRedirect(urlrequest.HTTPRedirectHandler):
        def redirect_request(
            self,
            req: urlrequest.Request,
            fp: Any,
            code: int,
            msg: str,
            headers: Any,
            newurl: str,
        ) -> None:
            return None

    return urlrequest.build_opener(_NoRedirect()).open(request, timeout=timeout)


def _validate_endpoint(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SNOMEDTerminologyConfigurationError(
            "A user-supplied SNOMED terminology endpoint is required; "
            "OpenMed does not bundle SNOMED CT."
        )
    endpoint = value.strip()
    parsed = urlparse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SNOMEDTerminologyConfigurationError(
            "SNOMED terminology endpoint must be an absolute http(s) URL."
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SNOMEDTerminologyConfigurationError(
            "SNOMED terminology endpoint must not contain credentials, query, "
            "or fragment data; configure credentials explicitly."
        )
    if any(character.isspace() for character in endpoint):
        raise SNOMEDTerminologyConfigurationError(
            "SNOMED terminology endpoint must not contain whitespace."
        )
    return endpoint.rstrip("/")


def _validate_timeout(value: object) -> float:
    if isinstance(value, bool):
        raise SNOMEDTerminologyConfigurationError("timeout must be a positive number.")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise SNOMEDTerminologyConfigurationError(
            "timeout must be a positive number."
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise SNOMEDTerminologyConfigurationError("timeout must be a positive number.")
    return timeout


def _optional_secret(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SNOMEDTerminologyConfigurationError(
            f"{field_name} must be a non-empty string when provided."
        )
    return value.strip()


def _validate_headers(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise SNOMEDTerminologyConfigurationError("headers must be a mapping.")
    normalized: dict[str, str] = {}
    for name, header_value in value.items():
        _validate_header_name(name)
        if not isinstance(header_value, str) or any(
            character in header_value for character in "\r\n"
        ):
            raise SNOMEDTerminologyConfigurationError(
                "SNOMED terminology header values must be strings without newlines."
            )
        normalized[str(name)] = header_value
    return normalized


def _validate_header_name(value: object) -> None:
    if not isinstance(value, str) or not _HEADER_NAME_RE.fullmatch(value):
        raise SNOMEDTerminologyConfigurationError(
            "SNOMED terminology header names must be valid HTTP field names."
        )


def _has_header(headers: Mapping[str, str], name: str) -> bool:
    return any(key.casefold() == name.casefold() for key in headers)


def _validate_query(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    query = value.strip()
    if not query:
        raise ValueError(f"{field_name} must not be empty")
    if len(query) > _MAX_QUERY_CHARS:
        raise ValueError(f"{field_name} must not exceed {_MAX_QUERY_CHARS} characters")
    return query


def _validate_limit(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("limit must be a positive integer")
    if value > _MAX_LIMIT:
        raise ValueError(f"limit must not exceed {_MAX_LIMIT}")
    return value


def _validate_language(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("language must be a string or None")
    language = value.strip()
    if not language:
        raise ValueError("language must not be empty when provided")
    if len(language) > 32 or any(character.isspace() for character in language):
        raise ValueError("language must be a compact language tag")
    return language


def _request_url(endpoint: str, path: str, params: Mapping[str, str]) -> str:
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    query = urlparse.urlencode(list(params.items()))
    return f"{url}?{query}" if query else url


def _response_status(response: Any) -> int:
    try:
        status_code = int(response.status_code)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SNOMEDTerminologyServerError(
            "SNOMED terminology server response omitted an HTTP status."
        ) from exc
    return status_code


def _content_length(headers: Any) -> int | None:
    if not isinstance(headers, Mapping):
        return None
    value = headers.get("Content-Length") or headers.get("content-length")
    if value is None:
        return None
    try:
        length = int(value)
    except (TypeError, ValueError):
        return None
    return length if length >= 0 else None


def _decode_client_response(response: Any) -> dict[str, Any]:
    content_length = _content_length(getattr(response, "headers", {}))
    if content_length is not None and content_length > _MAX_RESPONSE_BYTES:
        raise SNOMEDTerminologyServerError(
            "SNOMED terminology server response is too large."
        )
    raw = getattr(response, "content", None)
    if raw is not None:
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if isinstance(raw, bytes):
            return _decode_json_bytes(raw)
    try:
        payload = response.json()
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SNOMEDTerminologyServerError(
            "SNOMED terminology server returned invalid JSON."
        ) from exc
    if not isinstance(payload, Mapping):
        raise SNOMEDTerminologyServerError(
            "SNOMED terminology server returned a non-object JSON response."
        )
    return dict(payload)


def _decode_json_bytes(raw: bytes) -> dict[str, Any]:
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise SNOMEDTerminologyServerError(
            "SNOMED terminology server response is too large."
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SNOMEDTerminologyServerError(
            "SNOMED terminology server returned invalid JSON."
        ) from exc
    if not isinstance(payload, Mapping):
        raise SNOMEDTerminologyServerError(
            "SNOMED terminology server returned a non-object JSON response."
        )
    return dict(payload)


def _iter_search_concepts(payload: Mapping[str, Any]) -> tuple[_RemoteConcept, ...]:
    records: list[_RemoteConcept] = []
    seen: set[int] = set()

    def visit(
        node: Any,
        *,
        system_hint: str | None = None,
        score_hint: float | None = None,
    ) -> None:
        if not isinstance(node, Mapping) or id(node) in seen:
            return
        seen.add(id(node))
        local_system = _system_value(node)
        system = local_system if local_system is not None else system_hint
        score = _remote_score(node)
        if score is None and isinstance(node.get("search"), Mapping):
            score = _remote_score(node["search"])
        if score is None:
            score = score_hint
        if system is None or _is_snomed_system(system):
            record = _remote_concept(node, score=score)
            if record is not None:
                records.append(record)
        for key in _NESTED_SEARCH_KEYS:
            child = node.get(key)
            if isinstance(child, Mapping):
                visit(child, system_hint=system, score_hint=score)
            elif isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
                for item in child:
                    visit(item, system_hint=system, score_hint=score)

    visit(payload)
    return tuple(records)


def _iter_lookup_concepts(
    payload: Mapping[str, Any], *, requested_code: str
) -> tuple[_RemoteConcept, ...]:
    if payload.get("resourceType") == "OperationOutcome":
        return ()
    parameters = payload.get("parameter")
    if isinstance(parameters, Sequence) and not isinstance(parameters, (str, bytes)):
        display: str | None = None
        terms: list[str] = []
        code = requested_code
        for parameter in parameters:
            if not isinstance(parameter, Mapping):
                continue
            name = parameter.get("name")
            if name == "display":
                display = _parameter_value(parameter) or display
            elif name == "code":
                code = _parameter_value(parameter) or code
            elif name == "designation":
                terms.extend(_parameter_designation_terms(parameter))
        if display:
            return (
                _RemoteConcept(
                    code=code,
                    display=display,
                    terms=tuple(dict.fromkeys((display, *terms))),
                ),
            )

    records = _iter_search_concepts(payload)
    matching = tuple(record for record in records if record.code == requested_code)
    return matching or records[:1]


def _parameter_value(parameter: Mapping[str, Any]) -> str | None:
    for key in (
        "valueString",
        "valueMarkdown",
        "valueCode",
        "valueUri",
        "valueCanonical",
    ):
        value = parameter.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    coding = parameter.get("valueCoding")
    if isinstance(coding, Mapping):
        display = coding.get("display")
        if isinstance(display, str) and display.strip():
            return display.strip()
    return None


def _parameter_designation_terms(parameter: Mapping[str, Any]) -> tuple[str, ...]:
    terms: list[str] = []
    parts = parameter.get("part")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        return ()
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        if part.get("name") == "value":
            value = _parameter_value(part)
            if value:
                terms.append(value)
    return tuple(dict.fromkeys(terms))


def _system_value(node: Mapping[str, Any]) -> str | None:
    for key in ("system", "url"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    coding = node.get("valueCoding")
    if isinstance(coding, Mapping):
        value = coding.get("system")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_snomed_system(value: str) -> bool:
    normalized = value.rstrip("/").casefold()
    return normalized == SNOMED_SYSTEM_URI or normalized.startswith(
        f"{SNOMED_SYSTEM_URI}/"
    )


def _remote_concept(
    node: Mapping[str, Any], *, score: float | None
) -> _RemoteConcept | None:
    resource_type = node.get("resourceType")
    code = _first_text(node, _CODE_KEYS)
    if not code or resource_type in {"Bundle", "CodeSystem", "Parameters"}:
        return None
    display_values: list[str] = []
    for key in _DISPLAY_KEYS:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            display_values.append(value.strip())
    for key in ("preferredSynonym", "synonym"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            display_values.append(value.strip())
    designations = node.get("designation")
    if isinstance(designations, Sequence) and not isinstance(
        designations, (str, bytes)
    ):
        for designation in designations:
            if not isinstance(designation, Mapping):
                continue
            value = designation.get("value") or designation.get("term")
            if isinstance(value, str) and value.strip():
                display_values.append(value.strip())
    unique_values = tuple(dict.fromkeys(display_values))
    if not unique_values:
        return None
    return _RemoteConcept(
        code=code,
        display=unique_values[0],
        terms=unique_values,
        remote_score=score,
    )


def _first_text(node: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return ""


def _remote_score(node: Mapping[str, Any]) -> float | None:
    for key in ("score", "matchScore", "similarity"):
        value = node.get(key)
        if isinstance(value, bool):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score) or score < 0:
            continue
        if score > 1.0 and score <= 100.0:
            score /= 100.0
        if 0.0 <= score <= 1.0:
            return score
    return None


def _to_matches(
    records: Sequence[_RemoteConcept],
    *,
    query: str,
    limit: int,
    source: str,
    force_exact: bool = False,
) -> tuple[ConceptMatch, ...]:
    normalized_query = normalize_term(query)
    best: dict[tuple[str, str], ConceptMatch] = {}
    for record in records:
        if force_exact:
            matched_term = record.display
            match_type = "exact"
            score = 1.0
        else:
            matched_term, match_type = _best_term(record.terms, query, normalized_query)
            if match_type == "exact":
                score = 1.0
            elif record.remote_score is not None:
                score = record.remote_score
            else:
                score = 0.95 if match_type == "normalized" else 0.75
        candidate = ConceptMatch(
            system_uri=SNOMED_SYSTEM_URI,
            code=record.code,
            display=record.display,
            score=score,
            match_type=match_type,  # type: ignore[arg-type]
            matched_term=matched_term,
            metadata={"source": source},
        )
        key = candidate.key
        existing = best.get(key)
        if existing is None or _match_sort_key(candidate) < _match_sort_key(existing):
            best[key] = candidate
    matches = sorted(best.values(), key=_match_sort_key)
    return tuple(matches[:limit])


def _best_term(
    terms: Sequence[str], query: str, normalized_query: str
) -> tuple[str, str]:
    for term in terms:
        if term == query:
            return term, "exact"
    for term in terms:
        if normalize_term(term) == normalized_query:
            return term, "normalized"
    return terms[0], "normalized"


def _match_sort_key(match: ConceptMatch) -> tuple[float, str, str, str]:
    return (-match.score, match.code, match.display, match.matched_term)
