"""Local-first FHIR R4 server de-identification connector.

The pure transformation functions in this module operate on in-memory FHIR
JSON and never make network requests.  :class:`FHIRServerClient` adds a small
HTTPX-backed read/paginate/write boundary for callers that explicitly opt in
to server access and write-back.  HTTPX is imported only when a client is
created without an injected transport, so importing OpenMed remains free of
HTTP-client side effects.
"""

from __future__ import annotations

import base64
import binascii
import copy
import html
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Final
from urllib.parse import quote, urljoin, urlsplit

_DEFAULT_POLICY = "hipaa_safe_harbor"
_DEFAULT_METHOD = "replace"

Deidentifier = Callable[..., Any]

# These keys describe FHIR structure, codes, references, or machine-readable
# values.  Everything else is treated as text, which covers common narrative
# fields such as ``description``, ``note[].text``, ``display``, and
# ``DiagnosticReport.conclusion`` without maintaining a resource-type table.
_SKIP_CONTAINERS: Final[frozenset[str]] = frozenset({"coding", "meta"})
_SKIP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "resourceType",
        "id",
        "fullUrl",
        "reference",
        "type",
        "system",
        "code",
        "version",
        "url",
        "uri",
        "profile",
        "status",
        "docStatus",
        "use",
        "gender",
        "unit",
        "comparator",
        "contentType",
        "language",
        "relation",
        "method",
        "mode",
        "format",
        "action",
        "direction",
        "kind",
        "purpose",
        "div",
        "data",
        "valueCode",
        "valueUri",
        "valueUrl",
        "valueCanonical",
        "valueId",
        "valueOid",
        "valueUuid",
        "valueDateTime",
        "valueDate",
        "valueInstant",
        "valueTime",
        "valueBase64Binary",
        "date",
        "dateTime",
        "instant",
        "birthDate",
        "deceasedDateTime",
        "start",
        "end",
        "issued",
        "authoredOn",
        "recorded",
        "created",
        "effectiveDateTime",
        "effectiveInstant",
        "time",
        "when",
        "timestamp",
    }
)
_TEXT_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {
        "application/fhir+json",
        "application/json",
        "application/rtf",
        "application/xhtml+xml",
        "application/xml",
        "text/html",
        "text/plain",
        "text/xml",
    }
)


def deidentify_resource(
    resource: Mapping[str, Any],
    policy: Any = _DEFAULT_POLICY,
    *,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
) -> dict[str, Any]:
    """Return a de-identified, deep-copied FHIR resource.

    Narrative XHTML, ordinary free-text primitives, and textual attachment
    payloads are passed to OpenMed's de-identification pipeline.  Coded
    values, references, URLs, dates, and other structural values are copied
    unchanged.  The input mapping is never mutated.

    Args:
        resource: FHIR R4 JSON resource containing ``resourceType``.
        policy: OpenMed policy profile passed to the de-identification
            callable.  It is positional for parity with the connector's
            small public API.
        method: De-identification method passed to the callable.
        deidentifier: Optional callable override for offline tests.  It must
            return an object with ``deidentified_text`` or a string.

    Raises:
        TypeError: If ``resource`` is not a mapping.
        ValueError: If ``resourceType`` is missing or empty.
    """

    _validate_resource(resource)
    transformed = copy.deepcopy(dict(resource))
    deid = _bind_text_deidentifier(
        deidentifier,
        policy=policy,
        method=method,
    )
    if transformed.get("resourceType") == "Bundle":
        _deidentify_bundle_in_place(transformed, deid)
    else:
        _walk_resource(transformed, str(transformed["resourceType"]), deid)
    return transformed


def deidentify_bundle(
    bundle: Mapping[str, Any],
    policy: Any = _DEFAULT_POLICY,
    *,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
) -> dict[str, Any]:
    """Return a de-identified, deep-copied FHIR ``Bundle``.

    Every ``entry.resource`` is transformed, including nested Bundles, while
    Bundle links, request/response metadata, references, and entry order are
    preserved.  The Bundle's own ``text.div`` narrative is also transformed.
    """

    if not isinstance(bundle, Mapping):
        raise TypeError("bundle must be a FHIR Bundle mapping")
    if bundle.get("resourceType") != "Bundle":
        raise ValueError("bundle resourceType must be 'Bundle'")

    transformed = copy.deepcopy(dict(bundle))
    deid = _bind_text_deidentifier(
        deidentifier,
        policy=policy,
        method=method,
    )
    _deidentify_bundle_in_place(transformed, deid)
    return transformed


# Compatibility aliases for callers that use the spelling from the existing
# FHIR operation module.
de_identify_resource = deidentify_resource
de_identify_bundle = deidentify_bundle


@dataclass(frozen=True)
class FHIRServerConfig:
    """Connection settings for one FHIR R4 server.

    Authentication is intentionally supplied by the caller through headers
    or ``bearer_token``; neither value is included in representations or
    error messages.  The configured base URL is also the trust boundary for
    pagination links and write-back targets.
    """

    base_url: str
    timeout: float = 30.0
    verify_tls: bool = True
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    bearer_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_base_url(self.base_url)
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        for name, value in self.headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise TypeError("FHIR headers must be string-to-string values")
            if any(character in value for character in "\r\n"):
                raise ValueError("FHIR headers must not contain line breaks")
        if self.bearer_token is not None and any(
            character in self.bearer_token for character in "\r\n"
        ):
            raise ValueError("bearer_token must not contain line breaks")


@dataclass(frozen=True)
class FHIRWriteResult:
    """PHI-free summary of one explicit FHIR PUT attempt."""

    resource_type: str
    resource_id: str
    url: str
    status_code: int | None
    written: bool


class FHIRServerClient:
    """Small synchronous FHIR R4 REST client with safe redaction defaults.

    ``client`` may be an HTTPX-compatible client or mock transport.  When it
    is omitted, HTTPX is imported lazily and an internal client is created.
    Reads follow FHIR Bundle ``link`` entries with ``relation == "next"``.
    Write-back is disabled unless a call passes ``write=True`` explicitly.
    """

    def __init__(
        self,
        config: FHIRServerConfig | str,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = (
            config if isinstance(config, FHIRServerConfig) else FHIRServerConfig(config)
        )
        self._owns_client = client is None
        if client is None:
            httpx = _import_httpx()
            headers = dict(self.config.headers)
            if self.config.bearer_token is not None:
                headers.setdefault(
                    "Authorization", f"Bearer {self.config.bearer_token}"
                )
            self._client = httpx.Client(
                headers=headers,
                timeout=self.config.timeout,
                verify=self.config.verify_tls,
            )
        else:
            self._client = client

    def close(self) -> None:
        """Close the internally-created HTTP client, if any."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "FHIRServerClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def get_resource(self, resource_type: str, resource_id: str) -> dict[str, Any]:
        """Fetch one FHIR resource by type and id."""

        url = self._resource_url(resource_type, resource_id)
        payload = self._get_json(url)
        _validate_resource(payload)
        return payload

    def get_bundle(
        self,
        resource_type: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fetch the first Bundle page for a FHIR resource collection."""

        return next(self.iter_bundle_pages(resource_type, params=params))

    fetch_bundle = get_bundle

    def iter_bundle_pages(
        self,
        resource_type: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield a collection Bundle and all of its ``next`` pages."""

        current_url = self._collection_url(resource_type)
        current_params: Mapping[str, Any] | None = params
        seen_urls: set[str] = set()

        while True:
            page = self._get_json(current_url, params=current_params)
            if page.get("resourceType") != "Bundle":
                raise ValueError("FHIR collection response must be a Bundle")
            yield page

            next_link = _next_page_link(page)
            if next_link is None:
                return
            current_url = self._resolve_server_url(next_link, current_url=current_url)
            if current_url in seen_urls:
                raise RuntimeError("FHIR pagination repeated a page")
            seen_urls.add(current_url)
            current_params = None

    def iter_resources(
        self,
        resource_type: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield resources from every page of a FHIR search Bundle."""

        for page in self.iter_bundle_pages(resource_type, params=params):
            entries = page.get("entry") or []
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                resource = entry.get("resource")
                if isinstance(resource, Mapping):
                    yield dict(resource)

    def put_resource(
        self,
        resource: Mapping[str, Any],
        *,
        write: bool = False,
    ) -> FHIRWriteResult:
        """PUT a resource only when the explicit ``write`` flag is true.

        A false flag performs no HTTP request and returns a summary with a
        ``None`` status code.  The payload is expected to have ``id`` and
        ``resourceType`` so the connector never guesses a write target.
        """

        resource_type, resource_id = _resource_identity(resource)
        url = self._resource_url(resource_type, resource_id)
        if not isinstance(write, bool):
            raise TypeError("write must be a bool")
        if not write:
            return FHIRWriteResult(
                resource_type=resource_type,
                resource_id=resource_id,
                url=url,
                status_code=None,
                written=False,
            )

        response = self._request("PUT", url, json=dict(resource))
        return FHIRWriteResult(
            resource_type=resource_type,
            resource_id=resource_id,
            url=url,
            status_code=getattr(response, "status_code", None),
            written=True,
        )

    def fetch_and_deidentify(
        self,
        resource_type: str,
        resource_id: str | None = None,
        *,
        params: Mapping[str, Any] | None = None,
        policy: Any = _DEFAULT_POLICY,
        method: str = _DEFAULT_METHOD,
        deidentifier: Deidentifier | None = None,
        write: bool = False,
    ) -> dict[str, Any]:
        """Fetch, de-identify, and optionally write a resource or Bundle.

        With ``resource_id`` set, the result is one resource fetched by id.
        Without it, the resource collection is read as a paginated search
        Bundle and returned as one merged, de-identified Bundle.  Every PUT
        is gated by the explicit ``write=True`` flag.
        """

        if not isinstance(write, bool):
            raise TypeError("write must be a bool")

        if resource_id is not None:
            source = self.get_resource(resource_type, resource_id)
            if source.get("resourceType") == "Bundle":
                transformed = deidentify_bundle(
                    source,
                    policy,
                    method=method,
                    deidentifier=deidentifier,
                )
            else:
                transformed = deidentify_resource(
                    source,
                    policy,
                    method=method,
                    deidentifier=deidentifier,
                )
            if write:
                self.put_resource(transformed, write=True)
            return transformed

        merged: dict[str, Any] | None = None
        for page in self.iter_bundle_pages(resource_type, params=params):
            transformed_page = deidentify_bundle(
                page,
                policy,
                method=method,
                deidentifier=deidentifier,
            )
            if write:
                for resource in _bundle_resources(transformed_page):
                    self.put_resource(resource, write=True)
            if merged is None:
                merged = transformed_page
                _remove_next_links(merged)
            else:
                entries = merged.setdefault("entry", [])
                if not isinstance(entries, list):
                    raise ValueError("FHIR Bundle entry must be a list")
                page_entries = transformed_page.get("entry") or []
                if isinstance(page_entries, list):
                    entries.extend(page_entries)

        if merged is None:  # pragma: no cover - iter_bundle_pages always yields
            raise ValueError("FHIR collection returned no Bundle")
        return merged

    deidentify = fetch_and_deidentify

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        response = self._client.request(method, url, **kwargs)
        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()
        elif getattr(response, "status_code", 200) >= 400:
            raise RuntimeError("FHIR server request failed")
        return response

    def _get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._request("GET", url, params=params)
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise ValueError("FHIR server response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("FHIR server response must be a JSON object")
        return payload

    def _resource_url(self, resource_type: str, resource_id: str) -> str:
        _validate_resource_type(resource_type)
        if not isinstance(resource_id, str) or not resource_id:
            raise ValueError("resource_id must be a non-empty string")
        return f"{self.config.base_url.rstrip('/')}/{quote(resource_type)}/{quote(resource_id, safe='')}"

    def _collection_url(self, resource_type: str) -> str:
        _validate_resource_type(resource_type)
        return f"{self.config.base_url.rstrip('/')}/{quote(resource_type)}"

    def _resolve_server_url(self, next_link: str, *, current_url: str) -> str:
        resolved = urljoin(current_url, next_link)
        _ensure_url_within_base(resolved, allowed_base=self.config.base_url)
        return resolved


def _validate_resource(resource: Any) -> None:
    if not isinstance(resource, Mapping):
        raise TypeError("resource must be a FHIR resource mapping")
    if not resource.get("resourceType"):
        raise ValueError("resource is missing 'resourceType'")


def _resource_identity(resource: Mapping[str, Any]) -> tuple[str, str]:
    _validate_resource(resource)
    resource_type = resource.get("resourceType")
    resource_id = resource.get("id")
    if not isinstance(resource_type, str) or not resource_type:
        raise ValueError("resource is missing 'resourceType'")
    if not isinstance(resource_id, str) or not resource_id:
        raise ValueError("resource is missing 'id'; write-back requires a resource id")
    _validate_resource_type(resource_type)
    return resource_type, resource_id


def _validate_resource_type(resource_type: Any) -> None:
    if not isinstance(resource_type, str) or not resource_type:
        raise ValueError("resource_type must be a non-empty FHIR resource name")
    if not resource_type[0].isalpha() or not resource_type.isalnum():
        raise ValueError("resource_type must be a FHIR resource name")


def _validate_base_url(value: str) -> None:
    parsed = urlsplit(str(value or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, query, or fragment")


def _ensure_url_within_base(url: str, *, allowed_base: str) -> None:
    candidate = urlsplit(url)
    allowed = urlsplit(allowed_base)
    if (candidate.scheme.lower(), candidate.netloc.lower()) != (
        allowed.scheme.lower(),
        allowed.netloc.lower(),
    ):
        raise ValueError("FHIR pagination URL changed origin")
    base_path = allowed.path.rstrip("/")
    if base_path and not (
        candidate.path == base_path or candidate.path.startswith(f"{base_path}/")
    ):
        raise ValueError("FHIR pagination URL escaped the configured base path")


def _next_page_link(bundle: Mapping[str, Any]) -> str | None:
    links = bundle.get("link") or []
    if not isinstance(links, list):
        return None
    for link in links:
        if not isinstance(link, Mapping):
            continue
        relation = str(link.get("relation") or link.get("rel") or "").lower()
        target = link.get("url")
        if relation == "next" and isinstance(target, str) and target:
            return target
    return None


def _remove_next_links(bundle: dict[str, Any]) -> None:
    links = bundle.get("link")
    if not isinstance(links, list):
        return
    bundle["link"] = [
        link
        for link in links
        if not isinstance(link, Mapping)
        or str(link.get("relation") or link.get("rel") or "").lower() != "next"
    ]


def _bundle_resources(bundle: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    entries = bundle.get("entry") or []
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        resource = entry.get("resource")
        if isinstance(resource, dict):
            yield resource


def _deidentify_bundle_in_place(
    bundle: dict[str, Any],
    deid: Callable[[str], str],
) -> None:
    text = bundle.get("text")
    if isinstance(text, dict) and "div" in text:
        transformed_div, changed = _deidentify_narrative(text["div"], deid)
        if changed:
            text["div"] = transformed_div

    entries = bundle.get("entry") or []
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource")
        if not isinstance(resource, dict) or not resource.get("resourceType"):
            continue
        if resource.get("resourceType") == "Bundle":
            _deidentify_bundle_in_place(resource, deid)
        else:
            _walk_resource(resource, str(resource["resourceType"]), deid)


def _walk_resource(
    node: Any,
    path: str,
    deid: Callable[[str], str],
) -> None:
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key in _SKIP_CONTAINERS:
                continue
            child_path = f"{path}.{key}"
            if key == "text" and isinstance(value, dict) and "div" in value:
                transformed_div, changed = _deidentify_narrative(value["div"], deid)
                if changed:
                    value["div"] = transformed_div
                continue
            if key == "data" and isinstance(value, str):
                transformed_data = _deidentify_attachment_data(node, value, deid)
                if transformed_data is not None:
                    node[key] = transformed_data
                continue
            if isinstance(value, str):
                if key in _SKIP_KEYS:
                    continue
                transformed = _deidentify_string(value, deid)
                if transformed != value:
                    node[key] = transformed
            elif isinstance(value, (dict, list)):
                _walk_resource(value, child_path, deid)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            child_path = f"{path}[{index}]"
            if isinstance(item, str):
                transformed = _deidentify_string(item, deid)
                if transformed != item:
                    node[index] = transformed
            elif isinstance(item, (dict, list)):
                _walk_resource(item, child_path, deid)


def _deidentify_string(value: str, deid: Callable[[str], str]) -> str:
    if not value.strip():
        return value
    return deid(value)


def _deidentify_attachment_data(
    attachment: Mapping[str, Any],
    encoded: str,
    deid: Callable[[str], str],
) -> str | None:
    content_type = attachment.get("contentType")
    if content_type is not None and not _is_text_media_type(content_type):
        return None
    try:
        normalized = "".join(encoded.split())
        decoded = base64.b64decode(normalized, validate=True).decode("utf-8")
    except (UnicodeDecodeError, ValueError, binascii.Error) as exc:
        if _is_text_media_type(content_type):
            raise ValueError("text attachment data is not valid UTF-8 base64") from exc
        return None

    if "html" in str(content_type).lower() or decoded.lstrip().startswith("<"):
        redacted, changed = _deidentify_narrative(decoded, deid)
        if not changed:
            return encoded
    else:
        redacted = deid(decoded)
        if redacted == decoded:
            return encoded
    return base64.b64encode(redacted.encode("utf-8")).decode("ascii")


def _is_text_media_type(content_type: Any) -> bool:
    if not isinstance(content_type, str) or not content_type.strip():
        return False
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type.startswith("text/") or media_type in _TEXT_MEDIA_TYPES


def _bind_text_deidentifier(
    deidentifier: Deidentifier | None,
    *,
    policy: Any,
    method: str,
) -> Callable[[str], str]:
    if deidentifier is None:
        from openmed.core.pii import deidentify

        deidentifier = deidentify
    assert deidentifier is not None

    def transform(text: str) -> str:
        kwargs: dict[str, Any] = {"method": method, "policy": policy}
        if method == "replace":
            kwargs["consistent"] = True
        try:
            result = deidentifier(text, **kwargs)
        except TypeError:
            kwargs.pop("consistent", None)
            result = deidentifier(text, **kwargs)
        if isinstance(result, str):
            return result
        transformed = getattr(result, "deidentified_text", None)
        if not isinstance(transformed, str):
            raise TypeError("deidentifier must return text or deidentified_text")
        return transformed

    return transform


def _deidentify_narrative(
    div: Any,
    deid: Callable[[str], str],
) -> tuple[Any, bool]:
    if not isinstance(div, str) or not div.strip():
        return div, False
    try:
        parser = _NarrativeRedactor(deid)
        parser.feed(div)
        parser.close()
    except Exception:
        redacted = deid(div)
        return redacted, redacted != div
    result = parser.result()
    return (result, parser.changed) if parser.changed else (div, False)


class _NarrativeRedactor(HTMLParser):
    """Rebuild XHTML while de-identifying visible text and safe attributes."""

    def __init__(self, deid: Callable[[str], str]) -> None:
        super().__init__(convert_charrefs=False)
        self._deid = deid
        self._parts: list[str] = []
        self._text_parts: list[tuple[int, str]] = []
        self.changed = False

    def result(self) -> str:
        self._redact_visible_text()
        return "".join(self._parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._parts.append(self._format_starttag(tag, attrs, self_closing=False))

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self._parts.append(self._format_starttag(tag, attrs, self_closing=True))

    def handle_endtag(self, tag: str) -> None:
        self._parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        index = len(self._parts)
        self._parts.append(data)
        if data.strip():
            self._text_parts.append((index, data))

    def handle_entityref(self, name: str) -> None:
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._parts.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self._parts.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self._parts.append(f"<!{decl}>")

    def _redact_visible_text(self) -> None:
        if not self._text_parts:
            return
        visible_text = "".join(text for _, text in self._text_parts)
        redacted = self._deid(visible_text)
        if redacted == visible_text:
            return
        first_index = self._text_parts[0][0]
        self._parts[first_index] = html.escape(redacted, quote=False)
        for index, _ in self._text_parts[1:]:
            self._parts[index] = ""
        self.changed = True

    def _format_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
        *,
        self_closing: bool,
    ) -> str:
        pieces = [tag]
        for name, value in attrs:
            if value is None:
                pieces.append(name)
            else:
                safe_value = self._deidentify_attribute(name, value)
                pieces.append(f'{name}="{html.escape(safe_value, quote=True)}"')
        inner = " ".join(pieces)
        return f"<{inner}/>" if self_closing else f"<{inner}>"

    def _deidentify_attribute(self, name: str, value: str) -> str:
        if name.lower() in {"xmlns", "id", "class", "style", "href", "src"}:
            return value
        if not value.strip():
            return value
        redacted = self._deid(value)
        if redacted != value:
            self.changed = True
        return redacted


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "FHIR server HTTP support requires the 'fhir' extra; "
            "install with `pip install openmed[fhir]`"
        ) from exc
    return httpx


__all__ = [
    "FHIRServerClient",
    "FHIRServerConfig",
    "FHIRWriteResult",
    "de_identify_bundle",
    "de_identify_resource",
    "deidentify_bundle",
    "deidentify_resource",
]
