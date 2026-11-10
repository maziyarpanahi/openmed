"""Local-first OpenMRS REST and FHIR2 de-identification adapter.

The adapter pulls resources from a facility-local OpenMRS installation,
de-identifies text on the same machine, and only then writes or exports the
result. Importing this module does not import an HTTP client or load a privacy
model; both are resolved when they are first used.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Literal
from urllib.parse import urljoin, urlsplit

from openmed.clinical.exporters.fhir import (
    OperationOutcomeIssue,
    to_bundle,
    to_operation_outcome,
)
from openmed.interop.fhir_bulk import NDJSONFileSummary, deidentify_ndjson_stream
from openmed.interop.fhir_operations import (
    Deidentifier,
    de_identify_resource_with_manifest,
)

APIKind = Literal["rest", "fhir2"]
Sleep = Callable[[float], None]

_REST_RESOURCES: Final[frozenset[str]] = frozenset({"obs", "encounter", "patient"})
_FHIR_RESOURCES: Final[dict[str, str]] = {
    "observation": "Observation",
    "encounter": "Encounter",
    "patient": "Patient",
}
_REST_TEXT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "value",
        "comment",
        "comments",
        "note",
        "notes",
        "encounternote",
        "encounternotes",
    }
)
_REST_TEXT_CONTAINER_KEYS: Final[frozenset[str]] = _REST_TEXT_KEYS - {"value"}
_RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset(
    {408, 425, 429, 500, 502, 503, 504}
)
_DEFAULT_POLICY: Final = "hipaa_safe_harbor"
_DEFAULT_METHOD: Final = "replace"


@dataclass(frozen=True)
class OpenMRSConfig:
    """Connection and retry settings for one OpenMRS installation.

    ``base_url`` should include the installation context path when present,
    for example ``http://kenyaemr.local/openmrs``. ``destination_url`` can
    point at a separate facility-controlled server; when omitted, write-back
    targets the source installation.
    """

    base_url: str
    destination_url: str | None = None
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    session_token: str | None = field(default=None, repr=False)
    session_cookie_name: str = "JSESSIONID"
    timeout: float = 30.0
    page_size: int = 100
    max_retries: int = 3
    backoff_factor: float = 0.5
    verify_tls: bool = True
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _validate_base_url(self.base_url, name="base_url")
        if self.destination_url is not None:
            _validate_base_url(self.destination_url, name="destination_url")
        if (self.username is None) != (self.password is None):
            raise ValueError("username and password must be provided together")
        if self.session_token and self.username:
            raise ValueError("use either basic authentication or a session token")
        if self.session_token and any(
            character in self.session_token for character in "\r\n;"
        ):
            raise ValueError("session_token contains invalid cookie characters")
        if not self.session_cookie_name or any(
            character in self.session_cookie_name for character in "\r\n;="
        ):
            raise ValueError("session_cookie_name is invalid")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.page_size <= 0:
            raise ValueError("page_size must be greater than zero")
        if self.max_retries < 0:
            raise ValueError("max_retries must be zero or greater")
        if self.backoff_factor < 0:
            raise ValueError("backoff_factor must be zero or greater")


@dataclass(frozen=True)
class DeidentifiedResource:
    """One de-identified OpenMRS resource and its PHI-free change manifest."""

    api: APIKind
    resource_name: str
    resource: dict[str, Any] = field(repr=False)
    manifest: dict[str, Any]
    source_id: str | None = None

    @property
    def transformed_paths(self) -> tuple[str, ...]:
        """Return transformed element paths recorded by the manifest."""

        return manifest_paths(self.manifest)


@dataclass(frozen=True)
class WriteResult:
    """Result of one OpenMRS write-back request or dry run."""

    method: str
    url: str
    status_code: int | None
    dry_run: bool
    response: dict[str, Any] | None = field(default=None, repr=False)


class OpenMRSClient:
    """Synchronous OpenMRS REST/FHIR2 client with pagination and retries.

    Args:
        config: Source, destination, authentication, and retry settings.
        client: Optional HTTPX-compatible client, primarily for offline tests.
        sleep: Retry delay callable.
    """

    def __init__(
        self,
        config: OpenMRSConfig,
        *,
        client: Any | None = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        self.config = config
        self._httpx = _import_httpx()
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or self._httpx.Client(
            timeout=config.timeout,
            verify=config.verify_tls,
        )
        self._auth = None
        if config.username is not None and config.password is not None:
            self._auth = self._httpx.BasicAuth(config.username, config.password)

    def close(self) -> None:
        """Close the internally created HTTP client."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenMRSClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def iter_rest_resources(
        self,
        resource_name: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield all pages from an OpenMRS legacy REST collection.

        Supported collections are ``obs``, ``encounter``, and ``patient``.
        Pagination follows a response ``next`` link when present and otherwise
        advances with OpenMRS's ``startIndex``/``limit`` convention.

        Args:
            resource_name: Legacy collection name.
            params: Additional OpenMRS search parameters.

        Yields:
            Deep-copied JSON resources in server order.

        Raises:
            ValueError: If the resource name, page URL, or response is invalid.
            RuntimeError: If the server repeats a page link.
        """

        name = _normalize_rest_resource(resource_name)
        url = _join_base(self.config.base_url, f"ws/rest/v1/{name}")
        query: dict[str, Any] | None = {
            "v": "full",
            "limit": self.config.page_size,
            **dict(params or {}),
        }
        seen_pages: set[str] = set()

        while True:
            signature = _page_signature(url, query)
            if signature in seen_pages:
                raise RuntimeError("OpenMRS REST pagination repeated a page")
            seen_pages.add(signature)

            page = self._get_json(url, params=query, allowed_base=self.config.base_url)
            results = page.get("results")
            if not isinstance(results, list):
                raise ValueError(
                    "OpenMRS REST collection response lacks a results list"
                )
            for resource in results:
                if not isinstance(resource, dict):
                    raise ValueError("OpenMRS REST results must be JSON objects")
                yield copy.deepcopy(resource)

            next_link = _next_page_link(page, link_key="links")
            if next_link:
                url = _resolve_page_url(
                    next_link,
                    current_url=url,
                    allowed_base=self.config.base_url,
                )
                query = None
                continue

            limit = _positive_int((query or {}).get("limit"))
            if not results or limit is None or len(results) < limit:
                break
            start_index = _nonnegative_int((query or {}).get("startIndex")) or 0
            query = dict(query or {})
            query["startIndex"] = start_index + len(results)

    def pull_rest(
        self,
        resource_name: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return a materialized legacy REST collection.

        Args:
            resource_name: Legacy collection name.
            params: Additional OpenMRS search parameters.

        Returns:
            Resources in server order.
        """

        return tuple(self.iter_rest_resources(resource_name, params=params))

    def iter_fhir_resources(
        self,
        resource_name: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield resources from all pages of a FHIR2 R4 search Bundle.

        Args:
            resource_name: FHIR2 resource type.
            params: Additional FHIR search parameters.

        Yields:
            Deep-copied FHIR resources in server order.

        Raises:
            ValueError: If the resource name, page URL, or Bundle is invalid.
            RuntimeError: If the server repeats a page link.
        """

        name = _normalize_fhir_resource(resource_name)
        url = _join_base(self.config.base_url, f"ws/fhir2/R4/{name}")
        query: dict[str, Any] | None = {
            "_count": self.config.page_size,
            **dict(params or {}),
        }
        seen_pages: set[str] = set()

        while True:
            signature = _page_signature(url, query)
            if signature in seen_pages:
                raise RuntimeError("OpenMRS FHIR2 pagination repeated a page")
            seen_pages.add(signature)

            bundle = self._get_json(
                url,
                params=query,
                allowed_base=self.config.base_url,
            )
            if bundle.get("resourceType") != "Bundle":
                raise ValueError("OpenMRS FHIR2 search response must be a Bundle")
            entries = bundle.get("entry") or []
            if not isinstance(entries, list):
                raise ValueError("OpenMRS FHIR2 Bundle.entry must be a list")
            for entry in entries:
                resource = entry.get("resource") if isinstance(entry, dict) else None
                if not isinstance(resource, dict):
                    raise ValueError("OpenMRS FHIR2 entries must contain resources")
                yield copy.deepcopy(resource)

            next_link = _next_page_link(bundle, link_key="link")
            if not next_link:
                break
            url = _resolve_page_url(
                next_link,
                current_url=url,
                allowed_base=self.config.base_url,
            )
            query = None

    def pull_fhir(
        self,
        resource_name: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return a materialized FHIR2 R4 search result.

        Args:
            resource_name: FHIR2 resource type.
            params: Additional FHIR search parameters.

        Returns:
            Resources in server order.
        """

        return tuple(self.iter_fhir_resources(resource_name, params=params))

    def write_rest_resource(
        self,
        resource_name: str,
        resource: Mapping[str, Any],
        *,
        dry_run: bool = False,
        obs_comment: bool = False,
    ) -> WriteResult:
        """POST a de-identified REST resource to the destination server.

        OpenMRS legacy REST uses ``POST /resource`` for creates and
        ``POST /resource/{uuid}`` for partial updates. With ``obs_comment``,
        only the de-identified string ``value`` (or existing ``comment``) is
        written to the source obs's comment field.

        Args:
            resource_name: Legacy resource type.
            resource: De-identified resource mapping.
            dry_run: Return the planned request without sending it.
            obs_comment: Send only a de-identified obs comment.

        Returns:
            The completed or planned request summary.

        Raises:
            ValueError: If the resource or obs-comment request is invalid.
        """

        name = _normalize_rest_resource(resource_name)
        payload = copy.deepcopy(dict(resource))
        resource_id = _optional_string(payload.get("uuid"))
        if obs_comment:
            if name != "obs":
                raise ValueError("obs_comment convention is only valid for obs")
            if resource_id is None:
                raise ValueError("obs_comment write-back requires an obs uuid")
            comment = payload.get("value")
            if not isinstance(comment, str):
                comment = payload.get("comment")
            if not isinstance(comment, str) or not comment.strip():
                raise ValueError("obs_comment write-back requires string text")
            payload = {"comment": comment}

        path = f"ws/rest/v1/{name}"
        if resource_id:
            path = f"{path}/{resource_id}"
        url = _join_base(self._destination_base, path)
        return self._write("POST", url, payload, dry_run=dry_run)

    def write_fhir_resource(
        self,
        resource: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> WriteResult:
        """PUT or POST one de-identified FHIR2 resource.

        Args:
            resource: De-identified Patient, Encounter, or Observation.
            dry_run: Return the planned request without sending it.

        Returns:
            The completed or planned request summary.

        Raises:
            ValueError: If the resource type is unsupported.
        """

        payload = copy.deepcopy(dict(resource))
        resource_name = _normalize_fhir_resource(str(payload.get("resourceType") or ""))
        resource_id = _optional_string(payload.get("id"))
        path = f"ws/fhir2/R4/{resource_name}"
        method = "POST"
        if resource_id:
            path = f"{path}/{resource_id}"
            method = "PUT"
        url = _join_base(self._destination_base, path)
        return self._write(method, url, payload, dry_run=dry_run)

    @property
    def _destination_base(self) -> str:
        return self.config.destination_url or self.config.base_url

    def _write(
        self,
        method: str,
        url: str,
        payload: dict[str, Any],
        *,
        dry_run: bool,
    ) -> WriteResult:
        if dry_run:
            return WriteResult(
                method=method,
                url=url,
                status_code=None,
                dry_run=True,
            )
        response = self._request(
            method,
            url,
            json_payload=payload,
            allowed_base=self._destination_base,
        )
        decoded = _decode_optional_json(response)
        return WriteResult(
            method=method,
            url=url,
            status_code=int(response.status_code),
            dry_run=False,
            response=decoded,
        )

    def _get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        allowed_base: str,
    ) -> dict[str, Any]:
        response = self._request(
            "GET",
            url,
            params=params,
            allowed_base=allowed_base,
        )
        decoded = _decode_optional_json(response)
        if decoded is None:
            raise ValueError("OpenMRS response body must be a JSON object")
        return decoded

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_payload: Mapping[str, Any] | None = None,
        allowed_base: str,
    ) -> Any:
        _ensure_url_within_base(url, allowed_base=allowed_base)
        headers = {
            "Accept": "application/fhir+json, application/json",
            **{str(key): str(value) for key, value in self.config.headers.items()},
        }
        if json_payload is not None:
            headers["Content-Type"] = "application/json"
        if self.config.session_token:
            headers["Cookie"] = (
                f"{self.config.session_cookie_name}={self.config.session_token}"
            )

        request_kwargs: dict[str, Any] = {"headers": headers}
        if params is not None:
            request_kwargs["params"] = dict(params)
        if json_payload is not None:
            request_kwargs["json"] = dict(json_payload)
        if self._auth is not None:
            request_kwargs["auth"] = self._auth

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.request(method, url, **request_kwargs)
            except self._httpx.RequestError:
                if attempt >= self.config.max_retries:
                    raise
                self._sleep(self._backoff_seconds(attempt, None))
                continue

            if (
                int(response.status_code) in _RETRYABLE_STATUS_CODES
                and attempt < self.config.max_retries
            ):
                retry_after = response.headers.get("Retry-After")
                response.close()
                self._sleep(self._backoff_seconds(attempt, retry_after))
                continue
            response.raise_for_status()
            return response

        raise RuntimeError("OpenMRS request retry loop exited unexpectedly")

    def _backoff_seconds(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 60.0)
            except ValueError:
                pass
        return min(self.config.backoff_factor * (2**attempt), 60.0)


class OpenMRSAdapter:
    """Pull, de-identify, write back, and export OpenMRS resources.

    Args:
        client: Configured facility-local OpenMRS client.
        policy: Privacy policy passed to the OpenMed pipeline.
        method: De-identification method.
        deidentifier: Optional pipeline override, mainly for offline tests.
    """

    def __init__(
        self,
        client: OpenMRSClient,
        *,
        policy: str = _DEFAULT_POLICY,
        method: str = _DEFAULT_METHOD,
        deidentifier: Deidentifier | None = None,
    ) -> None:
        self.client = client
        self.policy = policy
        self.method = method
        self.deidentifier = deidentifier

    def pull_rest(
        self,
        resource_name: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> tuple[DeidentifiedResource, ...]:
        """Pull and de-identify one legacy REST resource collection.

        Args:
            resource_name: ``obs``, ``encounter``, or ``patient``.
            params: Additional OpenMRS search parameters.

        Returns:
            De-identified records with OperationOutcome-style manifests.
        """

        name = _normalize_rest_resource(resource_name)
        return tuple(
            self._transform_rest(name, resource)
            for resource in self.client.iter_rest_resources(name, params=params)
        )

    def pull_fhir(
        self,
        resource_name: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> tuple[DeidentifiedResource, ...]:
        """Pull and de-identify one FHIR2 R4 resource collection.

        Args:
            resource_name: Patient, Encounter, or Observation.
            params: Additional FHIR search parameters.

        Returns:
            De-identified records with OperationOutcome-style manifests.
        """

        name = _normalize_fhir_resource(resource_name)
        records: list[DeidentifiedResource] = []
        for resource in self.client.iter_fhir_resources(name, params=params):
            transformed, manifest = de_identify_resource_with_manifest(
                resource,
                policy=self.policy,
                method=self.method,
                deidentifier=self.deidentifier,
            )
            records.append(
                DeidentifiedResource(
                    api="fhir2",
                    resource_name=name,
                    source_id=_optional_string(resource.get("id")),
                    resource=transformed,
                    manifest=manifest,
                )
            )
        return tuple(records)

    def write_back(
        self,
        records: Iterable[DeidentifiedResource],
        *,
        dry_run: bool = False,
        obs_comment: bool = False,
    ) -> tuple[WriteResult, ...]:
        """Write de-identified records to the configured destination.

        Args:
            records: Records returned by this adapter.
            dry_run: Plan requests without sending them.
            obs_comment: Use the legacy REST obs-comment convention.

        Returns:
            One request summary per input record.

        Raises:
            ValueError: If a record cannot be safely written by this path.
        """

        results: list[WriteResult] = []
        for record in records:
            if record.api == "rest":
                if record.resource_name == "patient":
                    raise ValueError(
                        "legacy REST Patient write-back is not supported; "
                        "use the FHIR2 Patient path so identifiers are de-identified"
                    )
                results.append(
                    self.client.write_rest_resource(
                        record.resource_name,
                        record.resource,
                        dry_run=dry_run,
                        obs_comment=obs_comment,
                    )
                )
            elif record.api == "fhir2":
                if obs_comment:
                    raise ValueError(
                        "obs_comment convention is only valid for REST obs"
                    )
                results.append(
                    self.client.write_fhir_resource(
                        record.resource,
                        dry_run=dry_run,
                    )
                )
            else:
                raise ValueError(f"unsupported OpenMRS API kind: {record.api!r}")
        return tuple(results)

    def export_bundle(
        self,
        records: Iterable[DeidentifiedResource],
        *,
        doc_id: str = "openmrs-handoff",
    ) -> dict[str, Any]:
        """Emit de-identified FHIR2 records as a transaction Bundle.

        Args:
            records: De-identified FHIR2 records.
            doc_id: Non-PHI stable seed for deterministic full URLs.

        Returns:
            A FHIR R4 transaction Bundle.

        Raises:
            ValueError: If a record is not FHIR2 or resource IDs collide.
        """

        resources = _fhir_resources(records)
        return to_bundle(resources, doc_id=doc_id, bundle_type="transaction")

    def export_ndjson(
        self,
        records: Iterable[DeidentifiedResource],
        out_path: str | Path,
    ) -> NDJSONFileSummary:
        """Emit records through the existing FHIR bulk NDJSON path.

        Args:
            records: De-identified FHIR2 records.
            out_path: Destination NDJSON file.

        Returns:
            The PHI-safe streaming export summary.

        Raises:
            ValueError: If a record is not a FHIR2 resource.
        """

        resources = _fhir_resources(records)
        destination = Path(out_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = (
            json.dumps(resource, ensure_ascii=False, separators=(",", ":")) + "\n"
            for resource in resources
        )
        with destination.open("w", encoding="utf-8") as output:
            return deidentify_ndjson_stream(
                lines,
                output,
                source="openmrs-fhir2-memory",
                destination=destination,
                deidentifier=_identity_deidentifier,
            )

    def _transform_rest(
        self,
        resource_name: str,
        resource: dict[str, Any],
    ) -> DeidentifiedResource:
        transformed, manifest = de_identify_rest_resource_with_manifest(
            resource,
            resource_name=resource_name,
            policy=self.policy,
            method=self.method,
            deidentifier=self.deidentifier,
        )
        return DeidentifiedResource(
            api="rest",
            resource_name=resource_name,
            source_id=_optional_string(resource.get("uuid")),
            resource=transformed,
            manifest=manifest,
        )


def de_identify_rest_resource(
    resource: Mapping[str, Any],
    *,
    resource_name: str = "resource",
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
) -> dict[str, Any]:
    """Return a copy with only REST text value/comment/note fields changed.

    Args:
        resource: Legacy OpenMRS REST resource.
        resource_name: Name used as the manifest path root.
        policy: Privacy policy passed to the pipeline.
        method: De-identification method.
        deidentifier: Optional pipeline override, mainly for offline tests.

    Returns:
        A deep-copied resource with eligible text transformed.
    """

    transformed, _ = de_identify_rest_resource_with_manifest(
        resource,
        resource_name=resource_name,
        policy=policy,
        method=method,
        deidentifier=deidentifier,
    )
    return transformed


def de_identify_rest_resource_with_manifest(
    resource: Mapping[str, Any],
    *,
    resource_name: str = "resource",
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a de-identified REST resource and an ``OperationOutcome``.

    Only string values under ``value``, comment, or note keys are transformed.
    Coded concepts, UUIDs, references, dates, numeric observations, and every
    other structural field are copied byte-for-byte.

    Args:
        resource: Legacy OpenMRS REST resource.
        resource_name: Name used as the manifest path root.
        policy: Privacy policy passed to the pipeline.
        method: De-identification method.
        deidentifier: Optional pipeline override, mainly for offline tests.

    Returns:
        A ``(resource, outcome)`` pair.
    """

    if not isinstance(resource, Mapping):
        raise TypeError("resource must be an OpenMRS REST mapping")
    transformed = copy.deepcopy(dict(resource))
    changes: list[str] = []
    deid = _bind_text_deidentifier(
        deidentifier,
        policy=policy,
        method=method,
    )
    root = f"OpenMRS.{str(resource_name or 'resource').strip()}"
    _walk_rest_text(transformed, root, changes, deid)
    return transformed, _outcome_from_paths(changes)


def manifest_paths(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract transformed element paths from an OperationOutcome manifest.

    Args:
        manifest: FHIR OperationOutcome-style mapping.

    Returns:
        Expressions in issue order.
    """

    paths: list[str] = []
    issues = manifest.get("issue") or []
    if not isinstance(issues, list):
        return ()
    for issue in issues:
        expressions = issue.get("expression") if isinstance(issue, dict) else None
        if not isinstance(expressions, list):
            continue
        paths.extend(
            expression for expression in expressions if isinstance(expression, str)
        )
    return tuple(paths)


def ensure_registered() -> None:
    """Registry hook; the OpenMRS adapter has no eager side effects."""


def _walk_rest_text(
    node: Any,
    path: str,
    changes: list[str],
    deid: Callable[[str], str],
    *,
    text_container: bool = False,
) -> None:
    if isinstance(node, dict):
        for key, value in list(node.items()):
            child_path = f"{path}.{key}"
            normalized_key = str(key).casefold()
            is_text = normalized_key in _REST_TEXT_KEYS
            if isinstance(value, str):
                is_nested_text = text_container and normalized_key == "text"
                if (is_text or is_nested_text) and value.strip():
                    transformed = deid(value)
                    if transformed != value:
                        node[key] = transformed
                        changes.append(child_path)
            elif isinstance(value, (dict, list)):
                _walk_rest_text(
                    value,
                    child_path,
                    changes,
                    deid,
                    text_container=(
                        text_container or normalized_key in _REST_TEXT_CONTAINER_KEYS
                    ),
                )
    elif isinstance(node, list):
        for index, value in enumerate(node):
            child_path = f"{path}[{index}]"
            if isinstance(value, str):
                if text_container and value.strip():
                    transformed = deid(value)
                    if transformed != value:
                        node[index] = transformed
                        changes.append(child_path)
            elif isinstance(value, (dict, list)):
                _walk_rest_text(
                    value,
                    child_path,
                    changes,
                    deid,
                    text_container=text_container,
                )


def _bind_text_deidentifier(
    deidentifier: Deidentifier | None,
    *,
    policy: str,
    method: str,
) -> Callable[[str], str]:
    if deidentifier is None:
        from openmed.core.pii import deidentify as deidentifier

    def transform(text: str) -> str:
        kwargs: dict[str, Any] = {"method": method, "policy": policy}
        if method == "replace":
            kwargs["consistent"] = True
        try:
            result = deidentifier(text, **kwargs)
        except TypeError:
            kwargs.pop("consistent", None)
            result = deidentifier(text, **kwargs)
        transformed = getattr(result, "deidentified_text", None)
        if not isinstance(transformed, str):
            raise TypeError("deidentifier must return an object with deidentified_text")
        return transformed

    return transform


def _outcome_from_paths(paths: Sequence[str]) -> dict[str, Any]:
    return to_operation_outcome(
        [
            OperationOutcomeIssue(
                severity="information",
                code="informational",
                diagnostics="De-identified element.",
                expression=path,
            )
            for path in paths
        ]
    )


def _fhir_resources(
    records: Iterable[DeidentifiedResource],
) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    for record in records:
        if record.api != "fhir2":
            raise ValueError("FHIR export accepts only FHIR2 resources")
        resources.append(copy.deepcopy(record.resource))
    return resources


@dataclass(frozen=True)
class _IdentityResult:
    deidentified_text: str


def _identity_deidentifier(text: str, **_: Any) -> _IdentityResult:
    return _IdentityResult(deidentified_text=text)


def _normalize_rest_resource(resource_name: str) -> str:
    name = str(resource_name or "").strip().lower()
    if name not in _REST_RESOURCES:
        allowed = ", ".join(sorted(_REST_RESOURCES))
        raise ValueError(
            f"unsupported OpenMRS REST resource {resource_name!r}: {allowed}"
        )
    return name


def _normalize_fhir_resource(resource_name: str) -> str:
    key = str(resource_name or "").strip().lower()
    try:
        return _FHIR_RESOURCES[key]
    except KeyError as exc:
        allowed = ", ".join(_FHIR_RESOURCES.values())
        raise ValueError(
            f"unsupported OpenMRS FHIR2 resource {resource_name!r}: {allowed}"
        ) from exc


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "OpenMRS HTTP support requires the 'openmrs' extra; "
            "install with `pip install openmed[openmrs]`"
        ) from exc
    return httpx


def _validate_base_url(value: str, *, name: str) -> None:
    parsed = urlsplit(str(value or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not contain credentials, query, or fragment")


def _join_base(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _ensure_url_within_base(url: str, *, allowed_base: str) -> None:
    candidate = urlsplit(url)
    allowed = urlsplit(allowed_base)
    if (candidate.scheme.lower(), candidate.netloc.lower()) != (
        allowed.scheme.lower(),
        allowed.netloc.lower(),
    ):
        raise ValueError("OpenMRS pagination/write URL changed origin")
    base_path = allowed.path.rstrip("/")
    if base_path and not (
        candidate.path == base_path or candidate.path.startswith(f"{base_path}/")
    ):
        raise ValueError("OpenMRS URL escaped the configured installation path")


def _resolve_page_url(
    next_link: str,
    *,
    current_url: str,
    allowed_base: str,
) -> str:
    resolved = urljoin(current_url, next_link)
    _ensure_url_within_base(resolved, allowed_base=allowed_base)
    return resolved


def _next_page_link(payload: Mapping[str, Any], *, link_key: str) -> str | None:
    links = payload.get(link_key) or []
    if not isinstance(links, list):
        return None
    for link in links:
        if not isinstance(link, Mapping):
            continue
        relation = str(link.get("rel") or link.get("relation") or "").lower()
        if relation != "next":
            continue
        target = link.get("url") or link.get("uri")
        if isinstance(target, str) and target:
            return target
    return None


def _decode_optional_json(response: Any) -> dict[str, Any] | None:
    if not response.content:
        return None
    decoded = response.json()
    if not isinstance(decoded, dict):
        raise ValueError("OpenMRS response body must be a JSON object")
    return decoded


def _page_signature(url: str, params: Mapping[str, Any] | None) -> str:
    query = json.dumps(dict(params or {}), sort_keys=True, default=str)
    return f"{url}\n{query}"


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "APIKind",
    "DeidentifiedResource",
    "OpenMRSAdapter",
    "OpenMRSClient",
    "OpenMRSConfig",
    "WriteResult",
    "de_identify_rest_resource",
    "de_identify_rest_resource_with_manifest",
    "manifest_paths",
]
