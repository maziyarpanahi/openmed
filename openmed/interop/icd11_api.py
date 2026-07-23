"""WHO ICD-API snapshot building and offline ICD-11 MMS grounding.

Network access is deliberately confined to :class:`ICD11APIClient` and
:func:`build_snapshot`. Loading a snapshot, grounding mentions, and exporting
FHIR ``CodeableConcept`` values use local files only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

ICD11_MMS_SYSTEM = "http://id.who.int/icd/release/11/mms"
ICD11_API_BASE_URL = "https://id.who.int"
ICD11_TOKEN_ENDPOINT = "https://icdaccessmanagement.who.int/connect/token"
ICD11_API_VERSION = "v2"
SNAPSHOT_SCHEMA_VERSION = 1
DEFAULT_CACHE_ENV = "OPENMED_ICD11_CACHE_DIR"
CLIENT_ID_ENV = "WHO_ICD_CLIENT_ID"
CLIENT_SECRET_ENV = "WHO_ICD_CLIENT_SECRET"
MAX_API_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_SNAPSHOT_ENTITIES = 100_000

_MAX_ENTITY_TEXT_CHARS = 4_096
_MAX_ENTITY_URI_CHARS = 2_048
_MAX_CODE_CHARS = 128
_MAX_ENTITY_SYNONYMS = 512
_MAX_QUERY_CHARS = 1_024
_MAX_TOKEN_CHARS = 16_384
_MAX_CHILDREN_PER_ENTITY = 100_000
_MAX_TOKEN_LIFETIME_SECONDS = 24 * 60 * 60

_RELEASE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}$")
_CHAPTER_RE = re.compile(r"^(?:0[1-9]|1[0-9]|2[0-6]|V|X)$")

__all__ = [
    "CLIENT_ID_ENV",
    "CLIENT_SECRET_ENV",
    "DEFAULT_CACHE_ENV",
    "ICD11APIClient",
    "ICD11APIError",
    "ICD11Entity",
    "ICD11Snapshot",
    "ICD11_MMS_SYSTEM",
    "MAX_API_RESPONSE_BYTES",
    "MAX_MANIFEST_BYTES",
    "MAX_SNAPSHOT_BYTES",
    "MAX_SNAPSHOT_ENTITIES",
    "SNAPSHOT_SCHEMA_VERSION",
    "SnapshotBuildResult",
    "SnapshotIntegrityError",
    "build_snapshot",
    "default_cache_dir",
    "ground_mention",
    "ground_to_codeable_concept",
    "load_snapshot",
    "snapshot_path",
]


class ICD11APIError(RuntimeError):
    """Raised when the WHO ICD-API cannot provide a valid response."""


class SnapshotIntegrityError(ValueError):
    """Raised when an ICD-11 snapshot or its manifest fails validation."""


class _RejectRedirectHandler(urlrequest.HTTPRedirectHandler):
    """Prevent credential-bearing requests from following redirects."""

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


def _open_without_redirects(
    request: urlrequest.Request,
    *,
    timeout: float,
) -> Any:
    opener = urlrequest.build_opener(_RejectRedirectHandler())
    return opener.open(request, timeout=timeout)


@dataclass(frozen=True)
class ICD11Entity:
    """One code-bearing ICD-11 MMS entity stored in a local snapshot."""

    uri: str
    code: str
    title: str
    synonyms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON representation of the entity."""

        return {
            "code": self.code,
            "synonyms": list(self.synonyms),
            "title": self.title,
            "uri": self.uri,
        }


class ICD11Snapshot:
    """Validated in-memory ICD-11 snapshot with an exact local term index.

    Args:
        release: Pinned WHO release id in ``YYYY-MM`` form.
        chapters: ICD-11 chapter codes represented by the snapshot.
        language: ISO language tag for titles and synonyms.
        entities: Code-bearing MMS entities to index.
        snapshot_sha256: Optional verified digest used in export provenance.
    """

    def __init__(
        self,
        *,
        release: str,
        chapters: Sequence[str],
        language: str,
        entities: Sequence[ICD11Entity],
        snapshot_sha256: str = "",
    ) -> None:
        entity_items = tuple(entities)
        if len(entity_items) > MAX_SNAPSHOT_ENTITIES:
            raise SnapshotIntegrityError(
                f"snapshot exceeds {MAX_SNAPSHOT_ENTITIES} entities"
            )
        if not all(isinstance(entity, ICD11Entity) for entity in entity_items):
            raise SnapshotIntegrityError(
                "snapshot entities must be ICD11Entity instances"
            )
        self.release = _normalize_release(release)
        self.chapters = _normalize_chapters(chapters)
        self.language = _normalize_language(language)
        self.entities = tuple(
            sorted(entity_items, key=lambda item: (item.code, item.uri))
        )
        if snapshot_sha256 and re.fullmatch(r"[0-9a-f]{64}", snapshot_sha256) is None:
            raise SnapshotIntegrityError(
                "snapshot sha256 must be a lowercase 64-character digest"
            )
        self.snapshot_sha256 = snapshot_sha256
        self._term_index = self._build_term_index()

    def _build_term_index(self) -> dict[str, ICD11Entity]:
        ranked: dict[str, tuple[int, str, str, ICD11Entity]] = {}
        for entity in self.entities:
            if not entity.uri or not entity.code or not entity.title:
                raise SnapshotIntegrityError(
                    "snapshot entities require non-empty uri, code, and title"
                )
            if len(entity.uri) > _MAX_ENTITY_URI_CHARS:
                raise SnapshotIntegrityError(
                    "snapshot entity URI exceeds the safety limit"
                )
            if len(entity.code) > _MAX_CODE_CHARS:
                raise SnapshotIntegrityError(
                    "snapshot entity code exceeds the safety limit"
                )
            if len(entity.title) > _MAX_ENTITY_TEXT_CHARS:
                raise SnapshotIntegrityError(
                    "snapshot entity title exceeds the safety limit"
                )
            if len(entity.synonyms) > _MAX_ENTITY_SYNONYMS or any(
                len(synonym) > _MAX_ENTITY_TEXT_CHARS for synonym in entity.synonyms
            ):
                raise SnapshotIntegrityError(
                    "snapshot entity terms exceed the safety limit"
                )
            for rank, term in ((0, entity.title),):
                _add_ranked_term(ranked, term, rank, entity)
            for synonym in entity.synonyms:
                _add_ranked_term(ranked, synonym, 1, entity)
        return {term: item[-1] for term, item in ranked.items()}

    def ground(self, mention: str) -> ICD11Entity | None:
        """Return the deterministic exact title/synonym match for ``mention``.

        Args:
            mention: Extracted condition or diagnosis surface form.

        Returns:
            The matching snapshot entity, or ``None`` when no exact normalized
            title or synonym exists.
        """

        if not isinstance(mention, str):
            raise TypeError("mention must be a string")
        if len(mention) > _MAX_ENTITY_TEXT_CHARS:
            return None
        normalized = _normalize_term(mention)
        if not normalized:
            return None
        return self._term_index.get(normalized)

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical snapshot payload."""

        return {
            "chapters": list(self.chapters),
            "entities": [entity.to_dict() for entity in self.entities],
            "language": self.language,
            "linearization": "mms",
            "release": self.release,
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
        }


@dataclass(frozen=True)
class SnapshotBuildResult:
    """Paths and integrity metadata emitted by :func:`build_snapshot`."""

    snapshot_path: Path
    manifest_path: Path
    snapshot_sha256: str
    entity_count: int


class ICD11APIClient:
    """Minimal WHO ICD-API v2 client used only during snapshot builds.

    The client implements the OAuth2 client-credentials token exchange plus
    pinned-release MMS search, release-index, and entity lookup requests.
    It has no role in runtime grounding.

    Args:
        client_id: Client id issued by the WHO ICD-API portal.
        client_secret: Matching client secret. It is kept in memory only.
        language: ISO language tag requested from the API.
        api_base_url: WHO ICD-API base URL.
        token_endpoint: WHO OAuth2 token endpoint.
        timeout: Per-request timeout in seconds.
        opener: Optional urllib-compatible opener for testing.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        language: str = "en",
        api_base_url: str = ICD11_API_BASE_URL,
        token_endpoint: str = ICD11_TOKEN_ENDPOINT,
        timeout: float = 30.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if (
            not isinstance(client_id, str)
            or not client_id.strip()
            or not isinstance(client_secret, str)
            or not client_secret
        ):
            raise ValueError("WHO ICD-API client id and client secret are required")
        if isinstance(timeout, bool):
            raise TypeError("timeout must be a finite number")
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError) as exc:
            raise TypeError("timeout must be a finite number") from exc
        if not math.isfinite(timeout_value) or timeout_value <= 0:
            raise ValueError("timeout must be a finite number greater than zero")

        self.client_id = str(client_id).strip()
        self._client_secret = str(client_secret)
        self.language = _normalize_language(language)
        self.api_base_url = _validated_service_url(
            api_base_url,
            label="api_base_url",
            require_root_path=True,
        )
        self.token_endpoint = _validated_service_url(
            token_endpoint,
            label="token_endpoint",
            require_root_path=False,
        )
        self.timeout = timeout_value
        self._opener = opener
        self._access_token: str | None = None
        self._token_expires_at = 0.0

    def access_token(self) -> str:
        """Fetch and cache an OAuth2 client-credentials access token."""

        if self._access_token is not None and time.monotonic() < self._token_expires_at:
            return self._access_token

        credentials = f"{self.client_id}:{self._client_secret}".encode("utf-8")
        authorization = base64.b64encode(credentials).decode("ascii")
        body = urlparse.urlencode(
            {"grant_type": "client_credentials", "scope": "icdapi_access"}
        ).encode("ascii")
        request = urlrequest.Request(
            self.token_endpoint,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Authorization": f"Basic {authorization}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        payload = self._request_json(request)
        token = payload.get("access_token")
        if not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_CHARS:
            raise ICD11APIError("WHO ICD-API token response omitted access_token")
        self._access_token = token
        expires_in = payload.get("expires_in", 3600)
        try:
            lifetime = float(expires_in)
        except (TypeError, ValueError):
            lifetime = 3600.0
        if not math.isfinite(lifetime) or lifetime <= 0:
            lifetime = 3600.0
        lifetime = min(lifetime, _MAX_TOKEN_LIFETIME_SECONDS)
        self._token_expires_at = time.monotonic() + max(1.0, lifetime - 60.0)
        return token

    def release_index(self, release: str) -> dict[str, Any]:
        """Return the pinned MMS release root containing chapter URIs."""

        release_id = _normalize_release(release)
        url = f"{self.api_base_url}/icd/release/11/{release_id}/mms"
        return self._get_json(url)

    def search(
        self,
        query: str,
        *,
        release: str,
        chapters: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Search a pinned MMS release, optionally within chapter codes."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if len(query) > _MAX_QUERY_CHARS:
            raise ValueError(f"query must not exceed {_MAX_QUERY_CHARS} characters")
        release_id = _normalize_release(release)
        parameters: dict[str, str] = {
            "flatResults": "true",
            "highlightingEnabled": "false",
            "medicalCodingMode": "true",
            "q": str(query).strip(),
        }
        if chapters:
            parameters["chapterFilter"] = ";".join(_normalize_chapters(chapters))
        query_string = urlparse.urlencode(parameters)
        url = (
            f"{self.api_base_url}/icd/release/11/{release_id}/mms/search?{query_string}"
        )
        return self._get_json(url)

    def lookup(self, foundation_uri: str, *, release: str) -> dict[str, Any]:
        """Map one WHO Foundation URI into the pinned MMS release."""

        release_id = _normalize_release(release)
        normalized_uri = _validated_foundation_uri(foundation_uri)
        query_string = urlparse.urlencode({"foundationUri": normalized_uri})
        url = (
            f"{self.api_base_url}/icd/release/11/{release_id}/mms/lookup?{query_string}"
        )
        return self._get_json(url)

    def get_entity(self, entity_uri: str, *, release: str) -> dict[str, Any]:
        """Fetch one code or grouping entity from the pinned MMS release."""

        release_id = _normalize_release(release)
        url = _validated_entity_url(entity_uri, release_id, self.api_base_url)
        return self._get_json(url)

    def _get_json(self, url: str) -> dict[str, Any]:
        request = urlrequest.Request(
            url,
            method="GET",
            headers={
                "API-Version": ICD11_API_VERSION,
                "Accept": "application/json",
                "Accept-Language": self.language,
                "Authorization": f"Bearer {self.access_token()}",
            },
        )
        return self._request_json(request)

    def _request_json(self, request: urlrequest.Request) -> dict[str, Any]:
        opener = self._opener or _open_without_redirects
        try:
            with opener(request, timeout=self.timeout) as response:
                raw = response.read(MAX_API_RESPONSE_BYTES + 1)
        except urlerror.HTTPError as exc:
            raise ICD11APIError(
                f"WHO ICD-API request failed with HTTP {exc.code}"
            ) from exc
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            raise ICD11APIError("WHO ICD-API request failed") from exc

        if len(raw) > MAX_API_RESPONSE_BYTES:
            raise ICD11APIError(
                f"WHO ICD-API response exceeds {MAX_API_RESPONSE_BYTES} bytes"
            )

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ICD11APIError("WHO ICD-API returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ICD11APIError("WHO ICD-API returned a non-object JSON response")
        return payload


def default_cache_dir() -> Path:
    """Return the configurable local ICD-11 snapshot cache directory.

    Returns:
        ``OPENMED_ICD11_CACHE_DIR`` when configured, otherwise the user's
        platform-neutral OpenMed cache path.
    """

    configured = os.environ.get(DEFAULT_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "openmed" / "icd11"


def snapshot_path(
    *,
    release: str,
    chapters: Sequence[str],
    language: str = "en",
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the deterministic path for a release/chapter snapshot.

    Args:
        release: Pinned WHO release id in ``YYYY-MM`` form.
        chapters: One or more ICD-11 chapter codes.
        language: Snapshot language tag.
        cache_dir: Optional cache root override.

    Returns:
        The deterministic snapshot JSON path.
    """

    release_id = _normalize_release(release)
    chapter_ids = _normalize_chapters(chapters)
    language_id = _normalize_language(language)
    chapter_slug = "-".join(chapter.lower() for chapter in chapter_ids)
    filename = f"icd11-mms-{release_id}-{language_id}-{chapter_slug}.json"
    root = default_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
    return root / filename


def build_snapshot(
    client: ICD11APIClient,
    *,
    release: str,
    chapters: Sequence[str],
    cache_dir: str | Path | None = None,
    language: str | None = None,
) -> SnapshotBuildResult:
    """Build a deterministic local snapshot by traversing selected chapters.

    This is the only high-level path that performs HTTP requests. Runtime
    callers should use :func:`load_snapshot` and :func:`ground_mention`.

    Args:
        client: Authenticated WHO ICD-API client configuration.
        release: Pinned WHO release id in ``YYYY-MM`` form.
        chapters: One or more ICD-11 chapter codes to traverse.
        cache_dir: Optional output directory override.
        language: Optional language assertion; must match the client.

    Returns:
        Snapshot and manifest paths plus integrity metadata.

    Raises:
        ICD11APIError: If the release or requested chapters cannot be fetched.
        ValueError: If a release, chapter, or language argument is invalid.
    """

    release_id = _normalize_release(release)
    chapter_ids = _normalize_chapters(chapters)
    language_id = _normalize_language(language or client.language)
    if language_id != client.language:
        raise ValueError("snapshot language must match the ICD11APIClient language")

    release_payload = client.release_index(release_id)
    top_level_uris = _validated_entity_links(
        release_payload.get("child"),
        release=release_id,
        api_base_url=client.api_base_url,
        label="release response",
    )
    if not top_level_uris:
        raise ICD11APIError("WHO ICD-API release response omitted chapter URIs")

    payload_cache: dict[str, dict[str, Any]] = {}
    chapter_roots: dict[str, str] = {}
    for uri in top_level_uris:
        payload = client.get_entity(uri, release=release_id)
        _validate_response_entity_uri(
            payload,
            requested_url=uri,
            release=release_id,
            api_base_url=client.api_base_url,
        )
        payload_cache[uri] = payload
        chapter_code = str(payload.get("code") or "").strip().upper()
        if chapter_code in chapter_ids:
            chapter_roots[chapter_code] = uri

    missing = sorted(set(chapter_ids) - set(chapter_roots))
    if missing:
        raise ICD11APIError(
            "WHO ICD-API release does not contain requested chapters: "
            + ", ".join(missing)
        )

    entities: dict[str, ICD11Entity] = {}
    pending = [chapter_roots[chapter] for chapter in reversed(chapter_ids)]
    queued = set(pending)
    visited: set[str] = set()
    while pending:
        uri = pending.pop()
        queued.discard(uri)
        if uri in visited:
            continue
        if len(visited) >= MAX_SNAPSHOT_ENTITIES:
            raise ICD11APIError(
                f"snapshot traversal exceeds {MAX_SNAPSHOT_ENTITIES} entities"
            )
        visited.add(uri)
        payload = payload_cache.pop(uri, None)
        if payload is None:
            payload = client.get_entity(uri, release=release_id)
        _validate_response_entity_uri(
            payload,
            requested_url=uri,
            release=release_id,
            api_base_url=client.api_base_url,
        )

        entity = _entity_from_api_payload(payload)
        if entity is not None:
            entities[uri] = entity

        children = _validated_entity_links(
            payload.get("child"),
            release=release_id,
            api_base_url=client.api_base_url,
            label="entity response",
        )
        for child in reversed(children):
            if child not in visited and child not in queued:
                pending.append(child)
                queued.add(child)
        if len(visited) + len(queued) > MAX_SNAPSHOT_ENTITIES:
            raise ICD11APIError(
                f"snapshot traversal exceeds {MAX_SNAPSHOT_ENTITIES} entities"
            )

    if not entities:
        raise ICD11APIError("requested ICD-11 chapters contained no coded entities")

    snapshot = ICD11Snapshot(
        release=release_id,
        chapters=chapter_ids,
        language=language_id,
        entities=tuple(entities.values()),
    )
    return _write_snapshot(snapshot, cache_dir=cache_dir)


def load_snapshot(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    expected_sha256: str | None = None,
) -> ICD11Snapshot:
    """Load and integrity-check a local snapshot without network access.

    Args:
        path: Snapshot JSON path.
        manifest_path: Optional sidecar manifest override.
        expected_sha256: Optional trusted digest pin. When supplied, the
            snapshot must match it in addition to its sidecar manifest.

    Returns:
        A verified, locally indexed snapshot.

    Raises:
        SnapshotIntegrityError: If either file is missing, malformed, or does
            not match its recorded integrity metadata.
    """

    snapshot_file = Path(path)
    manifest_file = (
        Path(manifest_path)
        if manifest_path is not None
        else snapshot_file.with_suffix(".manifest.json")
    )
    try:
        snapshot_bytes = _read_limited_file(
            snapshot_file,
            max_bytes=MAX_SNAPSHOT_BYTES,
            label="snapshot",
        )
        manifest_bytes = _read_limited_file(
            manifest_file,
            max_bytes=MAX_MANIFEST_BYTES,
            label="manifest",
        )
    except SnapshotIntegrityError:
        raise
    except OSError as exc:
        raise SnapshotIntegrityError(
            "snapshot and manifest must both be readable"
        ) from exc

    try:
        payload = json.loads(snapshot_bytes)
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SnapshotIntegrityError(
            "snapshot or manifest contains invalid JSON"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(manifest, dict):
        raise SnapshotIntegrityError("snapshot and manifest must be JSON objects")

    actual_sha256 = _snapshot_sha256(snapshot_bytes)
    if manifest.get("snapshot_sha256") != actual_sha256:
        raise SnapshotIntegrityError("snapshot sha256 does not match its manifest")
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise ValueError("expected_sha256 must be a lowercase 64-character digest")
        if actual_sha256 != expected_sha256:
            raise SnapshotIntegrityError("snapshot sha256 does not match trusted pin")
    if manifest.get("snapshot_file") != snapshot_file.name:
        raise SnapshotIntegrityError("manifest snapshot filename does not match")
    if (
        payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
    ):
        raise SnapshotIntegrityError("unsupported ICD-11 snapshot schema version")
    if payload.get("linearization") != "mms" or manifest.get("linearization") != "mms":
        raise SnapshotIntegrityError("snapshot linearization must be ICD-11 MMS")

    for key in ("release", "language", "chapters"):
        if payload.get(key) != manifest.get(key):
            raise SnapshotIntegrityError(f"snapshot and manifest disagree on {key}")
    if not isinstance(payload.get("release"), str) or not isinstance(
        payload.get("language"), str
    ):
        raise SnapshotIntegrityError("snapshot release and language must be strings")
    raw_chapters = payload.get("chapters")
    if not isinstance(raw_chapters, list) or not all(
        isinstance(chapter, str) for chapter in raw_chapters
    ):
        raise SnapshotIntegrityError("snapshot chapters must be strings")
    raw_entities = payload.get("entities")
    if not isinstance(raw_entities, list):
        raise SnapshotIntegrityError("snapshot entities must be a JSON array")
    entity_count = manifest.get("entity_count")
    if (
        isinstance(entity_count, bool)
        or not isinstance(entity_count, int)
        or entity_count != len(raw_entities)
    ):
        raise SnapshotIntegrityError(
            "snapshot entity count does not match its manifest"
        )
    if len(raw_entities) > MAX_SNAPSHOT_ENTITIES:
        raise SnapshotIntegrityError(
            f"snapshot exceeds {MAX_SNAPSHOT_ENTITIES} entities"
        )

    try:
        release_id = _normalize_release(payload["release"])
        chapter_ids = _normalize_chapters(raw_chapters)
        language_id = _normalize_language(payload["language"])
        entities = tuple(
            _entity_from_snapshot(item, release=release_id) for item in raw_entities
        )
        return ICD11Snapshot(
            release=release_id,
            chapters=chapter_ids,
            language=language_id,
            entities=entities,
            snapshot_sha256=actual_sha256,
        )
    except SnapshotIntegrityError:
        raise
    except (TypeError, ValueError) as exc:
        raise SnapshotIntegrityError("snapshot metadata is invalid") from exc


def ground_mention(mention: str, snapshot: ICD11Snapshot) -> ICD11Entity | None:
    """Ground one diagnosis mention using only the loaded snapshot index.

    Args:
        mention: Extracted condition or diagnosis surface form.
        snapshot: Previously loaded local snapshot.

    Returns:
        The matching entity, or ``None``.
    """

    return snapshot.ground(mention)


def ground_to_codeable_concept(
    mention: str,
    snapshot: ICD11Snapshot,
) -> dict[str, Any] | None:
    """Ground ``mention`` and return a provenance-stamped FHIR concept.

    The canonical snapshot title is used for ``CodeableConcept.text``. Raw
    note context and the caller's surface form are never copied into output.

    Args:
        mention: Extracted condition or diagnosis surface form.
        snapshot: Previously loaded local snapshot.

    Returns:
        A provenance-stamped FHIR ``CodeableConcept``, or ``None`` when the
        mention does not match a snapshot title or synonym.
    """

    entity = ground_mention(mention, snapshot)
    if entity is None:
        return None

    from openmed.clinical.exporters.code_provenance import (
        stamp_coding_provenance,
    )
    from openmed.clinical.exporters.codeable_concept import (
        GroundedSpan,
        to_codeable_concept,
    )
    from openmed.clinical.grounding import Candidate

    grounded_span = GroundedSpan(
        text=entity.title,
        start=0,
        end=len(entity.title),
        candidates=(
            Candidate(
                system="ICD11",
                code=entity.code,
                display=entity.title,
                score=1.0,
                source_language=snapshot.language,
            ),
        ),
    )
    concept = to_codeable_concept(grounded_span)
    digest_label = (
        f"ICD-11 MMS snapshot sha256:{snapshot.snapshot_sha256}"
        if snapshot.snapshot_sha256
        else "ICD-11 MMS local snapshot"
    )
    concept["coding"] = [
        stamp_coding_provenance(
            coding,
            {ICD11_MMS_SYSTEM: snapshot.release},
            source_label=digest_label,
        )
        for coding in concept["coding"]
    ]
    return concept


def _write_snapshot(
    snapshot: ICD11Snapshot,
    *,
    cache_dir: str | Path | None,
) -> SnapshotBuildResult:
    output_path = snapshot_path(
        release=snapshot.release,
        chapters=snapshot.chapters,
        language=snapshot.language,
        cache_dir=cache_dir,
    )
    snapshot_bytes = _canonical_json_bytes(snapshot.to_payload())
    if len(snapshot_bytes) > MAX_SNAPSHOT_BYTES:
        raise ICD11APIError(f"generated snapshot exceeds {MAX_SNAPSHOT_BYTES} bytes")
    snapshot_sha256 = _snapshot_sha256(snapshot_bytes)
    manifest = {
        "chapters": list(snapshot.chapters),
        "entity_count": len(snapshot.entities),
        "language": snapshot.language,
        "license": "CC BY-ND 3.0 IGO",
        "linearization": "mms",
        "release": snapshot.release,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_file": output_path.name,
        "snapshot_sha256": snapshot_sha256,
        "source": "WHO ICD-API",
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_path, snapshot_bytes)
    _atomic_write(manifest_path, _canonical_json_bytes(manifest))
    return SnapshotBuildResult(
        snapshot_path=output_path,
        manifest_path=manifest_path,
        snapshot_sha256=snapshot_sha256,
        entity_count=len(snapshot.entities),
    )


def _atomic_write(path: Path, content: bytes) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


def _read_limited_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    with path.open("rb") as handle:
        content = handle.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise SnapshotIntegrityError(f"{label} exceeds {max_bytes} bytes")
    return content


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _snapshot_sha256(content: bytes) -> str:
    """Hash canonical snapshot JSON independent of checkout line endings."""

    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def _entity_from_api_payload(payload: Mapping[str, Any]) -> ICD11Entity | None:
    class_kind = str(payload.get("classKind") or "").strip().lower()
    if class_kind and class_kind != "category":
        return None
    uri = str(payload.get("@id") or "").strip()
    code = str(payload.get("code") or "").strip()
    title = _language_specific_text(payload.get("title"))
    if not uri or not code or not title:
        return None
    if len(uri) > _MAX_ENTITY_URI_CHARS:
        raise ICD11APIError("WHO ICD-API entity URI exceeds the safety limit")
    if len(code) > _MAX_CODE_CHARS:
        raise ICD11APIError("WHO ICD-API entity code exceeds the safety limit")
    if len(title) > _MAX_ENTITY_TEXT_CHARS:
        raise ICD11APIError("WHO ICD-API entity title exceeds the safety limit")

    synonyms: set[str] = set()
    fully_specified_name = _language_specific_text(payload.get("fullySpecifiedName"))
    if len(fully_specified_name) > _MAX_ENTITY_TEXT_CHARS:
        raise ICD11APIError("WHO ICD-API entity term exceeds the safety limit")
    if fully_specified_name and _normalize_term(
        fully_specified_name
    ) != _normalize_term(title):
        synonyms.add(fully_specified_name)
    for field in ("indexTerm", "inclusion"):
        raw_terms = payload.get(field)
        if not isinstance(raw_terms, list):
            continue
        for term in raw_terms:
            if not isinstance(term, Mapping) or term.get("deprecated") is True:
                continue
            label = _language_specific_text(term.get("label"))
            if len(label) > _MAX_ENTITY_TEXT_CHARS:
                raise ICD11APIError("WHO ICD-API entity term exceeds the safety limit")
            if label and _normalize_term(label) != _normalize_term(title):
                synonyms.add(label)
                if len(synonyms) > _MAX_ENTITY_SYNONYMS:
                    raise ICD11APIError(
                        "WHO ICD-API entity exceeds the synonym safety limit"
                    )

    return ICD11Entity(
        uri=uri,
        code=code,
        title=title,
        synonyms=tuple(
            sorted(synonyms, key=lambda value: (_normalize_term(value), value))
        ),
    )


def _entity_from_snapshot(payload: Any, *, release: str) -> ICD11Entity:
    if not isinstance(payload, Mapping):
        raise SnapshotIntegrityError("snapshot entity must be a JSON object")
    raw_uri = payload.get("uri")
    raw_code = payload.get("code")
    raw_title = payload.get("title")
    if not all(isinstance(value, str) for value in (raw_uri, raw_code, raw_title)):
        raise SnapshotIntegrityError("snapshot entity fields must be strings")
    uri = raw_uri.strip()
    code = raw_code.strip()
    title = raw_title.strip()
    if not uri or not code or not title:
        raise SnapshotIntegrityError(
            "snapshot entities require non-empty uri, code, and title"
        )
    if len(uri) > _MAX_ENTITY_URI_CHARS:
        raise SnapshotIntegrityError("snapshot entity URI exceeds the safety limit")
    if len(code) > _MAX_CODE_CHARS:
        raise SnapshotIntegrityError("snapshot entity code exceeds the safety limit")
    if len(title) > _MAX_ENTITY_TEXT_CHARS:
        raise SnapshotIntegrityError("snapshot entity title exceeds the safety limit")
    try:
        _validated_entity_url(uri, release, ICD11_API_BASE_URL)
    except ValueError as exc:
        raise SnapshotIntegrityError(
            "snapshot entity URI is outside the pinned WHO MMS release"
        ) from exc
    synonyms = payload.get("synonyms", [])
    if not isinstance(synonyms, list) or not all(
        isinstance(value, str) for value in synonyms
    ):
        raise SnapshotIntegrityError("snapshot entity synonyms must be strings")
    if len(synonyms) > _MAX_ENTITY_SYNONYMS:
        raise SnapshotIntegrityError("snapshot entity exceeds the synonym safety limit")
    if any(len(value) > _MAX_ENTITY_TEXT_CHARS for value in synonyms):
        raise SnapshotIntegrityError("snapshot entity term exceeds the safety limit")
    return ICD11Entity(
        uri=uri,
        code=code,
        title=title,
        synonyms=tuple(synonyms),
    )


def _language_specific_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        text = value.get("@value")
        if isinstance(text, str):
            return text.strip()
    return ""


def _validated_service_url(
    value: str,
    *,
    label: str,
    require_root_path: bool,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a URL string")
    parsed = urlparse.urlparse(value.strip())
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} has an invalid port") from exc
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError(f"{label} must use HTTPS or loopback HTTP")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be an absolute URL without credentials")
    if require_root_path and parsed.path not in {"", "/"}:
        raise ValueError(f"{label} must not include a path")
    if not require_root_path and not parsed.path.startswith("/"):
        raise ValueError(f"{label} must include an absolute path")

    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if port is not None:
        netloc = f"{netloc}:{port}"
    path = "" if require_root_path else parsed.path
    return urlparse.urlunparse((parsed.scheme, netloc, path, "", "", ""))


def _validated_entity_links(
    value: Any,
    *,
    release: str,
    api_base_url: str,
    label: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ICD11APIError(f"WHO ICD-API {label} contains invalid entity links")
    if len(value) > _MAX_CHILDREN_PER_ENTITY:
        raise ICD11APIError(f"WHO ICD-API {label} exceeds the child-link safety limit")
    try:
        links = {_validated_entity_url(item, release, api_base_url) for item in value}
    except ValueError as exc:
        raise ICD11APIError(
            f"WHO ICD-API {label} contains an out-of-scope entity link"
        ) from exc
    return tuple(sorted(links))


def _validate_response_entity_uri(
    payload: Mapping[str, Any],
    *,
    requested_url: str,
    release: str,
    api_base_url: str,
) -> None:
    entity_uri = payload.get("@id")
    if not isinstance(entity_uri, str) or not entity_uri:
        raise ICD11APIError("WHO ICD-API entity response omitted @id")
    try:
        normalized = _validated_entity_url(entity_uri, release, api_base_url)
    except ValueError as exc:
        raise ICD11APIError(
            "WHO ICD-API entity response contains an out-of-scope @id"
        ) from exc
    if normalized != requested_url:
        raise ICD11APIError("WHO ICD-API entity response @id does not match request")


def _validated_entity_url(
    entity_uri: str,
    release: str,
    api_base_url: str,
) -> str:
    parsed = urlparse.urlparse(str(entity_uri))
    expected_base = urlparse.urlparse(api_base_url)
    expected_prefix = f"/icd/release/11/{release}/mms/"
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != expected_base.hostname
        or re.fullmatch(rf"{re.escape(expected_prefix)}[0-9]+", parsed.path) is None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("entity URI is outside the pinned WHO MMS release")
    return urlparse.urlunparse(
        (
            expected_base.scheme or "https",
            expected_base.netloc,
            parsed.path,
            "",
            "",
            "",
        )
    )


def _validated_foundation_uri(foundation_uri: str) -> str:
    parsed = urlparse.urlparse(str(foundation_uri))
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != "id.who.int"
        or not re.fullmatch(r"/icd/entity/[0-9]+", parsed.path)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("foundation URI must identify one WHO ICD entity")
    return urlparse.urlunparse(("http", "id.who.int", parsed.path, "", "", ""))


def _normalize_release(release: str) -> str:
    normalized = str(release).strip()
    if not _RELEASE_RE.fullmatch(normalized):
        raise ValueError("release must be pinned as YYYY-MM")
    month = int(normalized[-2:])
    if month < 1 or month > 12:
        raise ValueError("release month must be between 01 and 12")
    return normalized


def _normalize_chapters(chapters: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(
        sorted(
            {
                str(chapter).strip().upper()
                for chapter in chapters
                if str(chapter).strip()
            }
        )
    )
    if not normalized:
        raise ValueError("at least one ICD-11 chapter code is required")
    invalid = [chapter for chapter in normalized if not _CHAPTER_RE.fullmatch(chapter)]
    if invalid:
        raise ValueError(
            "chapter codes must be two digits or one uppercase letter: "
            + ", ".join(invalid)
        )
    return normalized


def _normalize_language(language: str) -> str:
    normalized = str(language).strip().lower()
    if not re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]{2,8})?", normalized):
        raise ValueError("language must be an ISO language tag such as en or pt-br")
    return normalized


def _normalize_term(term: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(term)).casefold()
    characters = [character if character.isalnum() else " " for character in normalized]
    return " ".join("".join(characters).split())


def _add_ranked_term(
    index: dict[str, tuple[int, str, str, ICD11Entity]],
    term: str,
    rank: int,
    entity: ICD11Entity,
) -> None:
    normalized = _normalize_term(term)
    if not normalized:
        return
    candidate = (rank, entity.code, entity.uri, entity)
    current = index.get(normalized)
    if current is None or candidate[:3] < current[:3]:
        index[normalized] = candidate


def _string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)
