"""WHO ICD-API snapshot building and offline ICD-11 MMS grounding.

Network access is deliberately confined to :class:`ICD11APIClient` and
:func:`build_snapshot`. Loading a snapshot, grounding mentions, and exporting
FHIR ``CodeableConcept`` values use local files only.
"""

from __future__ import annotations

import base64
import hashlib
import json
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
        self.release = _normalize_release(release)
        self.chapters = _normalize_chapters(chapters)
        self.language = _normalize_language(language)
        self.entities = tuple(sorted(entities, key=lambda item: (item.code, item.uri)))
        self.snapshot_sha256 = snapshot_sha256
        self._term_index = self._build_term_index()

    def _build_term_index(self) -> dict[str, ICD11Entity]:
        ranked: dict[str, tuple[int, str, str, ICD11Entity]] = {}
        for entity in self.entities:
            if not entity.uri or not entity.code or not entity.title:
                raise SnapshotIntegrityError(
                    "snapshot entities require non-empty uri, code, and title"
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
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.client_id = str(client_id).strip()
        self._client_secret = str(client_secret)
        self.language = _normalize_language(language)
        self.api_base_url = api_base_url.rstrip("/")
        self.token_endpoint = token_endpoint
        self.timeout = float(timeout)
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
        if not isinstance(token, str) or not token:
            raise ICD11APIError("WHO ICD-API token response omitted access_token")
        self._access_token = token
        expires_in = payload.get("expires_in", 3600)
        try:
            lifetime = max(1.0, float(expires_in))
        except (TypeError, ValueError):
            lifetime = 3600.0
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

        if not str(query).strip():
            raise ValueError("query must not be empty")
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
        opener = self._opener or urlrequest.urlopen
        try:
            with opener(request, timeout=self.timeout) as response:
                raw = response.read()
        except urlerror.HTTPError as exc:
            raise ICD11APIError(
                f"WHO ICD-API request failed with HTTP {exc.code}"
            ) from exc
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            raise ICD11APIError("WHO ICD-API request failed") from exc

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
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
    top_level_uris = _string_sequence(release_payload.get("child"))
    if not top_level_uris:
        raise ICD11APIError("WHO ICD-API release response omitted chapter URIs")

    payload_cache: dict[str, dict[str, Any]] = {}
    chapter_roots: dict[str, str] = {}
    for uri in sorted(top_level_uris):
        payload = client.get_entity(uri, release=release_id)
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
    visited: set[str] = set()
    while pending:
        uri = pending.pop()
        if uri in visited:
            continue
        visited.add(uri)
        payload = payload_cache.pop(uri, None)
        if payload is None:
            payload = client.get_entity(uri, release=release_id)

        entity = _entity_from_api_payload(payload)
        if entity is not None:
            entities[entity.uri] = entity

        children = sorted(_string_sequence(payload.get("child")), reverse=True)
        pending.extend(child for child in children if child not in visited)

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
) -> ICD11Snapshot:
    """Load and integrity-check a local snapshot without network access.

    Args:
        path: Snapshot JSON path.
        manifest_path: Optional sidecar manifest override.

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
        snapshot_bytes = snapshot_file.read_bytes()
        manifest_bytes = manifest_file.read_bytes()
    except OSError as exc:
        raise SnapshotIntegrityError(
            "snapshot and manifest must both be readable"
        ) from exc

    try:
        payload = json.loads(snapshot_bytes)
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise SnapshotIntegrityError(
            "snapshot or manifest contains invalid JSON"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(manifest, dict):
        raise SnapshotIntegrityError("snapshot and manifest must be JSON objects")

    actual_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    if manifest.get("snapshot_sha256") != actual_sha256:
        raise SnapshotIntegrityError("snapshot sha256 does not match its manifest")
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
    raw_entities = payload.get("entities")
    if not isinstance(raw_entities, list):
        raise SnapshotIntegrityError("snapshot entities must be a JSON array")
    if manifest.get("entity_count") != len(raw_entities):
        raise SnapshotIntegrityError(
            "snapshot entity count does not match its manifest"
        )

    entities = tuple(_entity_from_snapshot(item) for item in raw_entities)
    return ICD11Snapshot(
        release=str(payload.get("release") or ""),
        chapters=_string_sequence(payload.get("chapters")),
        language=str(payload.get("language") or ""),
        entities=entities,
        snapshot_sha256=actual_sha256,
    )


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
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
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


def _entity_from_api_payload(payload: Mapping[str, Any]) -> ICD11Entity | None:
    class_kind = str(payload.get("classKind") or "").strip().lower()
    if class_kind and class_kind != "category":
        return None
    uri = str(payload.get("@id") or "").strip()
    code = str(payload.get("code") or "").strip()
    title = _language_specific_text(payload.get("title"))
    if not uri or not code or not title:
        return None

    synonyms: set[str] = set()
    fully_specified_name = _language_specific_text(payload.get("fullySpecifiedName"))
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
            if label and _normalize_term(label) != _normalize_term(title):
                synonyms.add(label)

    return ICD11Entity(
        uri=uri,
        code=code,
        title=title,
        synonyms=tuple(
            sorted(synonyms, key=lambda value: (_normalize_term(value), value))
        ),
    )


def _entity_from_snapshot(payload: Any) -> ICD11Entity:
    if not isinstance(payload, Mapping):
        raise SnapshotIntegrityError("snapshot entity must be a JSON object")
    synonyms = payload.get("synonyms", [])
    if not isinstance(synonyms, list) or not all(
        isinstance(value, str) for value in synonyms
    ):
        raise SnapshotIntegrityError("snapshot entity synonyms must be strings")
    return ICD11Entity(
        uri=str(payload.get("uri") or "").strip(),
        code=str(payload.get("code") or "").strip(),
        title=str(payload.get("title") or "").strip(),
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
        or not parsed.path.startswith(expected_prefix)
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
