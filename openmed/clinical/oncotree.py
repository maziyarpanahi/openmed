"""Map free-text tumor types to OncoTree codes from a caller-supplied release.

No OncoTree payload is bundled or downloaded. Callers supply a local JSON
snapshot (path or ``OPENMED_ONCOTREE_PATH``) and a release version
(``version=...`` or ``OPENMED_ONCOTREE_VERSION``); OncoTree JSON does not
carry a version field. The snapshot must be a flat JSON list of tumor-type
node objects (the OncoTree ``tumorTypes`` / API list shape). Callers may add an
optional ``synonyms`` string list to nodes in their local snapshot; this is an
OpenMed extension, not a field supplied by the OncoTree API. Nested tree dumps
(root object with recursive ``children``) are unsupported. Matching is
deterministic exact / normalized name, synonym, and code lookup (including
``history`` / ``revocations`` former codes). History and revocation aliases
that collide with a still-live code in the same release are not indexed, so
current codes win over former-code aliases. Outputs are assistive only -- see
:data:`ONCOTREE_ADVISORY`.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict

from openmed.clinical.normalization.backend import normalize_surface

ONCOTREE_ADVISORY = (
    "OncoTree tumor-type mapping is deterministic assistive terminology "
    "grounding against a user-supplied OncoTree release snapshot. OpenMed does "
    "not bundle, download, or refresh OncoTree data; every mapping is scoped to "
    "the loaded release version stamped in oncotree_version and may change "
    "across releases. Matching uses exact and normalized name/code lookup, "
    "plus optional caller-supplied synonyms, and indexes history and revocation "
    "code aliases from that snapshot except "
    "when an alias collides with a still-live code (current codes win). "
    "Ambiguous or unmatched mentions are returned unmapped with a reason "
    "rather than guessed. The mapper does not predict tumor type, derive "
    "histology, stage disease, recommend treatment, or cross-map to other "
    "ontologies (ICD-O, SNOMED CT, UMLS). Output assists review and "
    "interoperability and is not a diagnosis or substitute for pathologist or "
    "oncologist judgment."
)

_ENV_PATH = "OPENMED_ONCOTREE_PATH"
_ENV_VERSION = "OPENMED_ONCOTREE_VERSION"

_CONFIDENCE_EXACT = 1.0
_CONFIDENCE_NORMALIZED = 0.95
_CONFIDENCE_UNMAPPED = 0.0


class OncoTreeMapping(TypedDict):
    """A tumor-type mention mapped to an OncoTree code, or explicitly unmapped.

    ``code``/``name``/``main_type``/``tissue`` are filled on a unique hit and
    ``None`` when unmapped. ``oncotree_version`` is always the loaded release
    version. ``match_confidence`` is ``1.0`` (exact), ``0.95`` (normalized),
    or ``0.0`` (unmapped). ``reason`` is set only for unmapped results.
    """

    code: Optional[str]
    name: Optional[str]
    main_type: Optional[str]
    tissue: Optional[str]
    oncotree_version: str
    match_confidence: float
    reason: Optional[str]
    advisory: str


@dataclass(frozen=True)
class OncoTreeNode:
    """One OncoTree tumor-type node from a user-supplied release."""

    code: str
    name: str
    main_type: str
    tissue: str
    synonyms: tuple[str, ...] = ()
    history: tuple[str, ...] = ()
    revocations: tuple[str, ...] = ()


@dataclass(frozen=True)
class OncoTreeRelease:
    """Indexed OncoTree release ready for deterministic mention mapping."""

    version: str
    nodes_by_code: Mapping[str, OncoTreeNode]
    _exact_name: Mapping[str, tuple[str, ...]] = field(repr=False)
    _exact_code: Mapping[str, tuple[str, ...]] = field(repr=False)
    _normalized_name: Mapping[str, tuple[str, ...]] = field(repr=False)
    _normalized_code: Mapping[str, tuple[str, ...]] = field(repr=False)

    @property
    def node_count(self) -> int:
        """Number of distinct OncoTree codes in the release."""

        return len(self.nodes_by_code)


def load_oncotree(
    path: str | Path | None = None,
    *,
    version: str | None = None,
) -> OncoTreeRelease:
    """Load and index a user-supplied OncoTree JSON release.

    Expects a flat JSON list of tumor-type node objects. Nested OncoTree tree
    dumps (a root object with recursive ``children``) are unsupported. OncoTree
    JSON never carries a version field; callers must supply the release version
    via ``version=...`` or ``OPENMED_ONCOTREE_VERSION``.

    Args:
        path: Path to the release file. When omitted, reads
            ``OPENMED_ONCOTREE_PATH``.
        version: Release version stamp for provenance. When omitted, reads
            ``OPENMED_ONCOTREE_VERSION``.

    Returns:
        An indexed :class:`OncoTreeRelease`.

    Raises:
        FileNotFoundError: When the release path is missing or unset.
        ValueError: When the payload is malformed (including nested tree
            dumps), empty, or version is missing.
    """

    resolved_path = _resolve_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"OncoTree release file not found: {resolved_path}. "
            f"Pass path=... or set {_ENV_PATH}."
        )

    with resolved_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(
            "OncoTree release must be a flat JSON list of tumor-type node "
            "objects; nested tree dumps (root object with children) are "
            "unsupported"
        )

    resolved_version = _resolve_version(version)
    nodes = [_coerce_node(item, index=index) for index, item in enumerate(payload)]
    if not nodes:
        raise ValueError(f"OncoTree release at {resolved_path} contains no nodes")

    return _build_release(resolved_version, nodes)


def map_tumor_type(mention: str, release: OncoTreeRelease) -> OncoTreeMapping:
    """Map a tumor-type mention to an OncoTree code from a loaded release.

    Args:
        mention: Free-text tumor-type string.
        release: Indexed release from :func:`load_oncotree` (carries
            ``oncotree_version`` provenance).

    Returns:
        An :class:`OncoTreeMapping` with version provenance on every result.
        Exact and normalized name/synonym/code tiers only; unmatched or
        ambiguous mentions stay unmapped with a reason.
    """

    stripped = mention.strip()
    if not stripped:
        return _unmapped(release.version, reason="empty_mention")

    normalized = normalize_surface(stripped)

    # (index, lookup_key, confidence)
    tiers: list[tuple[Mapping[str, tuple[str, ...]], str, float]] = [
        (release._exact_code, stripped, _CONFIDENCE_EXACT),
        (release._exact_name, stripped, _CONFIDENCE_EXACT),
        (release._normalized_code, normalized, _CONFIDENCE_NORMALIZED),
        (release._normalized_name, normalized, _CONFIDENCE_NORMALIZED),
    ]

    for index, key, confidence in tiers:
        if not key:
            continue
        codes = index.get(key)
        if not codes:
            continue
        unique = tuple(dict.fromkeys(codes))
        if len(unique) > 1:
            return _unmapped(release.version, reason="ambiguous")
        return _mapped(release.nodes_by_code[unique[0]], release.version, confidence)

    return _unmapped(release.version, reason="no_match")


def _resolve_path(path: str | Path | None) -> Path:
    if path is None:
        env_path = os.getenv(_ENV_PATH)
        if not env_path or not str(env_path).strip():
            raise FileNotFoundError(
                f"OncoTree release path not provided. Pass path=... or set {_ENV_PATH}."
            )
        return Path(env_path).expanduser()
    return Path(path).expanduser()


def _resolve_version(explicit: str | None) -> str:
    if explicit is not None:
        if not explicit.strip():
            raise ValueError("version must be a non-empty string when provided")
        return explicit.strip()
    env_version = os.getenv(_ENV_VERSION)
    if env_version and env_version.strip():
        return env_version.strip()
    raise ValueError(
        "OncoTree release version is required. Provide version=... or set "
        f"{_ENV_VERSION}. OncoTree JSON does not carry a version field."
    )


def _coerce_node(item: Any, *, index: int) -> OncoTreeNode:
    if not isinstance(item, Mapping):
        raise ValueError(f"OncoTree node at index {index} must be an object")
    code = item.get("code")
    name = item.get("name")
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"OncoTree node at index {index} requires a non-empty 'code'")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"OncoTree node at index {index} requires a non-empty 'name'")
    main_type = item.get("mainType", "")
    tissue = item.get("tissue", "")
    if main_type is None:
        main_type = ""
    if tissue is None:
        tissue = ""
    if not isinstance(main_type, str):
        raise ValueError(f"OncoTree node {code!r} mainType must be a string")
    if not isinstance(tissue, str):
        raise ValueError(f"OncoTree node {code!r} tissue must be a string")
    code = code.strip()
    return OncoTreeNode(
        code=code,
        name=name.strip(),
        main_type=main_type.strip(),
        tissue=tissue.strip(),
        synonyms=_string_list(item.get("synonyms"), field="synonyms", code=code),
        history=_string_list(item.get("history"), field="history", code=code),
        revocations=_string_list(
            item.get("revocations"), field="revocations", code=code
        ),
    )


def _string_list(raw: Any, *, field: str, code: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"OncoTree node {code!r} '{field}' must be a list of strings")
    collected: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str):
            raise ValueError(
                f"OncoTree node {code!r} '{field}' entries must be strings, "
                f"got {type(value).__name__}"
            )
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            collected.append(text)
    return tuple(collected)


def _build_release(version: str, nodes: Iterable[OncoTreeNode]) -> OncoTreeRelease:
    nodes_by_code: dict[str, OncoTreeNode] = {}
    exact_name: dict[str, list[str]] = defaultdict(list)
    exact_code: dict[str, list[str]] = defaultdict(list)
    normalized_name: dict[str, list[str]] = defaultdict(list)
    normalized_code: dict[str, list[str]] = defaultdict(list)

    node_list = list(nodes)
    for node in node_list:
        if node.code in nodes_by_code:
            raise ValueError(f"Duplicate OncoTree code in release: {node.code!r}")
        nodes_by_code[node.code] = node

    # Live codes are indexed first. History/revocation aliases that equal a
    # still-live code are skipped so map_tumor_type(live_code) resolves to the
    # current node instead of ambiguous.
    for node in node_list:
        _index_surface(exact_name, normalized_name, node.name, node.code)
        for synonym in node.synonyms:
            _index_surface(exact_name, normalized_name, synonym, node.code)
        _index_code_alias(exact_code, normalized_code, node.code, target=node.code)
        for alias in (*node.history, *node.revocations):
            if alias in nodes_by_code:
                continue
            _index_code_alias(exact_code, normalized_code, alias, target=node.code)

    return OncoTreeRelease(
        version=version,
        nodes_by_code=nodes_by_code,
        _exact_name=_freeze_index(exact_name),
        _exact_code=_freeze_index(exact_code),
        _normalized_name=_freeze_index(normalized_name),
        _normalized_code=_freeze_index(normalized_code),
    )


def _index_surface(
    exact: dict[str, list[str]],
    normalized: dict[str, list[str]],
    surface: str,
    code: str,
) -> None:
    exact[surface].append(code)
    normalized_key = normalize_surface(surface)
    if normalized_key:
        normalized[normalized_key].append(code)


def _index_code_alias(
    exact: dict[str, list[str]],
    normalized: dict[str, list[str]],
    alias: str,
    *,
    target: str,
) -> None:
    exact[alias].append(target)
    normalized_key = normalize_surface(alias)
    if normalized_key:
        normalized[normalized_key].append(target)


def _freeze_index(index: Mapping[str, list[str]]) -> dict[str, tuple[str, ...]]:
    return {key: tuple(values) for key, values in index.items()}


def _mapped(
    node: OncoTreeNode,
    version: str,
    confidence: float,
) -> OncoTreeMapping:
    return OncoTreeMapping(
        code=node.code,
        name=node.name,
        main_type=node.main_type,
        tissue=node.tissue,
        oncotree_version=version,
        match_confidence=confidence,
        reason=None,
        advisory=ONCOTREE_ADVISORY,
    )


def _unmapped(version: str, *, reason: str) -> OncoTreeMapping:
    return OncoTreeMapping(
        code=None,
        name=None,
        main_type=None,
        tissue=None,
        oncotree_version=version,
        match_confidence=_CONFIDENCE_UNMAPPED,
        reason=reason,
        advisory=ONCOTREE_ADVISORY,
    )


__all__ = [
    "ONCOTREE_ADVISORY",
    "OncoTreeMapping",
    "OncoTreeNode",
    "OncoTreeRelease",
    "load_oncotree",
    "map_tumor_type",
]
