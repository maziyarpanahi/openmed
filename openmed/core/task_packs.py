"""Portable task packs: one declaration of which model serves which job.

A task pack names a job (PII detection, clinical-entity extraction,
medication-mention extraction) instead of a repository. The declarations in
this module are the single source of truth; ``scripts/manifest/build_task_packs.py``
applies their selection rules to the committed ``models.jsonl`` and writes the
ordered, per-runtime candidates to ``openmed/core/data/task_packs.json``.
OpenMedKit ports read that generated file; they do not re-run the selection.

:func:`resolve` only reads the bundled manifest. It never downloads a model,
contacts a hub or imports an inference runtime.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from openmed.core.labels import CANONICAL_LABELS, normalize_label

TASK_PACKS_SCHEMA = "openmed.task_packs"
TASK_PACKS_SCHEMA_VERSION = "1.0.0"
TASK_PACKS_DATA_FILE = Path(__file__).with_name("data") / "task_packs.json"
TASK_PACK_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "json" / "task_pack.schema.json"
)

#: Runtime name -> the single model format a kit on that runtime loads.
RUNTIME_FORMATS: Mapping[str, str] = {
    "android": "onnx",
    "apple": "mlx-fp",
    "web": "onnx",
}
RUNTIMES: tuple[str, ...] = tuple(sorted(RUNTIME_FORMATS))

STATUS_RESOLVED = "resolved"
STATUS_NO_CANDIDATE = "no_candidate"
CANDIDATE_ORDER = "param_count_asc_repo_id_asc"
LABEL_COVERAGE_ALL = "all"

#: Licenses a candidate may carry. Rows with any other license, or none, are
#: never selected.
PERMISSIVE_LICENSES: tuple[str, ...] = (
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "isc",
    "mit",
    "unlicense",
)

NOTICE_PII_REVIEW = "openmed.notice.pii.review_required"
NOTICE_CLINICAL_ENTITY_EXTRACTION_ONLY = (
    "openmed.notice.clinical_entity.extraction_only"
)
NOTICE_MEDICATION_EXTRACTION_ONLY = "openmed.notice.medication.extraction_only"

#: Notice key -> default English text. Kits localize by key.
NOTICES: Mapping[str, str] = {
    NOTICE_CLINICAL_ENTITY_EXTRACTION_ONLY: (
        "Reports condition mentions and their offsets only. It does not "
        "diagnose, rank or recommend care."
    ),
    NOTICE_MEDICATION_EXTRACTION_ONLY: (
        "Reports medication mentions and their offsets only. It never gives "
        "dosing advice and never triggers an action."
    ),
    NOTICE_PII_REVIEW: (
        "Detected identifiers are redaction candidates. Automated detection "
        "can miss identifiers, so review output before it leaves the device."
    ),
}

_PACK_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_TOKEN_SPLIT_RE = re.compile(r"[-_]+")


class TaskPackError(ValueError):
    """Raised with a value-free reason code when a pack cannot be resolved."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class TaskPackSelection:
    """Rules that pick candidate rows from ``models.jsonl``.

    Attributes:
        family: Required manifest ``family`` value (for example ``"PII"``).
        family_token: Optional repository-name token, compared
            case-insensitively against the ``-``/``_`` separated parts of the
            repository name (for example ``"PharmaDetect"``).
        manifest_task: Required manifest ``task`` value.
        languages: Every language the candidate must claim.
        label_coverage: ``"all"``: the candidate's normalized labels must
            include every pack canonical label.
        max_param_count: Size cap; rows without a parameter count fail it.
        licenses: Accepted licenses, all permissive.
        max_candidates: Upper bound on candidates kept per runtime.
        order: Candidate order; ascending parameter count, then repository id.
    """

    family: str
    manifest_task: str
    languages: tuple[str, ...]
    max_param_count: int
    max_candidates: int
    family_token: str | None = None
    label_coverage: str = LABEL_COVERAGE_ALL
    licenses: tuple[str, ...] = PERMISSIVE_LICENSES
    order: str = CANDIDATE_ORDER

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready selection rules."""

        return {
            "family": self.family,
            "family_token": self.family_token,
            "label_coverage": self.label_coverage,
            "languages": list(self.languages),
            "licenses": list(self.licenses),
            "manifest_task": self.manifest_task,
            "max_candidates": self.max_candidates,
            "max_param_count": self.max_param_count,
            "order": self.order,
        }


@dataclass(frozen=True, slots=True)
class TaskPackDeclaration:
    """A task pack before its candidates are resolved against a manifest."""

    pack_id: str
    version: str
    task: str
    canonical_labels: tuple[str, ...]
    selection: TaskPackSelection
    notice_key: str


@dataclass(frozen=True, slots=True)
class TaskPackCandidate:
    """One ordered model candidate for a pack on one runtime."""

    rank: int
    repo_id: str
    param_count: int
    license: str


@dataclass(frozen=True, slots=True)
class TaskPackResolution:
    """The ordered candidates of one pack on one runtime.

    Attributes:
        pack_id: Pack identifier.
        pack_version: Semantic version of the pack declaration.
        task: Job the pack performs.
        runtime: ``apple``, ``android`` or ``web``.
        format: Model format the runtime loads.
        status: ``"resolved"`` or ``"no_candidate"``.
        candidates: Candidates, best first. Empty when ``status`` is
            ``"no_candidate"``.
        canonical_labels: Labels every candidate is guaranteed to emit.
        notice_key: Key of the notice a kit must show with pack output.
    """

    pack_id: str
    pack_version: str
    task: str
    runtime: str
    format: str
    status: str
    candidates: tuple[TaskPackCandidate, ...]
    canonical_labels: tuple[str, ...]
    notice_key: str

    @property
    def repo_ids(self) -> tuple[str, ...]:
        """Return candidate repository ids, best first."""

        return tuple(candidate.repo_id for candidate in self.candidates)


_PII_LABELS = (
    "ACCOUNT_NUMBER",
    "DATE_OF_BIRTH",
    "EMAIL",
    "FIRST_NAME",
    "ID_NUM",
    "IP_ADDRESS",
    "LAST_NAME",
    "PERSON",
    "PHONE",
    "SSN",
    "STREET_ADDRESS",
    "URL",
)

#: The portable task packs. Edit here, then regenerate the manifest with
#: ``python scripts/manifest/build_task_packs.py``.
TASK_PACK_DECLARATIONS: tuple[TaskPackDeclaration, ...] = (
    TaskPackDeclaration(
        pack_id="clinical_entity",
        version="1.0.0",
        task="clinical_entity_extraction",
        canonical_labels=("CONDITION", "DISEASE", "PATHOLOGY"),
        selection=TaskPackSelection(
            family="NER",
            family_token="DiseaseDetect",
            manifest_task="token-classification",
            languages=("en",),
            max_param_count=150_000_000,
            max_candidates=3,
        ),
        notice_key=NOTICE_CLINICAL_ENTITY_EXTRACTION_ONLY,
    ),
    TaskPackDeclaration(
        pack_id="medication",
        version="1.0.0",
        task="medication_mention_extraction",
        canonical_labels=("CHEMICAL",),
        selection=TaskPackSelection(
            family="NER",
            family_token="PharmaDetect",
            manifest_task="token-classification",
            languages=("en",),
            max_param_count=150_000_000,
            max_candidates=3,
        ),
        notice_key=NOTICE_MEDICATION_EXTRACTION_ONLY,
    ),
    TaskPackDeclaration(
        pack_id="pii",
        version="1.0.0",
        task="pii_detection",
        canonical_labels=_PII_LABELS,
        selection=TaskPackSelection(
            family="PII",
            manifest_task="token-classification",
            languages=("en",),
            max_param_count=150_000_000,
            max_candidates=3,
        ),
        notice_key=NOTICE_PII_REVIEW,
    ),
)


def _repo_tokens(repo_id: str) -> set[str]:
    name = repo_id.rsplit("/", 1)[-1]
    return {token.casefold() for token in _TOKEN_SPLIT_RE.split(name) if token}


def _normalized_labels(values: object) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {normalize_label(value) for value in values if isinstance(value, str)}


def _row_matches(
    row: Mapping[str, Any],
    declaration: TaskPackDeclaration,
    model_format: str,
) -> bool:
    selection = declaration.selection
    repo_id = row.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        return False
    if row.get("family") != selection.family:
        return False
    if row.get("task") != selection.manifest_task:
        return False
    if selection.family_token is not None and (
        selection.family_token.casefold() not in _repo_tokens(repo_id)
    ):
        return False
    formats = row.get("formats")
    if not isinstance(formats, list) or model_format not in formats:
        return False
    languages = row.get("languages")
    if not isinstance(languages, list) or not set(selection.languages) <= set(
        languages
    ):
        return False
    if not set(declaration.canonical_labels) <= _normalized_labels(
        row.get("canonical_labels")
    ):
        return False
    param_count = row.get("param_count")
    if type(param_count) is not int or param_count <= 0:
        return False
    if param_count > selection.max_param_count:
        return False
    return row.get("license") in selection.licenses


def _select_candidates(
    rows: Sequence[Mapping[str, Any]],
    declaration: TaskPackDeclaration,
    model_format: str,
) -> list[dict[str, Any]]:
    matches = [row for row in rows if _row_matches(row, declaration, model_format)]
    matches.sort(key=lambda row: (row["param_count"], row["repo_id"]))
    kept = matches[: declaration.selection.max_candidates]
    return [
        {
            "license": row["license"],
            "param_count": row["param_count"],
            "rank": rank,
            "repo_id": row["repo_id"],
        }
        for rank, row in enumerate(kept, start=1)
    ]


def build_task_packs(
    rows: Iterable[Mapping[str, Any]],
    declarations: Sequence[TaskPackDeclaration] = TASK_PACK_DECLARATIONS,
) -> dict[str, Any]:
    """Resolve pack declarations against manifest rows.

    Args:
        rows: ``models.jsonl`` rows. Non-object rows are ignored.
        declarations: Pack declarations to resolve.

    Returns:
        The JSON-ready task pack manifest. A runtime with no matching row is
        recorded with ``status: "no_candidate"`` and an empty candidate list.
    """

    materialized = [row for row in rows if isinstance(row, Mapping)]
    packs: list[dict[str, Any]] = []
    for declaration in sorted(declarations, key=lambda item: item.pack_id):
        runtimes: dict[str, Any] = {}
        for runtime in RUNTIMES:
            model_format = RUNTIME_FORMATS[runtime]
            candidates = _select_candidates(materialized, declaration, model_format)
            runtimes[runtime] = {
                "candidates": candidates,
                "format": model_format,
                "status": STATUS_RESOLVED if candidates else STATUS_NO_CANDIDATE,
            }
        packs.append(
            {
                "canonical_labels": sorted(declaration.canonical_labels),
                "notice_key": declaration.notice_key,
                "pack_id": declaration.pack_id,
                "runtimes": runtimes,
                "selection": declaration.selection.to_dict(),
                "task": declaration.task,
                "version": declaration.version,
            }
        )
    notice_keys = sorted({declaration.notice_key for declaration in declarations})
    document = {
        "notices": {key: NOTICES[key] for key in notice_keys},
        "packs": packs,
        "schema": TASK_PACKS_SCHEMA,
        "schema_version": TASK_PACKS_SCHEMA_VERSION,
        "source": "models.jsonl",
    }
    validate_task_packs(document)
    return document


def render_task_packs(document: Mapping[str, Any]) -> str:
    """Return the canonical, byte-stable JSON text of a task pack manifest."""

    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


@lru_cache(maxsize=1)
def load_task_pack_schema() -> dict[str, Any]:
    """Load the bundled ``task_pack.schema.json``."""

    return json.loads(TASK_PACK_SCHEMA_FILE.read_text(encoding="utf-8"))


def validate_task_packs(document: Any) -> None:
    """Validate a task pack manifest and fail closed.

    Args:
        document: Parsed manifest.

    Raises:
        TaskPackError: With reason ``schema_invalid`` when the JSON Schema
            rejects the document, or a specific value-free reason when a
            cross-field rule fails.
    """

    validator = Draft202012Validator(load_task_pack_schema())
    if next(iter(validator.iter_errors(document)), None) is not None:
        raise TaskPackError("schema_invalid")
    notices = document["notices"]
    seen: set[str] = set()
    for pack in document["packs"]:
        if pack["pack_id"] in seen:
            raise TaskPackError("duplicate_pack")
        seen.add(pack["pack_id"])
        if pack["notice_key"] not in notices:
            raise TaskPackError("unknown_notice_key")
        if set(pack["runtimes"]) != set(RUNTIMES):
            raise TaskPackError("runtime_set_mismatch")
        licenses = set(pack["selection"]["licenses"])
        if not licenses <= set(PERMISSIVE_LICENSES):
            raise TaskPackError("license_not_permissive")
        if not set(pack["canonical_labels"]) <= CANONICAL_LABELS:
            raise TaskPackError("label_not_canonical")
        for runtime, entry in pack["runtimes"].items():
            if entry["format"] != RUNTIME_FORMATS[runtime]:
                raise TaskPackError("runtime_format_mismatch")
            candidates = entry["candidates"]
            expected_status = STATUS_RESOLVED if candidates else STATUS_NO_CANDIDATE
            if entry["status"] != expected_status:
                raise TaskPackError("status_mismatch")
            if [item["rank"] for item in candidates] != list(
                range(1, len(candidates) + 1)
            ):
                raise TaskPackError("rank_not_sequential")
            if len(candidates) > pack["selection"]["max_candidates"]:
                raise TaskPackError("too_many_candidates")
            if any(item["license"] not in licenses for item in candidates):
                raise TaskPackError("license_not_permissive")
            if any(
                item["param_count"] > pack["selection"]["max_param_count"]
                for item in candidates
            ):
                raise TaskPackError("size_cap_exceeded")
            order = [(item["param_count"], item["repo_id"]) for item in candidates]
            if order != sorted(order) or len(set(order)) != len(order):
                raise TaskPackError("candidate_order_invalid")


@lru_cache(maxsize=1)
def _bundled_task_packs() -> dict[str, Any]:
    return load_task_packs(TASK_PACKS_DATA_FILE)


def load_task_packs(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a task pack manifest from disk.

    Args:
        path: Manifest path. Defaults to the bundled
            ``openmed/core/data/task_packs.json``.

    Returns:
        The validated manifest document.

    Raises:
        TaskPackError: ``manifest_unreadable`` when the file cannot be read or
            parsed, or a validation reason from :func:`validate_task_packs`.
    """

    source = Path(path) if path is not None else TASK_PACKS_DATA_FILE
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise TaskPackError("manifest_unreadable") from None
    validate_task_packs(document)
    return document


def available_packs(manifest: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Return the pack ids declared in a manifest, sorted.

    Args:
        manifest: A validated manifest. Defaults to the bundled one.
    """

    document = _bundled_task_packs() if manifest is None else manifest
    return tuple(sorted(pack["pack_id"] for pack in document["packs"]))


def resolve(
    pack_id: str,
    runtime: str,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> TaskPackResolution:
    """Return the ordered model candidates of a pack on a runtime.

    This reads metadata only: it never downloads, verifies or loads a model.

    Args:
        pack_id: Pack identifier such as ``"pii"``, ``"clinical_entity"`` or
            ``"medication"``.
        runtime: ``"apple"``, ``"android"`` or ``"web"``.
        manifest: A manifest already validated by :func:`load_task_packs`.
            Defaults to the bundled manifest.

    Returns:
        The resolution. ``status`` is ``"no_candidate"`` with no candidates
        when the manifest records none for that runtime.

    Raises:
        TaskPackError: ``unknown_runtime`` or ``unknown_pack``.
    """

    if not isinstance(runtime, str) or runtime not in RUNTIME_FORMATS:
        raise TaskPackError("unknown_runtime")
    if not isinstance(pack_id, str) or _PACK_ID_RE.fullmatch(pack_id) is None:
        raise TaskPackError("unknown_pack")
    document = _bundled_task_packs() if manifest is None else manifest
    for pack in document["packs"]:
        if pack["pack_id"] != pack_id:
            continue
        entry = pack["runtimes"][runtime]
        return TaskPackResolution(
            pack_id=pack["pack_id"],
            pack_version=pack["version"],
            task=pack["task"],
            runtime=runtime,
            format=entry["format"],
            status=entry["status"],
            candidates=tuple(
                TaskPackCandidate(
                    rank=item["rank"],
                    repo_id=item["repo_id"],
                    param_count=item["param_count"],
                    license=item["license"],
                )
                for item in entry["candidates"]
            ),
            canonical_labels=tuple(pack["canonical_labels"]),
            notice_key=pack["notice_key"],
        )
    raise TaskPackError("unknown_pack")


__all__ = [
    "CANDIDATE_ORDER",
    "NOTICES",
    "NOTICE_CLINICAL_ENTITY_EXTRACTION_ONLY",
    "NOTICE_MEDICATION_EXTRACTION_ONLY",
    "NOTICE_PII_REVIEW",
    "PERMISSIVE_LICENSES",
    "RUNTIMES",
    "RUNTIME_FORMATS",
    "STATUS_NO_CANDIDATE",
    "STATUS_RESOLVED",
    "TASK_PACKS_SCHEMA",
    "TASK_PACKS_SCHEMA_VERSION",
    "TASK_PACK_DECLARATIONS",
    "TaskPackCandidate",
    "TaskPackDeclaration",
    "TaskPackError",
    "TaskPackResolution",
    "TaskPackSelection",
    "available_packs",
    "build_task_packs",
    "load_task_pack_schema",
    "load_task_packs",
    "render_task_packs",
    "resolve",
    "validate_task_packs",
]
