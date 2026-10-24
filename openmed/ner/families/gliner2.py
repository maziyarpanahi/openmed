"""GLiNER2.5 multi-task checkpoints loaded through ``gliner2.AutoExtractor``.

GLiNER2.5 checkpoints ship in the separate Apache-2.0 ``gliner2`` package
(``pip install openmed[gliner2]``). Its ``AutoExtractor`` reads the architecture
saved with each checkpoint (``span`` or ``boundary``) and returns the matching
extractor; ``GLiNER2.from_pretrained`` is the legacy span-only loader and cannot
load the boundary checkpoints.

The integration keeps OpenMed's local-first defaults:

* ``import openmed`` never imports ``gliner2``. Availability is probed with
  :func:`importlib.util.find_spec` through the shared capability registry.
* A checkpoint is resolved to a local snapshot before it is loaded. A cached
  snapshot is reused without a network call, downloads happen only when the
  snapshot is missing and offline mode is off, and inference never uses the
  network. The ``gliner2`` cloud API client is refused.
* Checkpoints that request remote code or ship only pickle weights are refused.
* Results carry labels, offsets and scores. Error messages never echo request
  text, and nothing in this module logs or caches request text.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union

from openmed.core.capabilities import backend_status, is_backend_available
from openmed.core.errors import ModelLoadError
from openmed.core.offline import (
    OfflineModeError,
    is_local_only,
    network_blocked_if_offline,
)

from ..exceptions import MissingDependencyError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..infer import Entity

_PRIMARY_IMPORT = "gliner2"
_BACKEND_NAME = "gliner2"
_GLINER2_HINT = (
    "Install with `pip install openmed[gliner2]` (or `pip install "
    '"gliner2[local]"`) to load GLiNER2.5 checkpoints through '
    "gliner2.AutoExtractor."
)
_SNAPSHOT_PATTERNS = ("*.json", "*.safetensors", "*.model", "*.txt")
_CONFIG_FILES = ("config.json", "encoder_config/config.json")
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
_WEIGHTS_FILE = "model.safetensors"
_REMOTE_CODE_FILES = (*_CONFIG_FILES, "tokenizer_config.json")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CLOUD_CLIENT_MODULE = "gliner2.api_client"
_TASK_OPTIONS = frozenset({"labels", "multi_label", "cls_threshold"})

DEFAULT_CHUNK_SIZE = 384
"""Word tokens per chunk for the GLiNER2 long-document helpers."""

DEFAULT_CHUNK_OVERLAP = 64
"""Word tokens shared by adjacent chunks in the long-document helpers."""

GLINER2_TASKS = ("entities", "classification", "relations", "json_extraction")
"""Tasks every GLiNER2.5 checkpoint exposes through ``AutoExtractor``."""

_availability_cache: Optional[bool] = None

LabelSpec = Union[Sequence[str], Mapping[str, str]]


class GLiNER2OutputError(ValueError):
    """Raised when a GLiNER2 payload does not match the requested contract.

    Messages name the failed check only; they never contain request text, span
    text or label descriptions.
    """


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def is_gliner2_available(force_refresh: bool = False) -> bool:
    """Return True when ``gliner2[local]`` and its runtime are importable.

    The probe uses :func:`importlib.util.find_spec` through the shared
    capability registry, so it never imports ``gliner2``, ``torch`` or
    ``transformers``.

    Args:
        force_refresh: Re-probe instead of returning the cached answer.

    Returns:
        True when ``gliner2``, ``torch``, ``transformers``, ``safetensors``
        and ``huggingface_hub`` can all be imported.
    """

    global _availability_cache
    if force_refresh or _availability_cache is None:
        _availability_cache = is_backend_available(_BACKEND_NAME)
    return bool(_availability_cache)


def ensure_gliner2_available() -> None:
    """Raise :class:`MissingDependencyError` unless ``gliner2[local]`` is usable.

    Raises:
        MissingDependencyError: Naming the first missing module and the
            ``openmed[gliner2]`` install hint.
    """

    if is_gliner2_available(force_refresh=False):
        return

    missing = _missing_dependency()
    dependency = missing or _PRIMARY_IMPORT
    raise MissingDependencyError(dependency=dependency, instruction=_GLINER2_HINT)


def _missing_dependency() -> Optional[str]:
    missing = backend_status(_BACKEND_NAME).missing
    return missing[0] if missing else None


# ---------------------------------------------------------------------------
# Checkpoint capability metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GLiNER2CheckpointSpec:
    """Capability metadata for one published GLiNER2.5 checkpoint.

    Attributes:
        model_id: Hugging Face repository id.
        parameters: Parameter count reported by the checkpoint's safetensors
            metadata.
        languages: Supported languages; ``"multi"`` marks multilingual
            checkpoints, matching the model index convention.
        architecture: Architecture saved in the reviewed checkpoint config.
            ``AutoExtractor`` dispatches on the saved value at load time.
        decision_tuned: True for the ``Decide`` checkpoints fine-tuned for
            fixed-option classification.
        license_id: SPDX license identifier of the weights.
        tasks: Tasks exposed through ``AutoExtractor``.
    """

    model_id: str
    parameters: int
    languages: tuple[str, ...]
    architecture: str
    decision_tuned: bool = False
    license_id: str = "Apache-2.0"
    tasks: tuple[str, ...] = GLINER2_TASKS

    @property
    def multilingual(self) -> bool:
        """Return True when the checkpoint is trained for many languages."""

        return "multi" in self.languages

    @property
    def size_label(self) -> str:
        """Return a short human-readable parameter count such as ``74M``."""

        if self.parameters >= 1_000_000_000:
            return f"{self.parameters / 1_000_000_000:.2f}B"
        return f"{round(self.parameters / 1_000_000)}M"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible capability record."""

        return {
            "model_id": self.model_id,
            "family": "gliner2",
            "parameters": self.parameters,
            "size": self.size_label,
            "languages": list(self.languages),
            "multilingual": self.multilingual,
            "architecture": self.architecture,
            "decision_tuned": self.decision_tuned,
            "license_id": self.license_id,
            "tasks": list(self.tasks),
            "runtime": "gliner2",
            "requires_remote_code": False,
        }


GLINER2_CHECKPOINTS: Mapping[str, GLiNER2CheckpointSpec] = MappingProxyType(
    {
        spec.model_id: spec
        for spec in (
            GLiNER2CheckpointSpec(
                model_id="fastino/gliner2.5-small-v1",
                parameters=73_881_879,
                languages=("en",),
                architecture="boundary",
            ),
            GLiNER2CheckpointSpec(
                model_id="fastino/gliner2.5-base-v1",
                parameters=193_581_591,
                languages=("en",),
                architecture="boundary",
            ),
            GLiNER2CheckpointSpec(
                model_id="fastino/gliner2.5-multi-v1",
                parameters=287_355_159,
                languages=("multi",),
                architecture="boundary",
            ),
            GLiNER2CheckpointSpec(
                model_id="fastino/GLiNER2.5-multi-Decide",
                parameters=287_355_159,
                languages=("multi",),
                architecture="boundary",
                decision_tuned=True,
            ),
            GLiNER2CheckpointSpec(
                model_id="fastino/GLiNER2.5-Decide",
                parameters=486_444_053,
                languages=("en",),
                architecture="span",
                decision_tuned=True,
            ),
            GLiNER2CheckpointSpec(
                model_id="fastino/GLiNER2.5-Decide-1B",
                parameters=1_188_796_693,
                languages=("en",),
                architecture="span",
                decision_tuned=True,
            ),
        )
    }
)
"""Published GLiNER2.5 checkpoints, keyed by Hugging Face repository id."""


def list_gliner2_checkpoints() -> tuple[GLiNER2CheckpointSpec, ...]:
    """Return every registered GLiNER2.5 checkpoint in publication order."""

    return tuple(GLINER2_CHECKPOINTS.values())


def get_gliner2_checkpoint(model_id: str) -> Optional[GLiNER2CheckpointSpec]:
    """Return capability metadata for ``model_id`` when it is registered.

    Hugging Face repository ids are case-insensitive, so the lookup is too.
    """

    if not isinstance(model_id, str):
        return None
    spec = GLINER2_CHECKPOINTS.get(model_id)
    if spec is not None:
        return spec
    folded = model_id.strip().casefold()
    for candidate in GLINER2_CHECKPOINTS.values():
        if candidate.model_id.casefold() == folded:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GLiNER2LabelScore:
    """One label and its probability for a classification task or choice."""

    label: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {"label": self.label, "score": self.score}


@dataclass(frozen=True)
class GLiNER2Classification:
    """Scores for every label of one classification task.

    ``scores`` follows the caller's label order. Single-label tasks hold a
    softmax distribution over the labels; multi-label tasks hold independent
    sigmoid probabilities. ``selected`` is derived from the scores rather than
    from the labels ``gliner2`` would print.
    """

    task: str
    scores: tuple[GLiNER2LabelScore, ...]
    multi_label: bool = False
    threshold: float = 0.5

    @property
    def selected(self) -> tuple[str, ...]:
        """Return the arg-max label, or every label at or above ``threshold``."""

        if not self.scores:
            return ()
        if self.multi_label:
            return tuple(
                item.label for item in self.scores if item.score >= self.threshold
            )
        best = max(range(len(self.scores)), key=lambda i: (self.scores[i].score, -i))
        return (self.scores[best].label,)

    @property
    def label(self) -> Optional[str]:
        """Return the first selected label, if any."""

        selected = self.selected
        return selected[0] if selected else None

    def score_for(self, label: str) -> float:
        """Return the score for ``label``.

        Raises:
            KeyError: If the task did not include ``label``.
        """

        for item in self.scores:
            if item.label == label:
                return item.score
        raise KeyError("label is not part of this classification task")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "task": self.task,
            "multi_label": self.multi_label,
            "threshold": self.threshold,
            "scores": [item.to_dict() for item in self.scores],
            "selected": list(self.selected),
        }


@dataclass(frozen=True)
class GLiNER2Relation:
    """A typed relation between two source spans.

    ``head`` and ``tail`` are :class:`~openmed.ner.infer.Entity` objects whose
    ``label`` is the relation type and whose ``group`` is ``"head"`` or
    ``"tail"``.
    """

    relation: str
    head: "Entity"
    tail: "Entity"
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "relation": self.relation,
            "head": self.head.to_dict(),
            "tail": self.tail.to_dict(),
            "score": self.score,
        }


@dataclass(frozen=True)
class GLiNER2Record:
    """One instance of a schema-driven JSON structure.

    ``fields`` holds source spans for extractive fields (empty when absent).
    ``choices`` holds values for fields declared with fixed choices; those
    values are labels rather than source spans, so they carry no offsets.
    """

    structure: str
    fields: Mapping[str, tuple["Entity", ...]] = field(default_factory=dict)
    choices: Mapping[str, tuple[GLiNER2LabelScore, ...]] = field(default_factory=dict)
    score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "structure": self.structure,
            "fields": {
                name: [entity.to_dict() for entity in values]
                for name, values in self.fields.items()
            },
            "choices": {
                name: [item.to_dict() for item in values]
                for name, values in self.choices.items()
            },
            "score": self.score,
        }


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------


@dataclass
class GLiNER2Handle:
    """Local GLiNER2.5 extractor with OpenMed-typed multi-task helpers.

    Attributes:
        model_id: Repository id or local directory supplied by the caller.
        model: The ``gliner2`` extractor returned by ``AutoExtractor``.
        revision: Resolved checkpoint revision (a commit hash for Hub
            snapshots, a config digest for unversioned local directories).
        license_id: SPDX license of the weights, when known.
        architecture: Architecture reported by the loaded extractor.
        source: ``"hub"`` for repository ids, ``"local"`` for directories.
    """

    model_id: str
    model: Any
    revision: Optional[str] = None
    license_id: Optional[str] = None
    architecture: Optional[str] = None
    source: str = "hub"

    def __post_init__(self) -> None:
        module = getattr(type(self.model), "__module__", "") or ""
        if module == _CLOUD_CLIENT_MODULE or module.startswith(
            _CLOUD_CLIENT_MODULE + "."
        ):
            raise ValueError(
                "GLiNER2 cloud API clients are not supported; OpenMed runs "
                "GLiNER2 checkpoints locally"
            )

    @property
    def public_model_id(self) -> str:
        """Return a model id that never exposes a local filesystem path."""

        if self.source != "local":
            return self.model_id
        name = os.path.basename(os.path.normpath(self.model_id)) or "checkpoint"
        return f"local/{name}"

    def metadata(self) -> dict[str, Any]:
        """Return path-free provenance for reports and backend identities."""

        spec = get_gliner2_checkpoint(self.model_id)
        return {
            "family": "gliner2",
            "runtime": "gliner2",
            "model_id": self.public_model_id,
            "revision": self.revision,
            "license_id": self.license_id,
            "architecture": self.architecture,
            "source": self.source,
            "checkpoint": spec.to_dict() if spec is not None else None,
        }

    # -- entities ---------------------------------------------------------

    def extract_entities(
        self,
        text: str,
        labels: LabelSpec,
        *,
        threshold: float = 0.5,
        long_document: Optional[bool] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        overlap_policy: Optional[str] = None,
    ) -> list["Entity"]:
        """Extract zero-shot entities as OpenMed entities with absolute offsets.

        Args:
            text: Source text.
            labels: Entity labels, or a mapping of label to description.
            threshold: Minimum span confidence in ``[0, 1]``.
            long_document: Use ``extract_entities_long`` (overlapping word
                chunks remapped to document offsets). ``None`` selects it when
                the text has more word tokens than ``chunk_size``.
            chunk_size: Word tokens per chunk for the long-document helper.
            chunk_overlap: Word tokens shared by adjacent chunks.
            overlap_policy: Optional ``gliner2`` overlap policy such as
                ``"flat"`` or ``"nested"``; ``None`` keeps the checkpoint
                default.

        Returns:
            Entities ordered by offset. ``text`` always equals
            ``source[start:end]``.

        Raises:
            GLiNER2OutputError: If the payload has invalid offsets, scores or
                labels.
        """

        source = _require_text(text)
        requested = _label_names(labels, "entity labels")
        payload = self._extract(
            "extract_entities",
            source,
            _model_labels(labels),
            threshold=threshold,
            long_document=long_document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overlap_policy=overlap_policy,
        )
        return _convert_entities(source, payload, requested)

    def predict_entities(
        self,
        text: str,
        labels: Optional[Sequence[str]] = None,
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return GLiNER v1-style entity dictionaries.

        Kept for callers of the earlier handle. ``flat_ner=False`` maps to the
        ``nested`` overlap policy. Extra keyword arguments are forwarded to
        :meth:`extract_entities`.
        """

        ensure_gliner2_available()
        options = dict(kwargs)
        if not flat_ner:
            options.setdefault("overlap_policy", "nested")
        entities = self.extract_entities(
            text, list(labels or ()), threshold=threshold, **options
        )
        return [
            {
                "text": entity.text,
                "start": entity.start,
                "end": entity.end,
                "label": entity.label,
                "score": entity.score,
            }
            for entity in entities
        ]

    # -- classification ---------------------------------------------------

    def classify_text(
        self,
        text: str,
        tasks: Mapping[str, Union[LabelSpec, Mapping[str, Any]]],
    ) -> dict[str, GLiNER2Classification]:
        """Score every label of one or more classification tasks.

        A task maps to its labels (single-label), or to a mapping with
        ``labels``, optional ``multi_label`` (default False) and optional
        ``cls_threshold`` (default 0.5, used for multi-label selection).
        Labels may carry descriptions as a label-to-description mapping.

        The handle always asks ``gliner2`` for the full score vector: a softmax
        distribution for single-label tasks and independent sigmoid
        probabilities for multi-label tasks. Selection is then derived from
        those scores.

        Args:
            text: Source text.
            tasks: Classification tasks keyed by task name.

        Returns:
            One :class:`GLiNER2Classification` per task, in task order.

        Raises:
            GLiNER2OutputError: If a label is missing a valid score.
        """

        source = _require_text(text)
        plans = _classification_plans(tasks)
        model_tasks = {
            name: {
                "labels": plan.model_labels,
                "multi_label": True,
                "cls_threshold": 0.0,
                "class_act": "sigmoid" if plan.multi_label else "softmax",
            }
            for name, plan in plans.items()
        }
        with network_blocked_if_offline():
            payload = self.model.classify_text(
                source, model_tasks, include_confidence=True
            )
        if not isinstance(payload, Mapping):
            raise GLiNER2OutputError("GLiNER2 classification payload is not a mapping")
        return {
            name: _convert_classification(name, payload.get(name), plan)
            for name, plan in plans.items()
        }

    # -- relations --------------------------------------------------------

    def extract_relations(
        self,
        text: str,
        relation_types: LabelSpec,
        *,
        threshold: float = 0.5,
        long_document: Optional[bool] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[GLiNER2Relation]:
        """Extract typed head/tail relations with absolute offsets.

        Args:
            text: Source text.
            relation_types: Relation names, or a mapping of name to
                description.
            threshold: Minimum argument confidence in ``[0, 1]``.
            long_document: Use ``extract_relations_long``; ``None`` decides
                from the text length as in :meth:`extract_entities`.
            chunk_size: Word tokens per chunk for the long-document helper.
            chunk_overlap: Word tokens shared by adjacent chunks.

        Returns:
            Relations ordered by head and tail offsets. The relation score is
            the instance confidence when ``gliner2`` reports one, otherwise
            the lower of the two argument confidences.
        """

        source = _require_text(text)
        requested = _label_names(relation_types, "relation types")
        payload = self._extract(
            "extract_relations",
            source,
            _model_labels(relation_types),
            threshold=threshold,
            long_document=long_document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return _convert_relations(source, payload, requested)

    def predict_relations(
        self,
        text: str,
        labels: Optional[Sequence[str]] = None,
        *,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return relation dictionaries; kept for callers of the earlier handle."""

        ensure_gliner2_available()
        relations = self.extract_relations(
            text, list(labels or ()), threshold=threshold, **kwargs
        )
        return [relation.to_dict() for relation in relations]

    # -- schema-driven JSON extraction -------------------------------------

    def extract_json(
        self,
        text: str,
        structures: Mapping[str, Sequence[Union[str, Mapping[str, Any]]]],
        *,
        threshold: float = 0.5,
        long_document: Optional[bool] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> dict[str, tuple[GLiNER2Record, ...]]:
        """Extract schema-driven records with span offsets for every field.

        ``structures`` uses the ``gliner2`` field syntax, for example
        ``{"prescription": ["drug::str", "dose::str", "route::[oral|iv]::str"]}``.

        Args:
            text: Source text.
            structures: Field specifications keyed by structure name.
            threshold: Minimum field confidence in ``[0, 1]``.
            long_document: Use ``extract_json_long``; ``None`` decides from the
                text length as in :meth:`extract_entities`.
            chunk_size: Word tokens per chunk for the long-document helper.
            chunk_overlap: Word tokens shared by adjacent chunks.

        Returns:
            Records keyed by structure name. Extractive fields become entities
            whose ``label`` is the field name; fixed-choice fields become
            label scores without offsets.
        """

        source = _require_text(text)
        if not isinstance(structures, Mapping) or not structures:
            raise ValueError("structures must be a non-empty mapping")
        names = []
        for name, fields in structures.items():
            names.append(_require_name(name, "structure name"))
            if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
                raise TypeError("structure fields must be a sequence")
            if not fields:
                raise ValueError("structures need at least one field")
        payload = self._extract(
            "extract_json",
            source,
            {name: list(fields) for name, fields in structures.items()},
            threshold=threshold,
            long_document=long_document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return _convert_records(source, payload, names)

    # -- helpers ----------------------------------------------------------

    def _extract(
        self,
        method: str,
        source: str,
        spec: Any,
        *,
        threshold: float,
        long_document: Optional[bool],
        chunk_size: int,
        chunk_overlap: int,
        **options: Any,
    ) -> Any:
        """Call ``method`` or its ``_long`` variant with spans and scores on."""

        _require_probability(threshold, "threshold")
        options.update(threshold=threshold, include_confidence=True, include_spans=True)
        if self._use_long(source, long_document, chunk_size, chunk_overlap):
            method = f"{method}_long"
            options.update(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        with network_blocked_if_offline():
            return getattr(self.model, method)(source, spec, **options)

    @staticmethod
    def _use_long(
        text: str,
        long_document: Optional[bool],
        chunk_size: int,
        chunk_overlap: int,
    ) -> bool:
        for value, name in (
            (chunk_size, "chunk_size"),
            (chunk_overlap, "chunk_overlap"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if chunk_size <= 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than a positive chunk_size")
        if long_document is None:
            return len(_WORD_RE.findall(text)) > chunk_size
        if not isinstance(long_document, bool):
            raise TypeError("long_document must be a boolean or None")
        return long_document


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class _LoadedCheckpoint(NamedTuple):
    model: Any
    revision: str
    architecture: Optional[str]
    source: str


def load_gliner2_handle(
    model_id: str,
    *,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    device: Optional[str] = None,
    revision: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    license_id: Optional[str] = None,
) -> GLiNER2Handle:
    """Load a GLiNER2.5 checkpoint through ``gliner2.AutoExtractor``.

    The checkpoint is resolved to a local directory first: an existing
    directory is used as is, and a repository id is looked up in the Hugging
    Face cache before any download. Loaded models are cached per argument set;
    use :func:`clear_gliner2_cache` to release them.

    Args:
        model_id: Hugging Face repository id or local checkpoint directory.
        cache_dir: Optional Hugging Face cache directory.
        token: Optional Hub token for gated or private repositories.
        device: Optional torch device such as ``"cpu"`` or ``"cuda"``.
        revision: Optional branch, tag or commit to pin.
        local_files_only: Never download. ``None`` follows
            ``OPENMED_OFFLINE``.
        license_id: SPDX license to record when the checkpoint is not in
            :data:`GLINER2_CHECKPOINTS`.

    Returns:
        A :class:`GLiNER2Handle` recording the model id, resolved revision,
        license and architecture.

    Raises:
        MissingDependencyError: If ``gliner2[local]`` is not installed.
        OfflineModeError: If the snapshot is not cached and downloads are
            disabled.
        ModelLoadError: If the checkpoint is incomplete, requests remote code,
            or ships no safetensors weights.
    """

    ensure_gliner2_available()
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if local_files_only is not None and not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a boolean or None")
    local_only = is_local_only() if local_files_only is None else local_files_only
    resolved_device = None if device in (None, "", "auto") else device
    loaded = _load_model(
        model_id,
        cache_dir or None,
        token or None,
        resolved_device,
        revision or None,
        local_only,
    )
    spec = get_gliner2_checkpoint(model_id)
    return GLiNER2Handle(
        model_id=model_id,
        model=loaded.model,
        revision=loaded.revision,
        license_id=license_id or (spec.license_id if spec is not None else None),
        architecture=loaded.architecture,
        source=loaded.source,
    )


def clear_gliner2_cache() -> None:
    """Clear cached GLiNER2 model instances."""

    _load_model.cache_clear()  # type: ignore[attr-defined]


@lru_cache(maxsize=4)
def _load_model(
    model_id: str,
    cache_dir: Optional[str],
    token: Optional[str],
    device: Optional[str],
    revision: Optional[str],
    local_only: bool,
) -> _LoadedCheckpoint:
    ensure_gliner2_available()
    if os.path.isdir(model_id):
        directory = model_id
        resolved_revision = revision or _local_revision(directory)
        source = "local"
    else:
        directory = _resolve_snapshot(
            model_id,
            cache_dir=cache_dir,
            token=token,
            revision=revision,
            local_only=local_only,
        )
        resolved_revision = _snapshot_revision(directory) or revision or "unknown"
        source = "hub"
    _verify_checkpoint(directory, model_id)

    module = importlib.import_module(_PRIMARY_IMPORT)
    extractor = getattr(module, "AutoExtractor", None)
    if extractor is None or not hasattr(extractor, "from_pretrained"):
        raise MissingDependencyError(
            dependency="gliner2>=2.1", instruction=_GLINER2_HINT
        )
    load_kwargs: dict[str, Any] = {}
    if device:
        load_kwargs["map_location"] = device
    model = extractor.from_pretrained(directory, **load_kwargs)
    if callable(getattr(model, "eval", None)):
        model.eval()
    architecture = getattr(model, "architecture", None)
    return _LoadedCheckpoint(
        model=model,
        revision=resolved_revision,
        architecture=architecture if isinstance(architecture, str) else None,
        source=source,
    )


def _resolve_snapshot(
    model_id: str,
    *,
    cache_dir: Optional[str],
    token: Optional[str],
    revision: Optional[str],
    local_only: bool,
) -> str:
    hub = importlib.import_module("huggingface_hub")
    snapshot_download = hub.snapshot_download
    not_cached = _local_entry_errors()
    options: dict[str, Any] = {
        "repo_id": model_id,
        "repo_type": "model",
        "revision": revision,
        "allow_patterns": list(_SNAPSHOT_PATTERNS),
    }
    if cache_dir:
        options["cache_dir"] = cache_dir
    if token:
        options["token"] = token

    try:
        cached = snapshot_download(local_files_only=True, **options)
    except not_cached:
        cached = None
    if cached and _snapshot_complete(cached):
        return str(cached)
    if local_only:
        raise OfflineModeError(
            f"{model_id} is not fully cached and offline mode blocks the download. "
            "Pre-fetch it once with offline mode disabled, or pass a local "
            "checkpoint directory."
        )
    return str(snapshot_download(local_files_only=False, **options))


def _local_entry_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [FileNotFoundError]
    try:
        hub_errors = importlib.import_module("huggingface_hub.errors")
    except ImportError:  # pragma: no cover - older huggingface_hub layouts
        hub_errors = None
    candidate = getattr(hub_errors, "LocalEntryNotFoundError", None)
    if isinstance(candidate, type) and issubclass(candidate, BaseException):
        errors.append(candidate)
    return tuple(errors)


def _snapshot_complete(directory: str) -> bool:
    return (
        all(os.path.isfile(os.path.join(directory, name)) for name in _CONFIG_FILES)
        and os.path.isfile(os.path.join(directory, _WEIGHTS_FILE))
        and any(
            os.path.isfile(os.path.join(directory, name)) for name in _TOKENIZER_FILES
        )
    )


def _verify_checkpoint(directory: str, model_id: str) -> None:
    name = _public_name(model_id)
    missing = [
        file_name
        for file_name in _CONFIG_FILES
        if not os.path.isfile(os.path.join(directory, file_name))
    ]
    if not any(
        os.path.isfile(os.path.join(directory, file_name))
        for file_name in _TOKENIZER_FILES
    ):
        missing.append("tokenizer.json")
    if missing:
        raise ModelLoadError(
            "GLiNER2 checkpoint is missing required files: " + ", ".join(missing),
            model_name=name,
        )
    if not os.path.isfile(os.path.join(directory, _WEIGHTS_FILE)):
        raise ModelLoadError(
            "GLiNER2 checkpoints must ship model.safetensors; pickle-only "
            "weights are refused",
            model_name=name,
        )
    for file_name in _REMOTE_CODE_FILES:
        path = os.path.join(directory, file_name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            raise ModelLoadError(
                f"GLiNER2 checkpoint file {file_name} is not valid JSON",
                model_name=name,
            ) from None
        if _requests_remote_code(payload):
            raise ModelLoadError(
                f"GLiNER2 checkpoint file {file_name} requests remote code "
                "(auto_map); OpenMed does not execute checkpoint code",
                model_name=name,
            )


def _requests_remote_code(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        if "auto_map" in payload:
            return True
        return any(_requests_remote_code(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_requests_remote_code(value) for value in payload)
    return False


def _snapshot_revision(directory: str) -> Optional[str]:
    name = os.path.basename(os.path.normpath(directory))
    return name if _COMMIT_RE.fullmatch(name) else None


def _local_revision(directory: str) -> str:
    commit = _snapshot_revision(directory)
    if commit is not None and (
        os.path.basename(os.path.dirname(os.path.normpath(directory))) == "snapshots"
    ):
        return commit
    digest = hashlib.sha256()
    for file_name in _CONFIG_FILES:
        path = os.path.join(directory, file_name)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                digest.update(file_name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(handle.read())
    return f"local-config-sha256:{digest.hexdigest()}"


def _public_name(model_id: str) -> str:
    if os.path.isdir(model_id):
        return "local/" + (os.path.basename(os.path.normpath(model_id)) or "checkpoint")
    return model_id


# ---------------------------------------------------------------------------
# Payload conversion
# ---------------------------------------------------------------------------


class _TaskPlan(NamedTuple):
    labels: tuple[str, ...]
    model_labels: Union[list[str], dict[str, str]]
    multi_label: bool
    threshold: float


def _entity_type() -> type:
    from ..infer import Entity

    return Entity


def _require_text(text: Any) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be blank")
    return text


def _require_name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} cannot contain control characters")
    return value


def _require_probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _label_names(labels: Any, name: str) -> tuple[str, ...]:
    if isinstance(labels, Mapping):
        values = tuple(labels)
        for description in labels.values():
            if not isinstance(description, str):
                raise TypeError(f"{name} descriptions must be strings")
    elif isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise TypeError(f"{name} must be a sequence or a mapping")
    else:
        values = tuple(labels)
    if not values:
        raise ValueError(f"{name} must not be empty")
    checked = tuple(_require_name(value, name) for value in values)
    if len(set(checked)) != len(checked):
        raise ValueError(f"{name} must be unique")
    return checked


def _model_labels(labels: LabelSpec) -> Union[list[str], dict[str, str]]:
    if isinstance(labels, Mapping):
        return {str(key): str(value) for key, value in labels.items()}
    return [str(label) for label in labels]


def _classification_plans(tasks: Any) -> dict[str, _TaskPlan]:
    if not isinstance(tasks, Mapping) or not tasks:
        raise ValueError("tasks must be a non-empty mapping")
    plans: dict[str, _TaskPlan] = {}
    for task, config in tasks.items():
        _require_name(task, "task name")
        multi_label = False
        threshold = 0.5
        if isinstance(config, Mapping) and "labels" in config:
            unknown = set(config) - _TASK_OPTIONS
            if unknown:
                raise ValueError("classification task options are not supported")
            labels = config["labels"]
            multi_label = config.get("multi_label", False)
            if not isinstance(multi_label, bool):
                raise TypeError("multi_label must be a boolean")
            threshold = _require_probability(
                config.get("cls_threshold", 0.5), "cls_threshold"
            )
        else:
            labels = config
        names = _label_names(labels, "classification labels")
        if not multi_label and len(names) < 2:
            raise ValueError("single-label tasks need at least two labels")
        plans[task] = _TaskPlan(
            labels=names,
            model_labels=_model_labels(labels),
            multi_label=multi_label,
            threshold=threshold,
        )
    return plans


def _convert_classification(
    task: str, value: Any, plan: _TaskPlan
) -> GLiNER2Classification:
    items = [value] if isinstance(value, Mapping) else value
    if not isinstance(items, list):
        raise GLiNER2OutputError("GLiNER2 classification result is malformed")
    scores: dict[str, float] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise GLiNER2OutputError("GLiNER2 classification result is malformed")
        label = item.get("label")
        if label not in plan.labels:
            raise GLiNER2OutputError("GLiNER2 returned an unrequested label")
        scores[label] = _score(item.get("confidence"))
    if set(scores) != set(plan.labels):
        raise GLiNER2OutputError("GLiNER2 did not score every requested label")
    return GLiNER2Classification(
        task=task,
        scores=tuple(GLiNER2LabelScore(label, scores[label]) for label in plan.labels),
        multi_label=plan.multi_label,
        threshold=plan.threshold,
    )


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GLiNER2OutputError("GLiNER2 score is not a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise GLiNER2OutputError("GLiNER2 score is outside [0, 1]")
    return result


def _offset(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise GLiNER2OutputError("GLiNER2 offsets must be non-negative integers")
    return value


def _span_bounds(source: str, item: Mapping[str, Any]) -> Optional[tuple[int, int]]:
    """Return trimmed offsets when ``item`` is a source span, else ``None``."""

    if "start" not in item or "end" not in item:
        return None
    start, end = _offset(item["start"]), _offset(item["end"])
    if end > len(source) or start >= end:
        raise GLiNER2OutputError("GLiNER2 offsets fall outside the source text")
    span_text = item.get("text")
    if isinstance(span_text, str) and span_text.strip() != source[start:end].strip():
        return None
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    if start >= end:
        raise GLiNER2OutputError("GLiNER2 returned an empty span")
    return start, end


def _span_entity(
    source: str,
    item: Any,
    label: str,
    *,
    group: Optional[str] = None,
    default_score: Optional[float] = None,
) -> "Entity":
    if not isinstance(item, Mapping):
        raise GLiNER2OutputError("GLiNER2 span result is malformed")
    bounds = _span_bounds(source, item)
    if bounds is None:
        raise GLiNER2OutputError("GLiNER2 span text does not match its offsets")
    start, end = bounds
    confidence = item.get("confidence", default_score)
    entity_type = _entity_type()
    return entity_type(
        text=source[start:end],
        start=start,
        end=end,
        label=label,
        score=_score(confidence),
        group=group,
    )


def _as_items(value: Any) -> list[Any]:
    if value is None or value == "" or value == {}:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _convert_entities(
    source: str, payload: Any, labels: tuple[str, ...]
) -> list["Entity"]:
    if not isinstance(payload, Mapping):
        raise GLiNER2OutputError("GLiNER2 entity payload is not a mapping")
    groups = payload.get("entities", {})
    if groups in (None, []):
        groups = {}
    if not isinstance(groups, Mapping):
        raise GLiNER2OutputError("GLiNER2 entity payload is malformed")
    best: dict[tuple[int, int, str], Any] = {}
    for label, value in groups.items():
        if label not in labels:
            raise GLiNER2OutputError("GLiNER2 returned an unrequested entity label")
        for item in _as_items(value):
            entity = _span_entity(source, item, label)
            key = (entity.start, entity.end, label)
            if key not in best or entity.score > best[key].score:
                best[key] = entity
    return [best[key] for key in sorted(best)]


def _convert_relations(
    source: str, payload: Any, relation_types: tuple[str, ...]
) -> list[GLiNER2Relation]:
    if not isinstance(payload, Mapping):
        raise GLiNER2OutputError("GLiNER2 relation payload is not a mapping")
    groups = payload.get("relation_extraction", {})
    if groups in (None, []):
        groups = {}
    if not isinstance(groups, Mapping):
        raise GLiNER2OutputError("GLiNER2 relation payload is malformed")
    relations: list[GLiNER2Relation] = []
    for relation, instances in groups.items():
        if relation not in relation_types:
            raise GLiNER2OutputError("GLiNER2 returned an unrequested relation type")
        for instance in _as_items(instances):
            if not isinstance(instance, Mapping):
                raise GLiNER2OutputError("GLiNER2 relation result is malformed")
            head = _span_entity(source, instance.get("head"), relation, group="head")
            tail = _span_entity(source, instance.get("tail"), relation, group="tail")
            confidence = instance.get("confidence")
            score = (
                _score(confidence)
                if confidence is not None
                else min(head.score, tail.score)
            )
            relations.append(
                GLiNER2Relation(relation=relation, head=head, tail=tail, score=score)
            )
    relations.sort(
        key=lambda item: (
            item.head.start,
            item.head.end,
            item.tail.start,
            item.tail.end,
            item.relation,
        )
    )
    return relations


def _convert_records(
    source: str, payload: Any, structures: list[str]
) -> dict[str, tuple[GLiNER2Record, ...]]:
    if not isinstance(payload, Mapping):
        raise GLiNER2OutputError("GLiNER2 JSON payload is not a mapping")
    records: dict[str, tuple[GLiNER2Record, ...]] = {}
    for name in structures:
        instances = payload.get(name, [])
        converted = []
        for instance in _as_items(instances):
            if not isinstance(instance, Mapping):
                raise GLiNER2OutputError("GLiNER2 structure result is malformed")
            converted.append(_convert_record(source, name, instance))
        records[name] = tuple(converted)
    return records


def _convert_record(
    source: str, structure: str, instance: Mapping[str, Any]
) -> GLiNER2Record:
    fields: dict[str, tuple[Any, ...]] = {}
    choices: dict[str, tuple[GLiNER2LabelScore, ...]] = {}
    score: Optional[float] = None
    for field_name, value in instance.items():
        if field_name == "confidence" and not isinstance(value, (Mapping, list)):
            score = _score(value) if value is not None else None
            continue
        spans: list[Any] = []
        labels: list[GLiNER2LabelScore] = []
        for item in _as_items(value):
            if not isinstance(item, Mapping):
                raise GLiNER2OutputError("GLiNER2 field result is malformed")
            if _span_bounds(source, item) is not None:
                spans.append(_span_entity(source, item, field_name))
                continue
            label = item.get("text", item.get("label"))
            if not isinstance(label, str) or not label:
                raise GLiNER2OutputError("GLiNER2 choice field is malformed")
            labels.append(GLiNER2LabelScore(label, _score(item.get("confidence"))))
        if labels:
            choices[field_name] = tuple(labels)
            if spans:
                raise GLiNER2OutputError("GLiNER2 field mixes spans and choices")
        else:
            fields[field_name] = tuple(
                sorted(spans, key=lambda entity: (entity.start, entity.end))
            )
    return GLiNER2Record(
        structure=structure,
        fields=MappingProxyType(fields),
        choices=MappingProxyType(choices),
        score=score,
    )


__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "GLINER2_CHECKPOINTS",
    "GLINER2_TASKS",
    "GLiNER2CheckpointSpec",
    "GLiNER2Classification",
    "GLiNER2Handle",
    "GLiNER2LabelScore",
    "GLiNER2OutputError",
    "GLiNER2Record",
    "GLiNER2Relation",
    "clear_gliner2_cache",
    "ensure_gliner2_available",
    "get_gliner2_checkpoint",
    "is_gliner2_available",
    "list_gliner2_checkpoints",
    "load_gliner2_handle",
]
