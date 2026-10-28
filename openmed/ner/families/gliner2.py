"""Local GLiNER2 / GLiNER2.5 loading, native extraction and typed task helpers.

``load_gliner2_handle`` loads a checkpoint through ``gliner2.AutoExtractor``,
which selects the span or boundary architecture saved with the checkpoint and
never constructs the package's hosted API client. The returned handle serves
two layers:

* :meth:`GLiNER2Handle.predict_entities` is the span-list contract used by
  OpenMed inference and ``analyze_text``.
* The typed helpers (:meth:`~GLiNER2Handle.extract_entities`,
  :meth:`~GLiNER2Handle.classify_text`,
  :meth:`~GLiNER2Handle.extract_relations` and
  :meth:`~GLiNER2Handle.extract_json`) return OpenMed entities,
  classifications, relations and schema-driven records with absolute offsets.
  The GLiNER2.5 decision backend builds on ``classify_text``.

``import openmed`` never imports ``gliner2``; availability is probed through
the shared capability registry. Typed helpers run with the network blocked in
offline mode, results carry labels, offsets and scores, and error messages
never echo request text.
"""

from __future__ import annotations

import importlib
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union

from openmed.core.capabilities import is_backend_available
from openmed.core.offline import network_blocked_if_offline

from ..exceptions import MissingDependencyError
from .gliner import _resolve_native_device

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..infer import Entity

_PRIMARY_IMPORT = "gliner2"
_GLINER2_HINT = (
    "Install with `pip install openmed[gliner2]` "
    "(requires gliner2[local]>=2.1.0 for local span and boundary checkpoints)."
)
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

LabelSpec = Union[Sequence[str], Mapping[str, str]]


class GLiNER2OutputError(ValueError):
    """Raised when a GLiNER2 payload does not match the requested contract.

    Messages name the failed check only; they never contain request text, span
    text or label descriptions.
    """


def is_gliner2_available(force_refresh: bool = False) -> bool:
    """Return whether the GLiNER2 local inference stack is installed."""

    return is_backend_available("gliner2")


def ensure_gliner2_available() -> None:
    """Require GLiNER2's local runtime, including its optional ML stack."""

    if is_gliner2_available(force_refresh=False):
        return

    raise MissingDependencyError(
        dependency="gliner2[local]>=2.1.0", instruction=_GLINER2_HINT
    )


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


@dataclass
class GLiNER2Handle:
    """Local GLiNER2 extractor with native spans and typed multi-task helpers.

    Attributes:
        model_id: Repository id or local directory supplied by the caller.
        model: The ``gliner2`` extractor returned by ``AutoExtractor``.
        revision: Resolved checkpoint revision: the snapshot commit for Hub
            checkpoints, otherwise the requested revision when one was given.
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

    def predict_entities(
        self,
        text: str,
        labels: Optional[Sequence[str]] = None,
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Extract all occurrences with native offsets and confidence scores."""

        result = self.model.extract_entities(
            text,
            list(labels) if labels else [],
            threshold=threshold,
            include_spans=True,
            include_confidence=True,
            overlap_policy="disallow" if flat_ner else "allow",
            **kwargs,
        )
        if not isinstance(result, dict) or not isinstance(result.get("entities"), dict):
            raise ValueError("GLiNER2 returned an invalid entities mapping")
        entities = []
        for label, spans in result["entities"].items():
            if not isinstance(spans, list):
                raise ValueError("GLiNER2 returned an invalid span list")
            for span in spans:
                if not isinstance(span, dict) or not {
                    "text",
                    "start",
                    "end",
                    "confidence",
                }.issubset(span):
                    raise ValueError("GLiNER2 must return spans with confidence")
                entities.append({**span, "label": label, "score": span["confidence"]})
        return sorted(entities, key=lambda item: (item["start"], item["end"]))

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

    def predict_relations(
        self,
        text: str,
        labels: Optional[Sequence[str]] = None,
        *,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> Any:
        """Optional relation extraction hook if the model provides it."""
        ensure_gliner2_available()
        if hasattr(self.model, "extract_relations"):
            predict_kwargs = dict(kwargs)
            predict_kwargs.setdefault("threshold", threshold)
            return self.model.extract_relations(
                text, list(labels or []), **predict_kwargs
            )
        raise AttributeError("GLiNER2 checkpoint does not support relation extraction")

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


class _LoadedCheckpoint(NamedTuple):
    model: Any
    reference: str


def load_gliner2_handle(
    model_id: str,
    *,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    device: Optional[str] = None,
    local_files_only: bool = False,
    revision: Optional[str] = None,
    license_id: Optional[str] = None,
) -> GLiNER2Handle:
    """Load a local GLiNER2 extractor with cache, authentication and device options.

    ``AutoExtractor`` selects the span or boundary architecture from checkpoint
    configuration; it never constructs the package's hosted API client.

    Args:
        model_id: Hugging Face repository id or local checkpoint directory.
        cache_dir: Optional Hugging Face cache directory.
        token: Optional Hub token for gated or private repositories.
        device: Optional torch device such as ``"cpu"`` or ``"cuda"``.
        local_files_only: Never download; use the local cache only.
        revision: Optional branch, tag or commit to pin.
        license_id: SPDX license to record when the checkpoint is not in
            :data:`GLINER2_CHECKPOINTS`.

    Returns:
        A :class:`GLiNER2Handle` recording the model id, resolved revision,
        license and architecture.
    """

    ensure_gliner2_available()
    loaded = _load_model(
        model_id,
        cache_dir or None,
        token or None,
        _resolve_native_device(device),
        local_files_only,
        revision,
    )
    spec = get_gliner2_checkpoint(model_id)
    architecture = getattr(loaded.model, "architecture", None)
    return GLiNER2Handle(
        model_id=model_id,
        model=loaded.model,
        revision=_snapshot_revision(loaded.reference) or revision,
        license_id=license_id or (spec.license_id if spec is not None else None),
        architecture=architecture if isinstance(architecture, str) else None,
        source="local" if Path(model_id).is_dir() else "hub",
    )


def clear_gliner2_cache() -> None:
    """Clear cached GLiNER v2 model instances."""

    _load_model.cache_clear()  # type: ignore[attr-defined]


@lru_cache(maxsize=4)
def _load_model(
    model_id: str,
    cache_dir: Optional[str],
    token: Optional[str],
    device: Optional[str],
    local_files_only: bool,
    revision: Optional[str],
) -> _LoadedCheckpoint:
    ensure_gliner2_available()
    module = importlib.import_module(_PRIMARY_IMPORT)

    loader = getattr(module, "AutoExtractor", None)
    if loader is None:
        raise MissingDependencyError(
            dependency="gliner2[local]>=2.1.0",
            instruction=_GLINER2_HINT,
        )

    reference = model_id
    if not Path(reference).is_dir():
        hub = importlib.import_module("huggingface_hub")
        # AutoExtractor 2.1 consumes Hub options while resolving config but
        # does not forward them to the weight loader. Resolve one snapshot so
        # authentication, cache, offline mode and revision bind both stages.
        reference = hub.snapshot_download(
            repo_id=model_id,
            repo_type="model",
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.model",
                "*.txt",
                "*.vocab",
                "*.merges",
            ],
        )
    model = loader.from_pretrained(reference, local_files_only=True)
    if device:
        model = model.to(device)
    model.eval()
    return _LoadedCheckpoint(model=model, reference=str(reference))


def _snapshot_revision(directory: str) -> Optional[str]:
    """Return the commit hash of a Hugging Face snapshot directory, if any."""

    name = os.path.basename(os.path.normpath(directory))
    return name if _COMMIT_RE.fullmatch(name) else None


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
