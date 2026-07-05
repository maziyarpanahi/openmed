"""Stateful multi-step workflow orchestration for the OpenMed MCP server."""

from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

from openmed.clinical.exporters.codeable_concept_simple import (
    codeable_concept,
    coding,
)
from openmed.clinical.exporters.fhir import to_bundle

WorkflowStepExecutor = Callable[..., Any]
StringDeidentifier = Callable[[str], str]


class WorkflowError(ValueError):
    """Base class for workflow orchestration errors."""


class WorkflowValidationError(WorkflowError):
    """Raised when a workflow declaration is invalid."""


class TransientWorkflowError(WorkflowError):
    """Raised by step adapters to signal a retryable transient failure."""


CLINICAL_PIPELINE_SCHEMA_VERSION = "openmed.clinical_pipeline.v1"
CLINICAL_STAGE_ORDER = (
    "detect",
    "context",
    "sections",
    "relations",
    "ground",
    "export",
    "risk",
)
_CLINICAL_STAGE_INDEX = {
    stage: index for index, stage in enumerate(CLINICAL_STAGE_ORDER)
}


class ClinicalStageOrderError(WorkflowValidationError):
    """Raised when a declared clinical workflow violates stage ordering."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        stages: Sequence[Any],
        stage: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        merged_details: dict[str, Any] = {
            "stages": [str(item) for item in stages],
            "allowed_order": list(CLINICAL_STAGE_ORDER),
        }
        if details:
            merged_details.update(dict(details))
        self.error = {
            "code": code,
            "message": message,
            "stage": stage,
            "details": merged_details,
        }


def validate_clinical_stage_order(stages: Sequence[Any]) -> tuple[str, ...]:
    """Return normalized clinical stages, or raise for unsafe order."""
    if isinstance(stages, (str, bytes, bytearray)) or not isinstance(stages, Sequence):
        raise ClinicalStageOrderError(
            code="invalid_stage_list",
            message="Clinical pipeline stages must be a non-empty list.",
            stages=(),
        )
    if not stages:
        raise ClinicalStageOrderError(
            code="invalid_stage_list",
            message="Clinical pipeline stages must be a non-empty list.",
            stages=stages,
        )

    normalized: list[str] = []
    seen: set[str] = set()
    previous_stage: str | None = None
    previous_index = -1
    for raw_stage in stages:
        stage = str(raw_stage).strip().lower()
        if stage not in _CLINICAL_STAGE_INDEX:
            raise ClinicalStageOrderError(
                code="unknown_stage",
                message=f"Unknown clinical pipeline stage: {raw_stage!r}.",
                stages=stages,
                stage=stage or None,
            )
        if stage in seen:
            raise ClinicalStageOrderError(
                code="duplicate_stage",
                message=f"Duplicate clinical pipeline stage: {stage}.",
                stages=stages,
                stage=stage,
            )

        stage_index = _CLINICAL_STAGE_INDEX[stage]
        if stage_index < previous_index:
            raise ClinicalStageOrderError(
                code="invalid_stage_order",
                message=(
                    f"Clinical pipeline stage {stage!r} cannot run after "
                    f"{previous_stage!r}."
                ),
                stages=stages,
                stage=stage,
                details={"previous_stage": previous_stage},
            )

        normalized.append(stage)
        seen.add(stage)
        previous_stage = stage
        previous_index = stage_index

    return tuple(normalized)


def plan_clinical_pipeline(
    stages: Sequence[Any],
    *,
    stage_callbacks: Mapping[str, Callable[[], Any]] | None = None,
) -> dict[str, Any]:
    """Validate and optionally dry-run a clinical pipeline stage declaration."""
    try:
        normalized_stages = validate_clinical_stage_order(stages)
    except ClinicalStageOrderError as exc:
        response_stages: Sequence[Any]
        if isinstance(stages, (str, bytes, bytearray)) or not isinstance(
            stages, Sequence
        ):
            response_stages = ()
        else:
            response_stages = stages
        return _clinical_pipeline_payload(
            status="rejected",
            stages=response_stages,
            error=exc.error,
        )

    callbacks = dict(stage_callbacks or {})
    artifacts: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    for stage in normalized_stages:
        callback = callbacks.get(stage)
        if callback is None:
            continue
        artifacts[stage] = callback()
        trace.append({"stage": stage, "status": "completed"})

    return _clinical_pipeline_payload(
        status="completed" if trace else "planned",
        stages=normalized_stages,
        artifacts=artifacts,
        final_output=artifacts.get(normalized_stages[-1]) if artifacts else None,
        trace=trace,
    )


def _clinical_pipeline_payload(
    *,
    status: str,
    stages: Sequence[Any],
    artifacts: Mapping[str, Any] | None = None,
    final_output: Any = None,
    error: Mapping[str, Any] | None = None,
    trace: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "schema_version": CLINICAL_PIPELINE_SCHEMA_VERSION,
        "status": status,
        "stages": [str(stage).strip().lower() for stage in stages],
        "artifacts": dict(artifacts or {}),
        "final_output": copy.deepcopy(final_output),
        "error": dict(error) if error is not None else None,
        "trace": [dict(item) for item in trace],
    }


@dataclass
class WorkflowStoredValue:
    """One server-side intermediate value stored behind an opaque handle."""

    handle: str
    session_id: str
    workflow_id: str
    step_id: str
    value: Any
    created_at: float = field(default_factory=time.time)


class WorkflowStateStore:
    """In-memory session-scoped workflow state.

    The store intentionally keeps values process-local and addressable only by
    opaque handles. Callers receive handles and trace metadata, not intermediate
    payloads that may contain PHI.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def ensure_session(self, session_id: Optional[str] = None) -> str:
        """Return an existing or newly-created session id."""
        resolved = _safe_identifier(session_id, prefix="wf-session")
        with self._lock:
            self._sessions.setdefault(
                resolved,
                {"handles": {}, "completed_steps": {}},
            )
        return resolved

    def put(
        self,
        *,
        session_id: str,
        workflow_id: str,
        step_id: str,
        value: Any,
    ) -> str:
        """Store *value* and return its opaque handle."""
        session_id = self.ensure_session(session_id)
        handle = f"wf-handle-{uuid.uuid4().hex}"
        stored = WorkflowStoredValue(
            handle=handle,
            session_id=session_id,
            workflow_id=workflow_id,
            step_id=step_id,
            value=copy.deepcopy(value),
        )
        with self._lock:
            session = self._sessions[session_id]
            session["handles"][handle] = stored
            session["completed_steps"][(workflow_id, step_id)] = handle
        return handle

    def get(self, session_id: str, handle: str) -> Any:
        """Return a deep copy of the value referenced by *handle*."""
        with self._lock:
            try:
                stored = self._sessions[session_id]["handles"][handle]
            except KeyError as exc:
                raise WorkflowValidationError("Unknown workflow state handle") from exc
            return copy.deepcopy(stored.value)

    def completed_handle(
        self,
        *,
        session_id: str,
        workflow_id: str,
        step_id: str,
    ) -> Optional[str]:
        """Return the stored handle for a completed workflow step, if any."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            handle = session["completed_steps"].get((workflow_id, step_id))
            if handle in session["handles"]:
                return handle
            return None


@dataclass
class _ResolvedInput:
    value: Any
    handles: tuple[str, ...] = ()


class WorkflowRunner:
    """Execute declarative OpenMed workflows against a session state store."""

    schema_version = "openmed.workflow.v1"

    def __init__(
        self,
        *,
        store: WorkflowStateStore,
        executors: Mapping[str, WorkflowStepExecutor],
        deidentifier: Optional[StringDeidentifier] = None,
    ) -> None:
        self.store = store
        self.executors = dict(executors)
        self.deidentifier = deidentifier or _conservative_deidentify_string

    def run(
        self,
        pipeline: Mapping[str, Any],
        *,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run a declared workflow and return PHI-safe structured metadata."""
        declaration = self._validate_pipeline(pipeline)
        resolved_session_id = self.store.ensure_session(
            session_id or declaration.get("session_id")
        )
        resolved_workflow_id = _safe_identifier(
            workflow_id or declaration.get("workflow_id"),
            prefix="wf",
        )
        steps = list(declaration["steps"])

        trace: list[dict[str, Any]] = []
        step_handles: dict[str, str] = {}
        returned_outputs: dict[str, Any] = {}
        final_step: Optional[Mapping[str, Any]] = None
        failed = False

        for step in steps:
            step_id = str(step["id"])
            tool_name = str(step["tool"])
            previously_completed = self.store.completed_handle(
                session_id=resolved_session_id,
                workflow_id=resolved_workflow_id,
                step_id=step_id,
            )
            if previously_completed is not None:
                step_handles[step_id] = previously_completed
                final_step = step
                trace.append(
                    _trace_entry(
                        step=step,
                        status="resumed",
                        duration_ms=0.0,
                        retry_count=0,
                        attempt_count=0,
                        input_handles=(),
                        output_handle=previously_completed,
                        resumed=True,
                    )
                )
                if _should_return_output(step):
                    raw_output = self.store.get(
                        resolved_session_id, previously_completed
                    )
                    returned_outputs[step_id] = self._egress_value(
                        raw_output,
                        allow_raw=_allow_raw_output(step),
                    )
                continue

            if not self._condition_allows(
                step,
                session_id=resolved_session_id,
                step_handles=step_handles,
            ):
                trace.append(
                    _trace_entry(
                        step=step,
                        status="skipped",
                        duration_ms=0.0,
                        retry_count=0,
                        attempt_count=0,
                        input_handles=(),
                        output_handle=None,
                    )
                )
                continue

            resolved_inputs = self._resolve_step_inputs(
                step,
                session_id=resolved_session_id,
                step_handles=step_handles,
            )
            executor = self._executor(tool_name)
            max_attempts = _max_attempts(step)
            started = time.perf_counter()
            attempt_count = 0
            last_error_type: Optional[str] = None

            while attempt_count < max_attempts:
                attempt_count += 1
                try:
                    output = executor(**resolved_inputs.value)
                    handle = self.store.put(
                        session_id=resolved_session_id,
                        workflow_id=resolved_workflow_id,
                        step_id=step_id,
                        value=output,
                    )
                    step_handles[step_id] = handle
                    final_step = step
                    duration_ms = (time.perf_counter() - started) * 1000
                    retry_count = attempt_count - 1
                    trace.append(
                        _trace_entry(
                            step=step,
                            status="completed",
                            duration_ms=duration_ms,
                            retry_count=retry_count,
                            attempt_count=attempt_count,
                            input_handles=resolved_inputs.handles,
                            output_handle=handle,
                        )
                    )
                    if _should_return_output(step):
                        returned_outputs[step_id] = self._egress_value(
                            output,
                            allow_raw=_allow_raw_output(step),
                        )
                    break
                except Exception as exc:  # noqa: BLE001 - retries wrap step adapters.
                    last_error_type = type(exc).__name__
                    if attempt_count >= max_attempts:
                        duration_ms = (time.perf_counter() - started) * 1000
                        trace.append(
                            _trace_entry(
                                step=step,
                                status="failed",
                                duration_ms=duration_ms,
                                retry_count=attempt_count - 1,
                                attempt_count=attempt_count,
                                input_handles=resolved_inputs.handles,
                                output_handle=None,
                                error_type=last_error_type,
                            )
                        )
                        failed = True
                        break

            if failed:
                break

        final_handle = step_handles.get(str(final_step["id"])) if final_step else None
        final_output = None
        if final_handle is not None and final_step is not None:
            raw_final = self.store.get(resolved_session_id, final_handle)
            final_output = self._egress_value(
                raw_final,
                allow_raw=_allow_raw_output(final_step),
            )

        return {
            "schema_version": self.schema_version,
            "session_id": resolved_session_id,
            "workflow_id": resolved_workflow_id,
            "status": "failed" if failed else "completed",
            "handles": dict(step_handles),
            "final_handle": final_handle,
            "final_output": final_output,
            "outputs": returned_outputs,
            "trace": trace,
        }

    def _validate_pipeline(self, pipeline: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(pipeline, Mapping):
            raise WorkflowValidationError("Workflow pipeline must be an object")
        steps = pipeline.get("steps")
        if not isinstance(steps, list) or not steps:
            raise WorkflowValidationError("Workflow pipeline requires non-empty steps")

        seen_ids: set[str] = set()
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise WorkflowValidationError("Workflow steps must be objects")
            step_id = step.get("id")
            tool_name = step.get("tool")
            if not isinstance(step_id, str) or not step_id.strip():
                raise WorkflowValidationError(f"Workflow step {index} requires an id")
            if step_id in seen_ids:
                raise WorkflowValidationError(f"Duplicate workflow step id: {step_id}")
            seen_ids.add(step_id)
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise WorkflowValidationError(
                    f"Workflow step {step_id} requires a tool"
                )
            if tool_name not in self.executors:
                raise WorkflowValidationError(f"Unsupported workflow tool: {tool_name}")
            inputs = step.get("inputs", {})
            if not isinstance(inputs, Mapping):
                raise WorkflowValidationError(
                    f"Workflow step {step_id} inputs must be an object"
                )

        return pipeline

    def _executor(self, tool_name: str) -> WorkflowStepExecutor:
        try:
            return self.executors[tool_name]
        except KeyError as exc:
            raise WorkflowValidationError(
                f"Unsupported workflow tool: {tool_name}"
            ) from exc

    def _resolve_step_inputs(
        self,
        step: Mapping[str, Any],
        *,
        session_id: str,
        step_handles: Mapping[str, str],
    ) -> _ResolvedInput:
        raw_inputs = step.get("inputs", {})
        resolved_inputs: dict[str, Any] = {}
        handles: list[str] = []
        for key, value in raw_inputs.items():
            resolved = self._resolve_value(
                value,
                session_id=session_id,
                step_handles=step_handles,
            )
            resolved_inputs[str(key)] = resolved.value
            handles.extend(resolved.handles)
        if not isinstance(resolved_inputs, MutableMapping):
            raise WorkflowValidationError(
                "Resolved workflow step inputs must be an object"
            )
        return _ResolvedInput(resolved_inputs, tuple(_dedupe(handles)))

    def _resolve_value(
        self,
        value: Any,
        *,
        session_id: str,
        step_handles: Mapping[str, str],
    ) -> _ResolvedInput:
        if isinstance(value, Mapping):
            if set(value.keys()) == {"value"}:
                return _ResolvedInput(copy.deepcopy(value["value"]))
            if "from_step" in value or "from_handle" in value:
                handle = self._resolve_handle(value, step_handles=step_handles)
                stored_value = self.store.get(session_id, handle)
                selected = _select_path(stored_value, value.get("path"))
                return _ResolvedInput(selected, (handle,))

            resolved_mapping: dict[str, Any] = {}
            handles: list[str] = []
            for key, child in value.items():
                resolved = self._resolve_value(
                    child,
                    session_id=session_id,
                    step_handles=step_handles,
                )
                resolved_mapping[str(key)] = resolved.value
                handles.extend(resolved.handles)
            return _ResolvedInput(resolved_mapping, tuple(_dedupe(handles)))

        if isinstance(value, list):
            resolved_items: list[Any] = []
            handles = []
            for item in value:
                resolved = self._resolve_value(
                    item,
                    session_id=session_id,
                    step_handles=step_handles,
                )
                resolved_items.append(resolved.value)
                handles.extend(resolved.handles)
            return _ResolvedInput(resolved_items, tuple(_dedupe(handles)))

        return _ResolvedInput(copy.deepcopy(value))

    def _resolve_handle(
        self,
        binding: Mapping[str, Any],
        *,
        step_handles: Mapping[str, str],
    ) -> str:
        if "from_handle" in binding:
            handle = binding["from_handle"]
            if not isinstance(handle, str) or not handle:
                raise WorkflowValidationError("from_handle must be a non-empty string")
            return handle

        step_id = binding.get("from_step")
        if not isinstance(step_id, str) or not step_id:
            raise WorkflowValidationError("from_step must be a non-empty string")
        try:
            return step_handles[step_id]
        except KeyError as exc:
            raise WorkflowValidationError(
                f"Workflow step {step_id} has no completed output"
            ) from exc

    def _condition_allows(
        self,
        step: Mapping[str, Any],
        *,
        session_id: str,
        step_handles: Mapping[str, str],
    ) -> bool:
        condition = step.get("condition")
        if condition is None:
            return True
        if not isinstance(condition, Mapping):
            raise WorkflowValidationError("Workflow condition must be an object")

        binding = {
            key: condition[key]
            for key in ("from_step", "from_handle", "path", "value")
            if key in condition
        }
        if not binding:
            raise WorkflowValidationError("Workflow condition requires a binding")
        resolved = self._resolve_value(
            binding,
            session_id=session_id,
            step_handles=step_handles,
        ).value
        operator = str(condition.get("operator", "truthy"))
        expected = condition.get("equals", condition.get("value"))

        if operator == "truthy":
            return bool(resolved)
        if operator == "exists":
            return resolved is not None
        if operator == "empty":
            return not bool(resolved)
        if operator == "equals":
            return resolved == expected
        if operator == "not_equals":
            return resolved != expected
        if operator == "contains":
            try:
                return expected in resolved
            except TypeError:
                return False
        raise WorkflowValidationError(f"Unsupported workflow condition: {operator}")

    def _egress_value(self, value: Any, *, allow_raw: bool) -> Any:
        if allow_raw:
            return copy.deepcopy(value)
        if isinstance(value, str):
            return self.deidentifier(value)
        if isinstance(value, Mapping):
            return {
                str(key): self._egress_value(child, allow_raw=False)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [self._egress_value(item, allow_raw=False) for item in value]
        if isinstance(value, tuple):
            return [self._egress_value(item, allow_raw=False) for item in value]
        return copy.deepcopy(value)


def builtin_workflow_step_executors() -> dict[str, WorkflowStepExecutor]:
    """Return workflow-only step adapters that do not require model execution."""
    return {
        "openmed_map_concepts": openmed_map_concepts,
        "openmed_export_fhir": openmed_export_fhir,
    }


def openmed_map_concepts(
    entities: Optional[list[Mapping[str, Any]]] = None,
    concepts: Optional[list[Mapping[str, Any]]] = None,
    analysis: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Map extracted entities or supplied coded concepts into CodeableConcepts."""
    source_items = concepts
    if source_items is None and analysis is not None:
        source_items = _extract_entity_rows(analysis)
    if source_items is None:
        source_items = entities or []

    mapped: list[dict[str, Any]] = []
    for index, item in enumerate(source_items):
        if not isinstance(item, Mapping):
            continue
        text = _first_string(item, ("text", "word", "display", "mention", "value"))
        system = _first_string(item, ("system", "vocabulary", "vocabulary_id"))
        code = _first_string(item, ("code", "concept_code", "id"))
        display = _first_string(item, ("display", "preferred_term", "name")) or text

        concept: dict[str, Any] = {
            "id": str(item.get("id") or f"concept-{index + 1}"),
            "text": text,
            "label": _first_string(item, ("label", "entity_type", "type")),
        }
        if system and code:
            concept["codeable_concept"] = codeable_concept(
                [coding(system, code, display)],
                text=text,
            )
        mapped.append(concept)

    return {"concepts": mapped, "count": len(mapped)}


def openmed_export_fhir(
    concepts: Optional[list[Mapping[str, Any]]] = None,
    mapped_concepts: Optional[Mapping[str, Any]] = None,
    doc_id: str = "workflow",
    resource_type: str = "Observation",
    bundle_type: str = "collection",
) -> dict[str, Any]:
    """Export mapped concepts into a small FHIR Bundle."""
    concept_rows = concepts
    if concept_rows is None and mapped_concepts is not None:
        raw_rows = mapped_concepts.get("concepts", [])
        concept_rows = raw_rows if isinstance(raw_rows, list) else []
    concept_rows = concept_rows or []

    resources = [
        _concept_to_resource(item, index=index, resource_type=resource_type)
        for index, item in enumerate(concept_rows)
        if isinstance(item, Mapping)
    ]
    bundle = to_bundle(resources, doc_id=doc_id, bundle_type=bundle_type)
    return {"bundle": bundle, "resource_count": len(resources)}


def _concept_to_resource(
    concept: Mapping[str, Any],
    *,
    index: int,
    resource_type: str,
) -> dict[str, Any]:
    normalized_type = resource_type.strip() or "Observation"
    resource_id = f"{normalized_type.lower()}-{index + 1}"
    code = concept.get("codeable_concept")
    if not isinstance(code, Mapping):
        text = _first_string(concept, ("text", "display", "label"))
        code = {"text": text} if text else {"text": "Uncoded concept"}

    if normalized_type == "Condition":
        return {
            "resourceType": "Condition",
            "id": resource_id,
            "code": dict(code),
        }

    return {
        "resourceType": normalized_type,
        "id": resource_id,
        "status": "final",
        "code": dict(code),
    }


def _extract_entity_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("entities", "pii_entities", "results", "concepts"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _first_string(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _select_path(value: Any, path: Any) -> Any:
    if path is None or path == "":
        return value
    selected = value
    for segment in _path_segments(path):
        if isinstance(selected, Mapping):
            selected = selected[segment]
        elif isinstance(selected, list):
            selected = selected[int(segment)]
        else:
            raise WorkflowValidationError("Workflow binding path cannot be resolved")
    return copy.deepcopy(selected)


def _path_segments(path: Any) -> tuple[str, ...]:
    if isinstance(path, (list, tuple)):
        return tuple(str(segment) for segment in path)
    raw = str(path)
    if raw.startswith("/"):
        return tuple(
            segment.replace("~1", "/").replace("~0", "~")
            for segment in raw[1:].split("/")
        )
    normalized = raw.replace("[", ".").replace("]", "")
    return tuple(segment for segment in normalized.split(".") if segment)


def _max_attempts(step: Mapping[str, Any]) -> int:
    if "max_attempts" in step:
        return max(1, int(step["max_attempts"]))
    if "max_retries" in step:
        return max(1, int(step["max_retries"]) + 1)

    retry = step.get("retry")
    if isinstance(retry, Mapping):
        if "max_attempts" in retry:
            return max(1, int(retry["max_attempts"]))
        if "max_retries" in retry:
            return max(1, int(retry["max_retries"]) + 1)
    return 1


def _should_return_output(step: Mapping[str, Any]) -> bool:
    return bool(step.get("return_output", False))


def _allow_raw_output(step: Mapping[str, Any]) -> bool:
    return bool(step.get("allow_raw_output") or step.get("allow_raw"))


def _trace_entry(
    *,
    step: Mapping[str, Any],
    status: str,
    duration_ms: float,
    retry_count: int,
    attempt_count: int,
    input_handles: tuple[str, ...],
    output_handle: Optional[str],
    resumed: bool = False,
    error_type: Optional[str] = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "step_id": str(step["id"]),
        "tool": str(step["tool"]),
        "status": status,
        "duration_ms": round(duration_ms, 3),
        "retry_count": retry_count,
        "attempt_count": attempt_count,
        "input_handles": list(input_handles),
        "output_handle": output_handle,
    }
    if resumed:
        entry["resumed"] = True
    if error_type:
        entry["error_type"] = error_type
    return entry


def _safe_identifier(value: Optional[Any], *, prefix: str) -> str:
    if value is not None:
        normalized = str(value).strip()
        if normalized:
            return normalized
    return f"{prefix}-{uuid.uuid4().hex}"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _conservative_deidentify_string(value: str) -> str:
    if not value:
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return "[REDACTED_TEXT]"
    if isinstance(parsed, (dict, list)):
        return json.dumps(_mask_json_strings(parsed), sort_keys=True)
    return "[REDACTED_TEXT]"


def _mask_json_strings(value: Any) -> Any:
    if isinstance(value, str):
        return "[REDACTED_TEXT]" if value else value
    if isinstance(value, Mapping):
        return {str(key): _mask_json_strings(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_mask_json_strings(item) for item in value]
    return value
