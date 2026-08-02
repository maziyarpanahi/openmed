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
from openmed.core.schemas import OpenMedSpan
from openmed.mcp.tool_registry import (
    CLINICAL_STAGE_ORDER,
    validate_registered_tool_output,
)

WorkflowStepExecutor = Callable[..., Any]
StringDeidentifier = Callable[[str], str]
ClinicalStageHandler = Callable[
    ["ClinicalPipelineArtifact", Mapping[str, Any]],
    Mapping[str, Any],
]
ClinicalExternalStageGateway = Callable[
    [str, Mapping[str, Any]],
    Mapping[str, Any],
]


class WorkflowError(ValueError):
    """Base class for workflow orchestration errors."""


class WorkflowValidationError(WorkflowError):
    """Raised when a workflow declaration is invalid."""


class TransientWorkflowError(WorkflowError):
    """Raised by step adapters to signal a retryable transient failure."""


CLINICAL_PIPELINE_SCHEMA_VERSION = "openmed.clinical_pipeline.v1"
CLINICAL_ARTIFACT_SCHEMA_VERSION = "openmed.clinical_artifact.v1"
EXTERNAL_LLM_CAPABLE_STAGES = frozenset({"ground"})
_CLINICAL_STAGE_INDEX = {
    stage: index for index, stage in enumerate(CLINICAL_STAGE_ORDER)
}
_REGISTERED_STAGE_TOOLS = {
    "ground": "openmed_ground",
    "export": "openmed_export_fhir",
    "risk": "openmed_risk_score",
}
_EXTERNAL_STAGE_OPTION_KEYS = {
    "ground": frozenset({"vocabularies", "max_candidates"}),
}


class ClinicalStageOrderError(WorkflowValidationError):
    """Describe an invalid clinical stage declaration without echoing input."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        stages: Sequence[str] = (),
        stage: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stages = tuple(stages)
        error_details: dict[str, Any] = {
            "allowed_order": list(CLINICAL_STAGE_ORDER),
        }
        if details:
            error_details.update(dict(details))
        self.error = {
            "code": code,
            "message": message,
            "stage": stage,
            "details": error_details,
        }


@dataclass(frozen=True)
class ClinicalPipelineArtifact:
    """Typed, process-local state passed between clinical pipeline stages.

    ``text`` may contain sensitive source material and is therefore excluded
    from the dataclass representation and every public artifact payload. The
    serializable span handoff is always the canonical, text-free
    :class:`~openmed.core.schemas.OpenMedSpan` contract.
    """

    text: str | None = field(default=None, repr=False)
    spans: tuple[Mapping[str, Any], ...] = ()
    artifacts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    session_id: str | None = None
    workflow_id: str | None = None

    def __post_init__(self) -> None:
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("clinical pipeline text must be a string or null")
        normalized_spans = _canonical_span_payloads(self.spans)
        if self.text is not None and any(
            int(span["end"]) > len(self.text) for span in normalized_spans
        ):
            raise ValueError("clinical pipeline span offsets exceed the input text")
        object.__setattr__(self, "spans", normalized_spans)
        object.__setattr__(
            self,
            "artifacts",
            {
                str(stage): copy.deepcopy(dict(payload))
                for stage, payload in self.artifacts.items()
            },
        )

    def public_spans(self) -> list[dict[str, Any]]:
        """Return deep-copied canonical spans safe for structured handoff."""

        return [copy.deepcopy(dict(span)) for span in self.spans]

    def external_request(
        self,
        stage: str,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return the text-free request allowed to cross the privacy gateway."""

        return {
            "schema_version": CLINICAL_ARTIFACT_SCHEMA_VERSION,
            "stage": stage,
            "spans": self.public_spans(),
            "options": copy.deepcopy(dict(options)),
        }

    def advance(
        self,
        stage: str,
        payload: Mapping[str, Any],
    ) -> "ClinicalPipelineArtifact":
        """Validate one stage output and return the next immutable artifact."""

        output = copy.deepcopy(dict(payload))
        next_spans = self.spans
        if "spans" in output:
            next_spans = _canonical_span_payloads(output["spans"])
            output["spans"] = [copy.deepcopy(dict(span)) for span in next_spans]
        next_artifacts = dict(self.artifacts)
        next_artifacts[stage] = output
        return ClinicalPipelineArtifact(
            text=self.text,
            spans=next_spans,
            artifacts=next_artifacts,
            session_id=self.session_id,
            workflow_id=self.workflow_id,
        )


def validate_clinical_stage_order(stages: Sequence[Any]) -> tuple[str, ...]:
    """Validate and normalize a declared clinical pipeline stage sequence.

    Intermediate stages may be omitted, but declared stages must retain their
    relative order from :data:`CLINICAL_STAGE_ORDER`. Unknown, duplicate, or
    reordered stages are rejected deterministically rather than rearranged.

    Args:
        stages: Declared pipeline stage names.

    Returns:
        The normalized stage names as a tuple.

    Raises:
        ClinicalStageOrderError: If the declaration is empty or invalid.
    """

    if isinstance(stages, (str, bytes, bytearray)) or not isinstance(stages, Sequence):
        raise ClinicalStageOrderError(
            code="invalid_stage_list",
            message="Clinical pipeline stages must be a non-empty sequence.",
            details={"received_type": type(stages).__name__},
        )
    if not stages:
        raise ClinicalStageOrderError(
            code="invalid_stage_list",
            message="Clinical pipeline stages must be a non-empty sequence.",
            details={"declared_count": 0},
        )

    normalized: list[str] = []
    seen: set[str] = set()
    previous_stage: str | None = None
    previous_index = -1
    for declared_index, raw_stage in enumerate(stages):
        if not isinstance(raw_stage, str):
            raise ClinicalStageOrderError(
                code="unknown_stage",
                message="Clinical pipeline contains an unknown stage.",
                stages=normalized,
                details={"declared_index": declared_index},
            )

        stage = raw_stage.strip().lower()
        if stage not in _CLINICAL_STAGE_INDEX:
            raise ClinicalStageOrderError(
                code="unknown_stage",
                message="Clinical pipeline contains an unknown stage.",
                stages=normalized,
                details={"declared_index": declared_index},
            )
        if stage in seen:
            raise ClinicalStageOrderError(
                code="duplicate_stage",
                message="Clinical pipeline stages must be unique.",
                stages=(*normalized, stage),
                stage=stage,
                details={"declared_index": declared_index},
            )

        stage_index = _CLINICAL_STAGE_INDEX[stage]
        if stage_index < previous_index:
            raise ClinicalStageOrderError(
                code="invalid_stage_order",
                message="Clinical pipeline stages are not in canonical order.",
                stages=(*normalized, stage),
                stage=stage,
                details={
                    "declared_index": declared_index,
                    "previous_stage": previous_stage,
                },
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
    """Validate a clinical stage plan before invoking any supplied callback.

    The MCP contract uses this function without callbacks as a pure planner.
    Callback support lets later execution layers reuse the same validate-first
    boundary and makes the no-work-on-rejection guarantee directly testable.

    Args:
        stages: Declared pipeline stage names.
        stage_callbacks: Optional zero-argument callbacks keyed by stage name.

    Returns:
        A schema-versioned, machine-readable plan or rejection payload.
    """

    try:
        normalized_stages = validate_clinical_stage_order(stages)
    except ClinicalStageOrderError as exc:
        return _clinical_pipeline_payload(
            status="rejected",
            stages=exc.stages,
            error=exc.error,
        )

    if stage_callbacks is None:
        return _clinical_pipeline_payload(
            status="planned",
            stages=normalized_stages,
            trace=[
                {"stage": stage, "status": "planned"} for stage in normalized_stages
            ],
        )

    callbacks = dict(stage_callbacks)
    artifacts: dict[str, Any] = {}
    trace: list[dict[str, str]] = []
    for stage in normalized_stages:
        callback = callbacks.get(stage)
        if callback is None:
            trace.append({"stage": stage, "status": "planned"})
            continue
        artifacts[stage] = callback()
        trace.append({"stage": stage, "status": "completed"})

    status = "completed" if len(artifacts) == len(normalized_stages) else "planned"
    return _clinical_pipeline_payload(
        status=status,
        stages=normalized_stages,
        artifacts=artifacts,
        trace=trace,
    )


def execute_clinical_pipeline(
    stages: Sequence[Any],
    *,
    text: str | None,
    spans: Sequence[Mapping[str, Any]] | None,
    stage_handlers: Mapping[str, ClinicalStageHandler],
    options: Mapping[str, Any] | None = None,
    allow_external_llm: bool = False,
    external_stage_gateway: ClinicalExternalStageGateway | None = None,
    session_id: str | None = None,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    """Execute a validate-first clinical pipeline over typed artifacts.

    The full stage list, handler availability, options, canonical input spans,
    and privacy-gateway requirement are preflighted before any stage runs.
    External-LLM-capable stages never receive source text and cannot execute
    directly when external routing is enabled.

    Args:
        stages: Declarative clinical stages in canonical relative order.
        text: Optional process-local source text. It is never returned.
        spans: Optional canonical, text-free OpenMedSpan inputs.
        stage_handlers: Local handlers keyed by clinical stage name.
        options: Optional per-stage option mappings.
        allow_external_llm: Route external-capable stages through the gateway.
        external_stage_gateway: Required gateway callback for external routing.
        session_id: Optional process-local workflow session identifier.
        workflow_id: Optional process-local workflow identifier.

    Returns:
        A schema-versioned completed, rejected, or failed pipeline payload.
    """

    try:
        normalized_stages = validate_clinical_stage_order(stages)
    except ClinicalStageOrderError as exc:
        return _clinical_pipeline_payload(
            status="rejected",
            stages=exc.stages,
            error=exc.error,
        )

    try:
        normalized_options = _clinical_stage_options(options, normalized_stages)
        handlers = _clinical_stage_handlers(stage_handlers, normalized_stages)
        artifact = ClinicalPipelineArtifact(
            text=text,
            spans=tuple(spans or ()),
            session_id=session_id,
            workflow_id=workflow_id,
        )
        external_stages = set(normalized_stages) & EXTERNAL_LLM_CAPABLE_STAGES
        if allow_external_llm and external_stages:
            for stage in external_stages:
                unknown_external_options = set(normalized_options[stage]) - set(
                    _EXTERNAL_STAGE_OPTION_KEYS[stage]
                )
                if unknown_external_options:
                    raise ValueError(
                        "external clinical stage options contain an unsupported field"
                    )
            if external_stage_gateway is None:
                return _clinical_pipeline_payload(
                    status="failed",
                    stages=normalized_stages,
                    error=_clinical_execution_error(
                        code="privacy_gateway_required",
                        message=(
                            "External clinical stages require privacy-gateway routing."
                        ),
                        details={"external_stage_count": len(external_stages)},
                    ),
                )
    except Exception as exc:
        return _clinical_pipeline_payload(
            status="failed",
            stages=normalized_stages,
            error=_clinical_execution_error(
                code="invalid_pipeline_input",
                message="Clinical pipeline inputs are invalid.",
                details={"error_type": exc.__class__.__name__},
            ),
        )

    trace: list[dict[str, str]] = []
    for stage in normalized_stages:
        try:
            stage_options = normalized_options[stage]
            if allow_external_llm and stage in EXTERNAL_LLM_CAPABLE_STAGES:
                assert external_stage_gateway is not None
                output = external_stage_gateway(
                    stage,
                    artifact.external_request(stage, stage_options),
                )
            else:
                output = handlers[stage](artifact, stage_options)
            if not isinstance(output, Mapping):
                raise TypeError("clinical stage output must be a mapping")
            registered_tool = _REGISTERED_STAGE_TOOLS.get(stage)
            if registered_tool is not None:
                output = validate_registered_tool_output(registered_tool, output)
            artifact = artifact.advance(stage, output)
            trace.append({"stage": stage, "status": "completed"})
        except Exception as exc:
            return _clinical_pipeline_payload(
                status="failed",
                stages=normalized_stages,
                artifacts=artifact.artifacts,
                error=_clinical_execution_error(
                    code="stage_execution_failed",
                    message="A clinical pipeline stage failed.",
                    stage=stage,
                    details={"error_type": exc.__class__.__name__},
                ),
                trace=trace,
            )

    return _clinical_pipeline_payload(
        status="completed",
        stages=normalized_stages,
        artifacts=artifact.artifacts,
        trace=trace,
    )


def _canonical_span_payloads(
    spans: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(spans, (str, bytes, bytearray)) or not isinstance(spans, Sequence):
        raise TypeError("clinical pipeline spans must be a sequence")

    normalized: list[dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, Mapping):
            raise TypeError("clinical pipeline spans must be mappings")
        canonical = OpenMedSpan.from_dict(span).to_dict()
        if set(span) != set(canonical):
            raise ValueError("clinical pipeline spans must use the canonical schema")
        normalized.append(canonical)
    return tuple(normalized)


def _clinical_stage_options(
    options: Mapping[str, Any] | None,
    stages: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if options is None:
        return {stage: {} for stage in stages}
    if not isinstance(options, Mapping):
        raise TypeError("clinical pipeline options must be an object")
    unknown = set(options) - set(CLINICAL_STAGE_ORDER)
    if unknown:
        raise ValueError("clinical pipeline options contain an unknown stage")

    normalized: dict[str, dict[str, Any]] = {}
    for stage in stages:
        stage_options = options.get(stage, {})
        if stage_options is None:
            stage_options = {}
        if not isinstance(stage_options, Mapping):
            raise TypeError("clinical stage options must be objects")
        normalized[stage] = copy.deepcopy(dict(stage_options))
    return normalized


def _clinical_stage_handlers(
    handlers: Mapping[str, ClinicalStageHandler],
    stages: Sequence[str],
) -> dict[str, ClinicalStageHandler]:
    if not isinstance(handlers, Mapping):
        raise TypeError("clinical stage handlers must be a mapping")
    normalized: dict[str, ClinicalStageHandler] = {}
    for stage in stages:
        handler = handlers.get(stage)
        if not callable(handler):
            raise ValueError("clinical pipeline stage handler is unavailable")
        normalized[stage] = handler
    return normalized


def _clinical_execution_error(
    *,
    code: str,
    message: str,
    stage: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "stage": stage,
        "details": copy.deepcopy(dict(details or {})),
    }


def _clinical_pipeline_payload(
    *,
    status: str,
    stages: Sequence[str],
    artifacts: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
    trace: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    artifacts_payload = copy.deepcopy(dict(artifacts or {}))
    final_output: Any = None
    for stage in reversed(tuple(stages)):
        if stage in artifacts_payload:
            final_output = copy.deepcopy(artifacts_payload[stage])
            break
    return {
        "schema_version": CLINICAL_PIPELINE_SCHEMA_VERSION,
        "status": status,
        "stages": list(stages),
        "artifacts": artifacts_payload,
        "final_output": final_output,
        "error": copy.deepcopy(dict(error)) if error is not None else None,
        "trace": [copy.deepcopy(dict(item)) for item in trace],
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
