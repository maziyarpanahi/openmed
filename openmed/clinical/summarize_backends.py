"""Bounded, cache-only summarizer resolution and runtime admission."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from openmed.clinical.brief_cancellation import (
    BriefCancellation,
    BriefInterrupted,
    check_cancellation,
)
from openmed.core.capabilities import MissingOptionalDependencyError
from openmed.core.model_registry import resolve_summarizer_model
from openmed.core.offline import network_blocked_if_offline
from openmed.models.clinical_slm_capabilities import probe_clinical_slm_capabilities
from openmed.models.clinical_slm_memory import (
    ClinicalSLMRuntimeProfile,
    preflight_clinical_slm_memory,
)
from openmed.models.clinical_slm_templates import compute_template_digest

if TYPE_CHECKING:
    from openmed.clinical.extractive_selection import (
        ExtractiveFact,
        ExtractiveSelection,
    )
    from openmed.clinical.summary_length_budget import SummaryLengthBudget
    from openmed.clinical.summary_omission_budget import ImportanceClassPolicy

MAX_INPUT_BYTES = 16_384
MAX_OUTPUT_BYTES = 8_192
MAX_RESPONSE_BYTES = 32_768
MAX_CONTEXT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 2048

_FAILURE_MESSAGES = {
    "execution_failed": "local summarizer execution failed",
    "unsupported_mode": "unsupported summarization mode",
    "input_limit_exceeded": "summarizer input limit exceeded",
    "invalid_memory_budget": "invalid summarizer memory budget",
    "unregistered_alias": "unregistered local summarizer alias",
    "artifact_not_cached": "local summarizer artifact is not cached",
    "invalid_configuration": "invalid local summarizer configuration",
    "capability_unsupported": "unsupported summarizer capability",
    "memory_budget_exceeded": "summarizer memory budget exceeded",
    "context_exceeded": "summarizer input context exceeded",
    "invalid_output": "invalid summarizer output",
    "output_limit_exceeded": "summary output limit exceeded",
    "deidentification_unavailable": "local de-identification is unavailable",
    "runtime_unavailable": "local summarizer runtime is unavailable",
    "remote_backend": "remote summarizer backends are prohibited",
    "invalid_backend": "invalid local summarizer backend",
}


class LocalSummarizerError(RuntimeError):
    """Content-free failure to resolve, admit, or run local summarization."""

    def __init__(self, message: str = "", *, reason: str = "execution_failed") -> None:
        # Retain the positional constructor for existing callers, but never
        # incorporate a caller-supplied message into an exception or traceback.
        self.reason = (
            reason
            if type(reason) is str and reason in _FAILURE_MESSAGES
            else "execution_failed"
        )
        super().__init__(_FAILURE_MESSAGES[self.reason])


class RemoteSummarizerError(LocalSummarizerError):
    """A network provider or URL was supplied to a local-only task."""


@dataclass(frozen=True)
class BriefGenerationEvidence:
    """Protected reviewed span exposed only to an injected local generator."""

    reference_id: str
    text: str = field(repr=False)
    start: int
    end: int


@dataclass(frozen=True, repr=False)
class BriefGeneratedClaim:
    """Atomic protected output with exactly one explicit evidence binding.

    The v1 contract refuses multiple references rather than guessing which
    reviewed clinical axes apply to the generated claim.
    """

    text: str
    reference_ids: tuple[str, ...]


@dataclass(frozen=True, repr=False)
class BriefGenerationResult:
    """Opt-in v1 brief output; legacy string summarization is unchanged.

    Claims are joined with one space, so no unbound prose can enter the output.
    Validation happens at the composer boundary, even for injected providers.
    """

    claims: tuple[BriefGeneratedClaim, ...]
    schema_version: int = 1

    def render(self) -> str:
        """Validate bounded atomic claims and return their protected text.

        Raises:
            LocalSummarizerError: For unknown versions or malformed bindings.
        """
        from openmed.clinical.summary_claim_segments import segment_summary_claims

        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.claims) is not tuple
            or not 0 < len(self.claims) <= 64
        ):
            raise LocalSummarizerError("invalid brief generation contract")
        total = 0
        for claim in self.claims:
            if (
                type(claim) is not BriefGeneratedClaim
                or type(claim.text) is not str
                or not claim.text
                or len(claim.text) > MAX_OUTPUT_BYTES
                or claim.text != claim.text.strip()
                or type(claim.reference_ids) is not tuple
                or len(claim.reference_ids) != 1
                or type(claim.reference_ids[0]) is not str
                or not 0 < len(claim.reference_ids[0]) <= 256
                or _utf8_size(claim.reference_ids[0]) > 256
            ):
                raise LocalSummarizerError("invalid brief claim binding")
            total += _utf8_size(claim.text)
            if total + len(self.claims) - 1 > MAX_OUTPUT_BYTES:
                raise LocalSummarizerError("brief output limit exceeded")
            segments = segment_summary_claims(claim.text).segments
            if (
                len(segments) != 1
                or segments[0].review_required
                or segments[0].text != claim.text
            ):
                raise LocalSummarizerError("non-atomic brief claim")
        return " ".join(claim.text for claim in self.claims)


class BoundBriefGenerator(Protocol):
    """Optional caller-owned local generator; no default model claims support."""

    def generate_brief(
        self, evidence: tuple[BriefGenerationEvidence, ...], *, mode: str
    ) -> BriefGenerationResult:
        """Return v1 claims referencing only the supplied reviewed evidence."""


def _utf8_size(text: str) -> int:
    if type(text) is str:
        try:
            return len(text.encode("utf-8"))
        except UnicodeEncodeError:
            pass
    # A decoder exception retains its input even if its message omits it.
    raise LocalSummarizerError(reason="invalid_output")


class ExtractiveSummarizerBackend:
    """Explicit CPU extraction, optionally bound to reviewed evidence.

    Args:
        evidence: Offset-only ``ExtractiveFact`` records, or ``None`` for the
            historical first-three-sentence baseline.
        importance_classes: Existing omission policies for those facts.
        length_budget: Existing class/global allowance for the complete extract.
    """

    backend_id = "deterministic-extractive"
    template_digest = compute_template_digest("extractive-script-aware-first-three-v2")

    def __init__(
        self,
        *,
        evidence: tuple[ExtractiveFact, ...] | None = None,
        importance_classes: tuple[ImportanceClassPolicy, ...] | None = None,
        length_budget: SummaryLengthBudget | None = None,
    ) -> None:
        self.evidence = evidence
        self.importance_classes = importance_classes
        self.length_budget = length_budget
        if evidence is not None:
            self.template_digest = compute_template_digest(
                "extractive-reviewed-fact-coverage-utf8-budget-v1"
            )

    def select(self, text: str) -> ExtractiveSelection:
        """Select whole sentences using the configured evidence and policies.

        Args:
            text: Already de-identified source matching the configured offsets.

        Returns:
            Protected text and value-free diagnostics, or an explicit refusal.
        """
        from openmed.clinical.extractive_selection import select_extractive_sentences

        return select_extractive_sentences(
            text,
            evidence=self.evidence,
            importance_classes=self.importance_classes,
            length_budget=self.length_budget,
        )

    def summarize(
        self,
        text: str,
        *,
        mode: str = "bhc",
        cancellation: BriefCancellation | None = None,
    ) -> str:
        """Select reviewed facts, or run the comparison baseline without evidence."""
        from openmed.clinical.summarize import _extractive_summary

        check_cancellation(cancellation)
        _validate_input(text, mode)
        if self.evidence is not None:
            from openmed.clinical.extractive_selection import ExtractiveSelectionError

            result = self.select(text)
            check_cancellation(cancellation)
            if result.status != "selected":
                raise ExtractiveSelectionError(result)
            return result.summary
        summary = _extractive_summary(text)
        check_cancellation(cancellation)
        return summary


def _validate_input(text: str, mode: str) -> None:
    if mode != "bhc":
        raise LocalSummarizerError(reason="unsupported_mode")
    if _utf8_size(text) > MAX_INPUT_BYTES:
        raise LocalSummarizerError(reason="input_limit_exceeded")


def _require_runtime() -> None:
    for package in ("mlx", "mlx_lm", "huggingface_hub"):
        if importlib.util.find_spec(package) is None:
            raise MissingOptionalDependencyError(
                package=package, feature="local summarization", extra="mlx"
            )


def _cached_artifact(model_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
            repo_type="model",
        )
    )


def _load_model(path: Path) -> Any:
    from openmed.mlx.lm import OpenMedMLXLanguageModel

    return OpenMedMLXLanguageModel(str(path))


class MLXSummarizerBackend:
    """Generate locally with pinned cached weights and admission before loading.

    Args:
        model: Reviewed registry alias. Arbitrary paths and remote providers
            are deliberately not accepted by this public clinical surface.
        memory_budget_bytes: Explicit upper bound for the conservative loading
            estimate, including weights, working memory and KV cache. This is
            an estimate, not proof of peak device memory or model quality.
    """

    backend_id = "local-mlx"

    def __init__(
        self, model: str = "mlx", *, memory_budget_bytes: int = 16 * 1024**3
    ) -> None:
        if type(memory_budget_bytes) is not int or not 0 < memory_budget_bytes <= 2**50:
            raise LocalSummarizerError(reason="invalid_memory_budget")
        failed = False
        try:
            self._model_id, self._revision = resolve_summarizer_model(model)
        except (TypeError, ValueError):
            failed = True
        if failed:
            raise LocalSummarizerError(reason="unregistered_alias")
        self._budget = memory_budget_bytes
        from openmed.mlx.maple import build_maple_task_messages

        # The digest binds the complete fixed message template, never the note.
        self.template_digest = compute_template_digest(
            json.dumps(
                build_maple_task_messages("summarize", "{source}"), sort_keys=True
            )
        )

    def summarize(
        self,
        text: str,
        *,
        mode: str = "bhc",
        cancellation: BriefCancellation | None = None,
    ) -> str:
        """Run all preflights, then generate under the outbound socket guard."""
        check_cancellation(cancellation)
        _validate_input(text, mode)
        _require_runtime()
        result: str | None = None
        reason = None
        try:
            with network_blocked_if_offline(local_only=True):
                result = self._generate(text, cancellation)
        except BriefInterrupted:
            raise
        except LocalSummarizerError as error:
            if type(error) is LocalSummarizerError:
                reason = error.reason
            else:
                reason = "execution_failed"
        except Exception:
            reason = "execution_failed"
        check_cancellation(cancellation)
        if reason is not None:
            # Raise outside the handler: upstream exceptions can contain PHI.
            raise LocalSummarizerError(reason=reason)
        assert result is not None
        return result

    def _generate(self, text: str, cancellation=None) -> str:
        from openmed.mlx.maple import (
            build_maple_task_messages,
            parse_maple_task_response,
        )

        path = None
        try:
            path = _cached_artifact(self._model_id, self._revision)
        except Exception:
            pass
        if path is None:
            raise LocalSummarizerError(reason="artifact_not_cached")
        config_path = path / "config.json"
        config = None
        try:
            if config_path.stat().st_size <= 1_048_576:
                config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        if not isinstance(config, dict) or not isinstance(
            config.get("quantization"), dict
        ):
            raise LocalSummarizerError(reason="invalid_configuration")
        bits = config["quantization"].get("bits")
        if type(bits) is not int or bits not in {2, 3, 4, 8}:
            raise LocalSummarizerError(reason="capability_unsupported")
        native_context = config.get("max_position_embeddings", MAX_CONTEXT_TOKENS)
        if type(native_context) is not int or native_context <= MAX_OUTPUT_TOKENS:
            raise LocalSummarizerError(reason="invalid_configuration")
        context = min(native_context, MAX_CONTEXT_TOKENS)
        messages = build_maple_task_messages("summarize", text)
        # UTF-8 byte count conservatively bounds byte-tokenizer input before load.
        prompt_bound = sum(len(m["content"].encode("utf-8")) for m in messages) + 256
        if prompt_bound + MAX_OUTPUT_TOKENS > context:
            raise LocalSummarizerError(reason="context_exceeded")
        capability = probe_clinical_slm_capabilities(
            {
                "supported_tasks": ["clinical-summarization"],
                "context_limits": {
                    "max_context_tokens": context,
                    "max_input_tokens": context - MAX_OUTPUT_TOKENS,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
                "required_runtime_features": ["mlx"],
                "quantization": {"scheme": f"int{bits}", "bits": bits},
                "offline": True,
                "human_review_required": True,
                "cloud_fallback": False,
            },
            required_tasks=["clinical-summarization"],
            available_runtime_features=["mlx"],
            min_context_tokens=prompt_bound,
        )
        if not capability.supported:
            raise LocalSummarizerError(reason="capability_unsupported")
        weights = sorted(path.glob("model*.safetensors"))
        if not weights or len(weights) > 256:
            raise LocalSummarizerError(reason="artifact_not_cached")
        weights_bytes = sum(p.stat().st_size for p in weights)
        if weights_bytes <= 0:
            raise LocalSummarizerError(reason="invalid_configuration")
        memory = preflight_clinical_slm_memory(
            {"weights_bytes": weights_bytes},
            ClinicalSLMRuntimeProfile(
                memory_budget_bytes=self._budget,
                headroom_bytes=max(self._budget // 10, 1),
                context_tokens=prompt_bound + MAX_OUTPUT_TOKENS,
                # Conservative cache/workspace assumptions, not a measured SLO.
                cache_bytes_per_token=1024**2,
                context_bytes_per_token=65_536,
                runtime_overhead_bytes=weights_bytes,
            ),
        )
        if not memory.accepted:
            raise LocalSummarizerError(reason="memory_budget_exceeded")
        check_cancellation(cancellation)
        runner = _load_model(path)
        try:
            check_cancellation(cancellation)
            prompt = runner.format_chat_prompt(messages)
            tokens = runner.tokenizer.encode(prompt)
            if len(tokens) + MAX_OUTPUT_TOKENS > context:
                raise LocalSummarizerError(reason="context_exceeded")
            check_cancellation(cancellation)
            output = runner.generate(
                prompt=prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                temp=0.0,
                verbose=False,
                speculative=False,
            )
            check_cancellation(cancellation)
            if not isinstance(output, str) or _utf8_size(output) > MAX_RESPONSE_BYTES:
                raise LocalSummarizerError(reason="invalid_output")
            parsed = None
            try:
                parsed = parse_maple_task_response("summarize", output, text)
            except (TypeError, ValueError):
                pass
            if parsed is None:
                raise LocalSummarizerError(reason="invalid_output")
            if not parsed.evidence or not parsed.answer:
                raise LocalSummarizerError(reason="invalid_output")
            if _utf8_size(parsed.answer) > MAX_OUTPUT_BYTES:
                raise LocalSummarizerError(reason="output_limit_exceeded")
            return parsed.answer
        finally:
            # This runner is created for this call; never unload caller-owned providers.
            runner.model = None
            runner.tokenizer = None


def resolve_summarizer_backend(model: object | None = None) -> object:
    """Resolve explicit extraction, registered MLX, or caller-owned local code.

    Custom callables are trusted application code, not sandboxed plugins. The
    caller must keep them local; OpenMed still blocks outbound Python sockets.
    """
    if model is None:
        return MLXSummarizerBackend()
    if isinstance(model, str):
        if model in {"extractive", "extractive-baseline"}:
            return ExtractiveSummarizerBackend()
        if ":" in model or model.lower() in {
            "remote",
            "openai",
            "anthropic",
            "azure",
            "bedrock",
        }:
            raise RemoteSummarizerError(reason="remote_backend")
        return MLXSummarizerBackend(model)
    if (
        callable(model)
        or callable(getattr(model, "summarize", None))
        or callable(getattr(model, "generate_brief", None))
    ):
        return model
    raise LocalSummarizerError(reason="invalid_backend")
