"""Bounded, cache-only summarizer resolution and runtime admission."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from openmed.core.capabilities import MissingOptionalDependencyError
from openmed.core.model_registry import resolve_summarizer_model
from openmed.core.offline import network_blocked_if_offline
from openmed.models.clinical_slm_capabilities import probe_clinical_slm_capabilities
from openmed.models.clinical_slm_memory import (
    ClinicalSLMRuntimeProfile,
    preflight_clinical_slm_memory,
)
from openmed.models.clinical_slm_templates import compute_template_digest

MAX_INPUT_BYTES = 16_384
MAX_OUTPUT_BYTES = 8_192
MAX_RESPONSE_BYTES = 32_768
MAX_CONTEXT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 2048


class LocalSummarizerError(RuntimeError):
    """Content-free failure to resolve, admit, or run local summarization."""


class RemoteSummarizerError(LocalSummarizerError):
    """A network provider or URL was supplied to a local-only task."""


class ExtractiveSummarizerBackend:
    """Explicit deterministic CPU baseline, not a trained summarizer."""

    backend_id = "deterministic-extractive"
    template_digest = compute_template_digest("extractive-first-three-sentences-v1")

    def summarize(self, text: str, *, mode: str = "bhc") -> str:
        """Select up to three sentences without model loading or network use."""
        from openmed.clinical.summarize import _extractive_summary

        _validate_input(text, mode)
        return _extractive_summary(text)


def _validate_input(text: str, mode: str) -> None:
    if mode != "bhc":
        raise LocalSummarizerError("unsupported summarization mode")
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise LocalSummarizerError("summarizer input limit exceeded")


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
            raise LocalSummarizerError("invalid summarizer memory budget")
        failed = False
        try:
            self._model_id, self._revision = resolve_summarizer_model(model)
        except (TypeError, ValueError):
            failed = True
        if failed:
            raise LocalSummarizerError("unregistered local summarizer alias")
        self._budget = memory_budget_bytes
        from openmed.mlx.maple import build_maple_task_messages

        # The digest binds the complete fixed message template, never the note.
        self.template_digest = compute_template_digest(
            json.dumps(
                build_maple_task_messages("summarize", "{source}"), sort_keys=True
            )
        )

    def summarize(self, text: str, *, mode: str = "bhc") -> str:
        """Run all preflights, then generate under the outbound socket guard."""
        _validate_input(text, mode)
        _require_runtime()
        result: str | None = None
        failed = False
        try:
            with network_blocked_if_offline(local_only=True):
                result = self._generate(text)
        except Exception:
            failed = True
        if failed:
            # Raise outside the handler: upstream exceptions can contain PHI.
            raise LocalSummarizerError("local summarizer admission or inference failed")
        assert result is not None
        return result

    def _generate(self, text: str) -> str:
        from openmed.mlx.maple import (
            build_maple_task_messages,
            parse_maple_task_response,
        )

        path = _cached_artifact(self._model_id, self._revision)
        config_path = path / "config.json"
        if config_path.stat().st_size > 1_048_576:
            raise LocalSummarizerError("invalid model configuration")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        bits = config.get("quantization", {}).get("bits")
        if type(bits) is not int or bits not in {2, 3, 4, 8}:
            raise LocalSummarizerError("unsupported model quantization")
        native_context = config.get("max_position_embeddings", MAX_CONTEXT_TOKENS)
        if type(native_context) is not int or native_context <= MAX_OUTPUT_TOKENS:
            raise LocalSummarizerError("invalid model context")
        context = min(native_context, MAX_CONTEXT_TOKENS)
        messages = build_maple_task_messages("summarize", text)
        # UTF-8 byte count conservatively bounds byte-tokenizer input before load.
        prompt_bound = sum(len(m["content"].encode("utf-8")) for m in messages) + 256
        if prompt_bound + MAX_OUTPUT_TOKENS > context:
            raise LocalSummarizerError("summarizer input context exceeded")
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
            raise LocalSummarizerError("unsupported summarizer capability")
        weights = sorted(path.glob("model*.safetensors"))
        if not weights or len(weights) > 256:
            raise LocalSummarizerError("missing or unbounded model weights")
        weights_bytes = sum(p.stat().st_size for p in weights)
        if weights_bytes <= 0:
            raise LocalSummarizerError("empty model weights")
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
            raise LocalSummarizerError("summarizer memory budget exceeded")
        runner = _load_model(path)
        prompt = runner.format_chat_prompt(messages)
        tokens = runner.tokenizer.encode(prompt)
        if len(tokens) + MAX_OUTPUT_TOKENS > context:
            raise LocalSummarizerError("summarizer token budget exceeded")
        output = runner.generate(
            prompt=prompt,
            max_tokens=MAX_OUTPUT_TOKENS,
            temp=0.0,
            verbose=False,
            speculative=False,
        )
        if (
            not isinstance(output, str)
            or len(output.encode("utf-8")) > MAX_RESPONSE_BYTES
        ):
            raise LocalSummarizerError("invalid summarizer output")
        parsed = parse_maple_task_response("summarize", output, text)
        if not parsed.evidence or not parsed.answer:
            raise LocalSummarizerError("summary evidence is required")
        if len(parsed.answer.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise LocalSummarizerError("summary output limit exceeded")
        return parsed.answer


def resolve_summarizer_backend(model: object | None = None) -> object:
    """Resolve explicit extraction, registered MLX, or caller-owned local code.

    Custom callables are trusted application code, not sandboxed plugins. The
    caller must keep them local; OpenMed still blocks outbound Python sockets.
    """
    if model is None:
        return MLXSummarizerBackend()
    if isinstance(model, str):
        if model == "extractive":
            return ExtractiveSummarizerBackend()
        if ":" in model or model.lower() in {
            "remote",
            "openai",
            "anthropic",
            "azure",
            "bedrock",
        }:
            raise RemoteSummarizerError("remote summarizer backends are prohibited")
        return MLXSummarizerBackend(model)
    if callable(model) or callable(getattr(model, "summarize", None)):
        return model
    raise LocalSummarizerError("invalid local summarizer backend")
