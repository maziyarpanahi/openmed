"""Bounded, provisioned local summarizer resolution and runtime admission."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from openmed.core.capabilities import MissingOptionalDependencyError
from openmed.core.model_registry import (
    resolve_summarizer_model,
    resolve_summarizer_package,
)
from openmed.core.offline import network_blocked_if_offline
from openmed.models.clinical_slm_capabilities import probe_clinical_slm_capabilities
from openmed.models.clinical_slm_manifest import (
    ClinicalSLMArtifactDigestMismatchError,
    ClinicalSLMArtifactError,
    ClinicalSLMArtifactManifest,
    ClinicalSLMArtifactMissingError,
    ClinicalSLMManifestError,
    ClinicalSLMValidationError,
    _open_local_file,
    load_clinical_slm_manifest,
    verify_clinical_slm_package,
)
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


_PACKAGE_REASON_CODES = frozenset(
    {
        "package_unpinned",
        "package_invalid",
        "platform_unsupported",
        "manifest_missing",
        "manifest_unreadable",
        "manifest_digest_required",
        "manifest_digest_mismatch",
        "model_identity_mismatch",
        "component_missing_on_disk",
        "component_unreadable",
        "unsafe_component_path",
        "component_size_mismatch",
        "component_digest_mismatch",
        "component_mutated",
        "undeclared_component",
        "task_unsupported",
        "capability_unsupported",
        "context_metadata_missing",
        "runtime_metadata_missing",
        "configuration_mismatch",
        "template_mismatch",
    }
)


class LocalSummarizerPackageError(LocalSummarizerError):
    """Refuse local package admission with one controlled reason code.

    Attributes:
        code: Stable reason code; no path, model identity or source payload.
    """

    def __init__(self, code: str) -> None:
        self.code = (
            code
            if type(code) is str and code in _PACKAGE_REASON_CODES
            else "package_invalid"
        )
        super().__init__(self.code)


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
    for package in ("mlx", "mlx_lm"):
        if importlib.util.find_spec(package) is None:
            raise MissingOptionalDependencyError(
                package=package, feature="local summarization", extra="mlx"
            )


def _read_package_json(
    root: Path, manifest: ClinicalSLMArtifactManifest, name: str, role: str
) -> Any:
    artifact = next(
        (
            item
            for item in manifest.components
            if item.path == name and item.component == role
        ),
        None,
    )
    if artifact is None or artifact.size_bytes > 1_048_576:
        raise LocalSummarizerPackageError("configuration_mismatch")
    with _open_local_file(root, name) as handle:
        payload = handle.read(1_048_577)
    if (
        len(payload) != artifact.size_bytes
        or "sha256:" + hashlib.sha256(payload).hexdigest() != artifact.sha256
    ):
        raise LocalSummarizerPackageError("component_digest_mismatch")
    invalid = False
    try:
        result = json.loads(payload)
    except (ValueError, UnicodeError, RecursionError):
        invalid = True
    if invalid:
        raise LocalSummarizerPackageError("configuration_mismatch")
    return result


def _load_model(path: Path) -> Any:
    from openmed.mlx.lm import OpenMedMLXLanguageModel

    return OpenMedMLXLanguageModel(str(path))


class MLXSummarizerBackend:
    """Generate locally with provisioned pinned weights and admission before loading.

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
        package_code = None
        try:
            with network_blocked_if_offline(local_only=True):
                result = self._generate(text)
        except LocalSummarizerPackageError as error:
            if type(error) is LocalSummarizerPackageError:
                package_code = error.code
            else:
                failed = True
        except ClinicalSLMManifestError as error:
            if type(error) in {
                ClinicalSLMManifestError,
                ClinicalSLMValidationError,
                ClinicalSLMArtifactError,
                ClinicalSLMArtifactMissingError,
                ClinicalSLMArtifactDigestMismatchError,
            }:
                package_code = error.code
            else:
                failed = True
        except Exception:
            failed = True
        if package_code is not None:
            raise LocalSummarizerPackageError(package_code)
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

        binding = resolve_summarizer_package(self._model_id)
        if binding is None:
            raise LocalSummarizerPackageError("package_unpinned")
        path, expected_digest = binding
        verify_clinical_slm_package(
            path, expected_manifest_digest=expected_digest, reject_undeclared_files=True
        )
        manifest = load_clinical_slm_manifest(path)
        if manifest.manifest_digest != expected_digest:
            raise LocalSummarizerPackageError("manifest_digest_mismatch")
        if (manifest.model_id, manifest.revision) != (self._model_id, self._revision):
            raise LocalSummarizerPackageError("model_identity_mismatch")
        if "clinical-summarization" not in manifest.supported_tasks:
            raise LocalSummarizerPackageError("task_unsupported")
        if manifest.context_limits is None:
            raise LocalSummarizerPackageError("context_metadata_missing")
        if (
            not manifest.required_runtime_features
            or "mlx" not in manifest.required_runtime_features
        ):
            raise LocalSummarizerPackageError("runtime_metadata_missing")
        config = _read_package_json(path, manifest, "config.json", "quantization")
        if not isinstance(config, dict):
            raise LocalSummarizerPackageError("configuration_mismatch")
        quantization = config.get("quantization")
        if not isinstance(quantization, dict):
            raise LocalSummarizerPackageError("configuration_mismatch")
        bits = quantization.get("bits")
        native_context = config.get("max_position_embeddings")
        if (
            type(bits) is not int
            or bits not in {2, 3, 4, 8}
            or bits != manifest.quantization.bits
            or manifest.quantization.scheme != f"int{bits}"
            or type(native_context) is not int
            or manifest.context_limits["max_context_tokens"] > native_context
        ):
            raise LocalSummarizerPackageError("configuration_mismatch")
        template = _read_package_json(path, manifest, "templates.json", "templates")
        if template != build_maple_task_messages("summarize", "{source}"):
            raise LocalSummarizerPackageError("template_mismatch")
        context = min(manifest.context_limits["max_context_tokens"], MAX_CONTEXT_TOKENS)
        output_tokens = min(
            manifest.context_limits["max_output_tokens"], MAX_OUTPUT_TOKENS
        )
        messages = build_maple_task_messages("summarize", text)
        # UTF-8 byte count conservatively bounds byte-tokenizer input before load.
        prompt_bound = sum(len(m["content"].encode("utf-8")) for m in messages) + 256
        if (
            prompt_bound > manifest.context_limits["max_input_tokens"]
            or prompt_bound + output_tokens > context
        ):
            raise LocalSummarizerError("summarizer input context exceeded")
        capability = probe_clinical_slm_capabilities(
            {
                **manifest.to_dict(),
                "cloud_fallback": False,
            },
            required_tasks=["clinical-summarization"],
            available_runtime_features=["mlx"],
            min_context_tokens=prompt_bound,
        )
        if not capability.supported:
            raise LocalSummarizerPackageError("capability_unsupported")
        weights = manifest.weights
        if not weights or len(weights) > 256:
            raise LocalSummarizerError("missing or unbounded model weights")
        weights_bytes = sum(item.size_bytes for item in weights)
        if weights_bytes <= 0:
            raise LocalSummarizerError("empty model weights")
        memory = preflight_clinical_slm_memory(
            {"weights_bytes": weights_bytes},
            ClinicalSLMRuntimeProfile(
                memory_budget_bytes=self._budget,
                headroom_bytes=max(self._budget // 10, 1),
                context_tokens=prompt_bound + output_tokens,
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
        if (
            len(tokens) > manifest.context_limits["max_input_tokens"]
            or len(tokens) + output_tokens > context
        ):
            raise LocalSummarizerError("summarizer token budget exceeded")
        output = runner.generate(
            prompt=prompt,
            max_tokens=output_tokens,
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
