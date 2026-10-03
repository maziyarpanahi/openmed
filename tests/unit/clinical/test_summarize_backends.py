"""Offline runtime, privacy and fail-closed summarizer acceptance tests."""

import json
import socket
from datetime import datetime
from types import SimpleNamespace

import pytest

import openmed.clinical.summarize_backends as backends
from openmed.clinical.summarize import (
    SummarizationLeakageError,
    summarize,
    summarize_deidentified,
)
from openmed.core.capabilities import MissingOptionalDependencyError
from openmed.core.model_registry import resolve_summarizer_model
from openmed.core.offline import OfflineModeError
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.mlx.maple import MapleTask, build_maple_task_messages


def deidentified():
    return DeidentificationResult(
        original_text="Casey Example has a cough.",
        deidentified_text="[NAME] has a cough.",
        pii_entities=[
            PIIEntity(
                text="Casey Example",
                label="NAME",
                start=0,
                end=13,
                confidence=1.0,
                redacted_text="[NAME]",
            )
        ],
        method="mask",
        timestamp=datetime(2026, 1, 1),
    )


@pytest.fixture
def local_runner(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 8192, "quantization": {"bits": 2}})
    )
    (tmp_path / "model.safetensors").write_bytes(b"synthetic weights")
    calls = []
    state = {"answer": "A cough is present.", "tokens": 20}

    def cache(model, revision):
        calls.append("cache")
        assert len(revision) == 40
        with pytest.raises(OfflineModeError):
            socket.create_connection(("example.invalid", 443))
        return tmp_path

    def load(path):
        calls.append("load")
        assert path == tmp_path
        return SimpleNamespace(
            format_chat_prompt=lambda messages: json.dumps(messages),
            tokenizer=SimpleNamespace(encode=lambda _: [1] * state["tokens"]),
            generate=generate,
        )

    def generate(**kwargs):
        calls.append("generate")
        assert kwargs["temp"] == 0 and kwargs["verbose"] is False
        assert kwargs["max_tokens"] == 2048
        assert "Casey" not in kwargs["prompt"]
        with pytest.raises(OfflineModeError):
            socket.create_connection(("example.invalid", 443))
        return "</think>" + json.dumps(
            {
                "answer": state["answer"],
                "uncertainties": [],
                "evidence": [{"text": "has a cough"}],
            }
        )

    monkeypatch.setattr(backends, "_require_runtime", lambda: calls.append("runtime"))
    monkeypatch.setattr(backends, "_cached_artifact", cache)
    monkeypatch.setattr(backends, "_load_model", load)
    return calls, state


def test_default_resolves_pinned_registry_model():
    backend = backends.resolve_summarizer_backend()
    assert isinstance(backend, backends.MLXSummarizerBackend)
    assert resolve_summarizer_model("mlx") == resolve_summarizer_model("maple")
    assert len(resolve_summarizer_model()[1]) == 40


def test_raw_note_missing_runtime_fails_before_deidentification(monkeypatch):
    monkeypatch.setattr(backends.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(MissingOptionalDependencyError, match=r"openmed\[mlx\]"):
        summarize("Synthetic note", model="mlx")


def test_default_pii_mlx_route_uses_existing_export():
    from openmed.mlx.inference import _MLX_MODEL_MAP

    model = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
    assert _MLX_MODEL_MAP[model] == model + "-mlx"


def test_raw_deidentification_error_drops_private_context(monkeypatch):
    import importlib

    module = importlib.import_module("openmed.clinical.summarize")

    def fail(*args, **kwargs):
        assert kwargs["config"].local_only
        raise RuntimeError("Casey Example private note")

    monkeypatch.setattr(module, "deidentify", fail)
    with pytest.raises(backends.LocalSummarizerError) as caught:
        summarize("Casey Example private note", model="extractive")
    assert "Casey" not in str(caught.value)
    assert caught.value.__context__ is None


def test_reasoning_budget_is_reserved_before_model_load(local_runner):
    calls, _ = local_runner
    with pytest.raises(backends.LocalSummarizerError):
        backends.MLXSummarizerBackend().summarize("x" * 6100)
    assert "load" not in calls


def test_local_runtime_receives_only_deidentified_input(local_runner):
    calls, _ = local_runner
    result = summarize_deidentified(deidentified(), model="mlx")
    assert result.summary == "A cough is present."
    assert result.metadata["backend_id"] == "local-mlx"
    assert result.metadata["template_digest"].startswith("sha256:")
    assert calls == ["runtime", "cache", "load", "generate"]
    assert "cough" not in json.dumps(result.metadata)
    assert "cough" not in repr(result)


def test_template_digest_does_not_depend_on_patient_text(local_runner):
    first = summarize_deidentified(deidentified(), model="mlx")
    second = backends.MLXSummarizerBackend()
    assert first.template_digest == second.template_digest
    messages = build_maple_task_messages(MapleTask.SUMMARIZE, "Synthetic note.")
    assert "three concise sentences" in messages[1]["content"]


@pytest.mark.parametrize(
    "model",
    [
        "https://invalid.test/private",
        "openai",
        "azure",
        "anthropic",
        "bedrock",
        "file:///private/note",
        "ftp://invalid",
        "remote",
    ],
)
def test_remote_backend_is_rejected_without_echo(model):
    with pytest.raises(backends.RemoteSummarizerError) as caught:
        backends.resolve_summarizer_backend(model)
    assert str(caught.value) == "remote summarizer backends are prohibited"


@pytest.mark.parametrize("model", ["unknown", "/private/note", "org/unreviewed", 3, {}])
def test_unregistered_backend_is_rejected(model):
    with pytest.raises(backends.LocalSummarizerError):
        backends.resolve_summarizer_backend(model)


def test_missing_extra_has_canonical_error_and_no_implicit_fallback(monkeypatch):
    monkeypatch.setattr(backends.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(MissingOptionalDependencyError, match=r"openmed\[mlx\]"):
        summarize_deidentified(deidentified())
    assert summarize_deidentified(
        deidentified(), model="extractive"
    ).leakage_check.passed


def test_missing_weights_fails_without_raw_exception_context(local_runner, monkeypatch):
    def fail(*args):
        raise OSError("SYNTHETIC_PRIVATE")

    monkeypatch.setattr(backends, "_cached_artifact", fail)
    with pytest.raises(backends.LocalSummarizerError) as caught:
        summarize_deidentified(deidentified(), model="mlx")
    assert caught.value.__context__ is None
    assert "SYNTHETIC_PRIVATE" not in str(caught.value)


def test_memory_rejection_happens_before_loading(local_runner):
    calls, _ = local_runner
    with pytest.raises(backends.LocalSummarizerError):
        summarize_deidentified(
            deidentified(), model=backends.MLXSummarizerBackend(memory_budget_bytes=100)
        )
    assert "load" not in calls


def test_context_rejection_happens_before_loading(local_runner):
    calls, _ = local_runner
    with pytest.raises(backends.LocalSummarizerError):
        backends.MLXSummarizerBackend().summarize("x" * 8000)
    assert "load" not in calls


def test_exact_token_count_is_checked_before_generation(local_runner):
    calls, state = local_runner
    state["tokens"] = 8192
    with pytest.raises(backends.LocalSummarizerError):
        summarize_deidentified(deidentified(), model="mlx")
    assert "generate" not in calls


def test_generated_source_identifier_is_rejected(local_runner):
    _, state = local_runner
    state["answer"] = "Casey has a cough."
    with pytest.raises(SummarizationLeakageError):
        summarize_deidentified(deidentified(), model="mlx")


def test_custom_backend_network_attempt_is_blocked_without_context():
    def remote(text):
        socket.create_connection(("example.invalid", 443))

    with pytest.raises(backends.LocalSummarizerError) as caught:
        summarize_deidentified(deidentified(), model=remote)
    assert caught.value.__context__ is None


def test_custom_backend_metadata_does_not_use_class_name():
    backend = type("SYNTHETIC_PRIVATE", (), {"__call__": lambda self, text: "Cough."})()
    result = summarize_deidentified(deidentified(), model=backend)
    assert result.backend == "caller-supplied-local"
    assert "SYNTHETIC_PRIVATE" not in json.dumps(result.metadata)


def test_custom_missing_dependency_cannot_echo_note_in_error():
    def backend(text):
        raise MissingOptionalDependencyError(package="local", feature=text, extra="hf")

    with pytest.raises(backends.LocalSummarizerError) as caught:
        summarize_deidentified(deidentified(), model=backend)
    assert "cough" not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5, 2**51])
def test_memory_budget_validation(budget):
    with pytest.raises(backends.LocalSummarizerError):
        backends.MLXSummarizerBackend(memory_budget_bytes=budget)


@pytest.mark.parametrize("output", [None, 1, {}, "x" * 8193])
def test_invalid_custom_output_is_rejected(output):
    with pytest.raises(backends.LocalSummarizerError):
        summarize_deidentified(deidentified(), model=lambda _: output)


def test_preflight_order_is_enforced(local_runner, monkeypatch):
    calls, _ = local_runner
    capability = backends.probe_clinical_slm_capabilities
    memory = backends.preflight_clinical_slm_memory

    def check_capability(*args, **kwargs):
        calls.append("capability")
        return capability(*args, **kwargs)

    def check_memory(*args, **kwargs):
        calls.append("memory")
        return memory(*args, **kwargs)

    monkeypatch.setattr(backends, "probe_clinical_slm_capabilities", check_capability)
    monkeypatch.setattr(backends, "preflight_clinical_slm_memory", check_memory)
    summarize_deidentified(deidentified(), model="mlx")
    assert calls == ["runtime", "cache", "capability", "memory", "load", "generate"]


@pytest.mark.parametrize(
    ("model", "reason"),
    [("not-a-registered-alias", "unregistered_alias"), (123, "invalid_backend")],
)
def test_backend_resolution_reasons_are_typed(model, reason):
    with pytest.raises(backends.LocalSummarizerError) as caught:
        backends.resolve_summarizer_backend(model)
    assert caught.value.reason == reason


def test_builtin_mode_rejected_before_deidentification(monkeypatch):
    import importlib

    module = importlib.import_module("openmed.clinical.summarize")
    monkeypatch.setattr(
        module, "deidentify", lambda *a, **kw: pytest.fail("must not deidentify")
    )
    with pytest.raises(backends.LocalSummarizerError) as caught:
        summarize("synthetic-private-note", mode="unsupported", model="extractive")
    assert caught.value.reason == "unsupported_mode"
    assert "synthetic-private" not in str(caught.value)


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("cache", "artifact_not_cached"),
        ("memory", "memory_budget_exceeded"),
        ("context", "context_exceeded"),
        ("capability", "capability_unsupported"),
        ("output", "invalid_output"),
    ],
)
def test_local_failures_preserve_safe_reasons(
    local_runner, monkeypatch, failure, reason
):
    _, state = local_runner
    backend = backends.MLXSummarizerBackend()
    if failure == "cache":

        def fail_cache(*args):
            raise OSError("synthetic-private-cache-path")

        monkeypatch.setattr(backends, "_cached_artifact", fail_cache)
    elif failure == "memory":
        backend = backends.MLXSummarizerBackend(memory_budget_bytes=100)
    elif failure == "context":
        state["tokens"] = 8192
    elif failure == "capability":
        monkeypatch.setattr(
            backends,
            "probe_clinical_slm_capabilities",
            lambda *a, **kw: SimpleNamespace(supported=False),
        )
    elif failure == "output":
        state["answer"] = None

    with pytest.raises(backends.LocalSummarizerError) as caught:
        summarize_deidentified(deidentified(), model=backend)
    assert caught.value.reason == reason
    assert caught.value.__context__ is None
    assert "synthetic-private" not in str(caught.value)


@pytest.mark.parametrize(
    ("output", "reason"),
    [(object(), "invalid_output"), ("x" * 8193, "output_limit_exceeded")],
)
def test_custom_type_and_size_failures_are_distinct(output, reason):
    with pytest.raises(backends.LocalSummarizerError) as caught:
        summarize_deidentified(deidentified(), model=lambda _: output)
    assert caught.value.reason == reason
    assert caught.value.__context__ is None


def test_uncontrolled_error_message_and_reason_never_escape():
    marker = "synthetic-private-error"

    def backend(_):
        raise backends.LocalSummarizerError(marker, reason=marker)

    with pytest.raises(backends.LocalSummarizerError) as caught:
        summarize_deidentified(deidentified(), model=backend)
    assert caught.value.reason == "execution_failed"
    assert marker not in str(caught.value)
    assert caught.value.__context__ is None


def test_summarizer_types_are_public_without_loading_models():
    from openmed import clinical

    for name in (
        "LocalSummarizerError",
        "RemoteSummarizerError",
        "ExtractiveSummarizerBackend",
        "MLXSummarizerBackend",
        "resolve_summarizer_backend",
    ):
        assert getattr(clinical, name) is getattr(backends, name)


@pytest.mark.parametrize(
    "sentences",
    [
        ["咳嗽持续。", "开始治疗。", "病情好转。", "随后出院。"],
        ["咳が続く。", "薬を投与した。", "改善した。", "退院した。"],
        ["खांसी है।", "उपचार दिया।", "सुधार हुआ।", "छुट्टी मिली।"],
        ["له سعال.", "تناول العلاج.", "تحسن.", "خرج."],
        [
            "Dr. Example assessed a cough.",
            "The value was 3.5.",
            "Stable.",
            "Discharged.",
        ],
    ],
)
def test_extractive_script_boundaries_keep_three_exact_source_sentences(sentences):
    note = " ".join(sentences)
    result = backends.ExtractiveSummarizerBackend().summarize(note)
    assert result == " ".join(sentences[:3])
    assert all(sentence in note for sentence in sentences[:3])
    assert sentences[3] not in result


def test_long_cjk_note_has_a_bounded_extractive_summary():
    note = "咳嗽持续。开始治疗。病情好转。随后出院。" * 170
    assert 8192 < len(note.encode("utf-8")) <= backends.MAX_INPUT_BYTES
    summary = backends.ExtractiveSummarizerBackend().summarize(note)
    assert summary == "咳嗽持续。 开始治疗。 病情好转。"
    assert len(summary.encode("utf-8")) <= backends.MAX_OUTPUT_BYTES


def test_extractive_algorithm_has_a_versioned_template_digest():
    from openmed.models.clinical_slm_templates import compute_template_digest

    assert (
        backends.ExtractiveSummarizerBackend.template_digest
        == compute_template_digest("extractive-script-aware-first-three-v2")
    )
