"""Offline runtime, privacy and fail-closed summarizer acceptance tests."""

import json
import re
import socket
import subprocess
from datetime import datetime
from enum import Enum
from importlib import import_module
from pathlib import Path
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

BACKEND_GUIDE = (
    Path(__file__).resolve().parents[3] / "docs" / "clinical" / "local-backends.md"
)
DOCUMENTED_ERROR_MODULES = (
    "openmed.clinical.summarize",
    "openmed.clinical.summarize_backends",
    "openmed.clinical.nli",
    "openmed.clinical.nli_backends",
    "openmed.models.clinical_slm_manifest",
    "openmed.models.clinical_slm_capabilities",
    "openmed.models.clinical_slm_memory",
)


def _backend_error_exports() -> tuple[str, ...]:
    names = {"MissingOptionalDependencyError"}
    for module_name in DOCUMENTED_ERROR_MODULES:
        module = import_module(module_name)
        exports = getattr(module, "__all__", None)
        if exports is None:
            exports = [
                name
                for name, value in vars(module).items()
                if not name.startswith("_")
                and getattr(value, "__module__", None) == module_name
            ]
        for name in exports:
            value = getattr(module, name)
            if isinstance(value, type) and issubclass(value, Exception):
                names.add(name)
    return tuple(sorted(names))


def _backend_table(markdown: str, heading: str) -> str:
    assert heading in markdown, "backend outcome table is missing"
    return markdown.split(heading, 1)[1].split("\n## ", 1)[0]


def _assert_documented_backend_outcomes(markdown: str) -> None:
    errors = re.findall(
        r"(?m)^\| `([A-Z]\w+Error)` \|",
        _backend_table(markdown, "## Backend exceptions\n"),
    )
    assert tuple(sorted(errors)) == _backend_error_exports(), (
        "documented backend errors differ from public exports"
    )
    refusals = re.findall(
        r"(?m)^\| `([A-Z_]+)` \| `([a-z_]+)` \|",
        _backend_table(markdown, "## Brief refusals\n"),
    )
    brief_module = import_module("openmed.clinical.brief")
    expected = tuple(
        sorted(
            (name, member.value)
            for name, member in brief_module.BriefRefusal.__members__.items()
        )
    )
    assert tuple(sorted(refusals)) == expected, (
        "documented brief refusals differ from enum members"
    )


def _run_backend_guide_examples(markdown: str) -> list[dict]:
    from openmed.core.offline import network_blocked_if_offline

    examples = re.findall(r"```python\n(.*?)\n```", markdown, flags=re.DOTALL)
    assert examples, "backend guide has no runnable Python examples"
    namespaces = []
    with network_blocked_if_offline(local_only=True):
        for index, source in enumerate(examples):
            namespace = {"__name__": "__local_backend_docs_example__"}
            exec(compile(source, f"<backend-doc-example-{index}>", "exec"), namespace)
            namespaces.append(namespace)
    return namespaces


@pytest.fixture
def forbid_backend_guide_model_loading(monkeypatch):
    from openmed.core.models import ModelLoader

    def fail(*args, **kwargs):
        raise AssertionError("backend documentation attempted external runtime work")

    monkeypatch.setattr(backends, "_require_runtime", fail)
    monkeypatch.setattr(backends, "_cached_artifact", fail)
    monkeypatch.setattr(backends, "_load_model", fail)
    monkeypatch.setattr(ModelLoader, "load_local_sequence_classifier", fail)
    monkeypatch.setattr(ModelLoader, "load_model", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)


def test_local_backend_guide_documents_every_error_and_brief_refusal():
    _assert_documented_backend_outcomes(BACKEND_GUIDE.read_text(encoding="utf-8"))


def test_local_backend_guide_examples_run_offline_with_only_synthetic_doubles(
    forbid_backend_guide_model_loading, capsys
):
    examples = _run_backend_guide_examples(BACKEND_GUIDE.read_text(encoding="utf-8"))
    assert len(examples) == 4
    assert examples[0]["custom"].backend == "caller-supplied-local"
    assert examples[1]["verdicts"][0]["label"] == "entailment"
    assert examples[2]["integrity"].verified
    assert examples[2]["capability"].supported
    assert examples[2]["memory"].accepted
    assert not examples[2]["package"].exists()
    assert examples[3]["audit"]["status"] == "needs_review"
    assert "summary" not in examples[3]["audit"]
    assert capsys.readouterr().out == ""


def test_local_backend_guide_check_rejects_new_exported_backend_error(monkeypatch):
    module = import_module("openmed.clinical.nli_backends")
    error = type("SyntheticNewBackendError", (RuntimeError,), {})
    monkeypatch.setattr(module, "SyntheticNewBackendError", error, raising=False)
    monkeypatch.setattr(
        module, "__all__", [*module.__all__, "SyntheticNewBackendError"]
    )
    with pytest.raises(AssertionError, match="errors differ from public exports"):
        _assert_documented_backend_outcomes(BACKEND_GUIDE.read_text(encoding="utf-8"))


def test_local_backend_guide_check_rejects_new_implicit_backend_export(monkeypatch):
    error = type(
        "SyntheticNewBackendError", (RuntimeError,), {"__module__": backends.__name__}
    )
    monkeypatch.setattr(backends, "SyntheticNewBackendError", error, raising=False)
    with pytest.raises(AssertionError, match="errors differ from public exports"):
        _assert_documented_backend_outcomes(BACKEND_GUIDE.read_text(encoding="utf-8"))


def test_local_backend_guide_check_rejects_new_brief_refusal(monkeypatch):
    module = import_module("openmed.clinical.brief")
    members = {
        name: item.value for name, item in module.BriefRefusal.__members__.items()
    }
    members["SYNTHETIC_NEW_REFUSAL"] = "synthetic_new_refusal"
    monkeypatch.setattr(module, "BriefRefusal", Enum("SyntheticRefusal", members))
    with pytest.raises(AssertionError, match="refusals differ from enum members"):
        _assert_documented_backend_outcomes(BACKEND_GUIDE.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        ("| `LocalNLIError` |", "| `SyntheticMissingError` |", "errors differ"),
        (
            "| `LocalNLIError` |",
            "| `LocalNLIError` |\n| `LocalNLIError` |",
            "errors differ",
        ),
        ("| `PRIVACY` | `privacy` |", "| `PRIVACY` | `changed` |", "refusals differ"),
        ("## Backend exceptions\n", "## Removed table\n", "table is missing"),
        ("## Brief refusals\n", "## Removed table\n", "table is missing"),
    ],
)
def test_local_backend_guide_check_rejects_table_drift(before, after, message):
    markdown = BACKEND_GUIDE.read_text(encoding="utf-8").replace(before, after, 1)
    with pytest.raises(AssertionError, match=message):
        _assert_documented_backend_outcomes(markdown)


def test_local_backend_guide_example_check_rejects_incompatible_brief_provider(
    forbid_backend_guide_model_loading,
):
    markdown = BACKEND_GUIDE.read_text(encoding="utf-8").replace(
        '"calibration_id": thresholds.calibration_id,',
        '"calibration_id": "synthetic-incompatible",',
        1,
    )
    with pytest.raises(AssertionError):
        _run_backend_guide_examples(markdown)


def test_local_backend_guide_example_check_rejects_missing_examples():
    with pytest.raises(AssertionError, match="no runnable Python examples"):
        _run_backend_guide_examples("No executable examples.\n")


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
