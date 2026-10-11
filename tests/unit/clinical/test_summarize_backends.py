"""Offline runtime, privacy and fail-closed summarizer acceptance tests."""

import hashlib
import json
import re
import socket
import subprocess
from dataclasses import replace
from datetime import datetime
from enum import Enum
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

import openmed.clinical.summarize_backends as backends
import openmed.core.model_registry as registry
import openmed.models.clinical_slm_manifest as manifests
from openmed.clinical.summarize import (
    SummarizationLeakageError,
    summarize,
    summarize_deidentified,
)
from openmed.core.capabilities import MissingOptionalDependencyError
from openmed.core.model_registry import (
    register_summarizer_package,
    resolve_summarizer_model,
)
from openmed.core.offline import OfflineModeError
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.mlx.maple import MapleTask, build_maple_task_messages
from openmed.models.clinical_slm_manifest import (
    ClinicalSLMArtifact,
    ClinicalSLMArtifactManifest,
)

BACKEND_GUIDE = (
    Path(__file__).resolve().parents[3] / "docs" / "clinical" / "local-backends.md"
)
DOCUMENTED_ERROR_MODULES = (
    "openmed.clinical.summarize",
    "openmed.clinical.summarize_backends",
    "openmed.clinical.nli",
    "openmed.clinical.nli_backends",
    "openmed.clinical.nli_qualification",
    "openmed.clinical.extractive_selection",
    "openmed.clinical.brief_cancellation",
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
        r"(?m)^\| `([A-Z]\w+)` \|",
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
    monkeypatch.setattr(backends, "verify_clinical_slm_package", fail)
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


def test_local_backend_guide_check_rejects_new_interruption_export(monkeypatch):
    module = import_module("openmed.clinical.brief_cancellation")
    error = type(
        "SyntheticInterrupted", (RuntimeError,), {"__module__": module.__name__}
    )
    monkeypatch.setattr(module, "SyntheticInterrupted", error, raising=False)
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


def write_package(tmp_path, **metadata):
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 8192, "quantization": {"bits": 2}})
    )
    (tmp_path / "model.safetensors").write_bytes(b"synthetic weights")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "templates.json").write_text(
        json.dumps(build_maple_task_messages("summarize", "{source}"))
    )
    model_id, revision = resolve_summarizer_model()
    manifest = ClinicalSLMArtifactManifest(
        model_id=model_id,
        revision=revision,
        components=tuple(
            ClinicalSLMArtifact(
                component=role,
                path=name,
                sha256=hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
                size_bytes=(tmp_path / name).stat().st_size,
            )
            for role, name in (
                ("weights", "model.safetensors"),
                ("tokenizer", "tokenizer.json"),
                ("templates", "templates.json"),
                ("quantization", "config.json"),
            )
        ),
        quantization={"scheme": "int2", "bits": 2},
        licenses={"all": "Apache-2.0"},
        supported_tasks=["clinical-summarization"],
        context_limits={
            "max_context_tokens": 8192,
            "max_input_tokens": 6144,
            "max_output_tokens": 2048,
        },
        required_runtime_features=("mlx",),
    )
    manifest = replace(manifest, manifest_digest=None, **metadata)
    repin_package(tmp_path, manifest)
    return manifest


def repin_package(root, manifest):
    (root / manifests.MANIFEST_FILENAME).write_text(manifest.to_json())
    register_summarizer_package(
        "mlx", package_root=root, manifest_digest=manifest.manifest_digest
    )


@pytest.fixture
def local_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_SUMMARIZER_PACKAGES", {})
    manifest = write_package(tmp_path)
    calls = []
    state = {
        "answer": "A cough is present.",
        "tokens": 20,
        "root": tmp_path,
        "manifest": manifest,
        "max_tokens": 2048,
    }
    resolve_package = backends.resolve_summarizer_package
    verify_package = backends.verify_clinical_slm_package

    def package(model):
        calls.append("package")
        with pytest.raises(OfflineModeError):
            socket.create_connection(("example.invalid", 443))
        return resolve_package(model)

    def verify(*args, **kwargs):
        calls.append("manifest")
        assert kwargs["reject_undeclared_files"] is True
        return verify_package(*args, **kwargs)

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
        assert kwargs["max_tokens"] == state["max_tokens"]
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
    monkeypatch.setattr(backends, "resolve_summarizer_package", package)
    monkeypatch.setattr(backends, "verify_clinical_slm_package", verify)
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
    assert calls == ["runtime", "package", "manifest", "load", "generate"]
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

    monkeypatch.setattr(backends, "resolve_summarizer_package", fail)
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
    assert calls == [
        "runtime",
        "package",
        "manifest",
        "capability",
        "memory",
        "load",
        "generate",
    ]


@pytest.mark.parametrize(
    "tamper,code",
    [
        ("weight", "component_digest_mismatch"),
        ("extra", "undeclared_component"),
        ("symlink", "unsafe_component_path"),
        ("missing", "manifest_missing"),
        ("manifest", "manifest_digest_mismatch"),
        ("unpin", "package_unpinned"),
    ],
)
def test_package_refusals_precede_model_construction(local_runner, tamper, code):
    calls, state = local_runner
    root = state["root"]
    if tamper == "weight":
        (root / "model.safetensors").write_bytes(b"Synthetic weights")
    elif tamper == "extra":
        (root / "private-unlisted-file").write_text("synthetic private contents")
    elif tamper == "symlink":
        weight = root / "model.safetensors"
        outside = root.parent / "synthetic-alternate-weight"
        outside.write_bytes(weight.read_bytes())
        weight.unlink()
        weight.symlink_to(outside)
    elif tamper == "missing":
        (root / manifests.MANIFEST_FILENAME).unlink()
    elif tamper == "manifest":
        changed = replace(
            state["manifest"], supported_tasks=("clinical-ner",), manifest_digest=None
        )
        (root / manifests.MANIFEST_FILENAME).write_text(changed.to_json())
    else:
        registry.clear_summarizer_package()
    with pytest.raises(backends.LocalSummarizerPackageError) as caught:
        summarize_deidentified(deidentified(), model="mlx")
    assert caught.value.code == code
    assert str(caught.value) == code
    assert caught.value.__context__ is None
    assert "load" not in calls and "generate" not in calls


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"supported_tasks": ("clinical-ner",)}, "task_unsupported"),
        ({"context_limits": None}, "context_metadata_missing"),
        ({"required_runtime_features": None}, "runtime_metadata_missing"),
        (
            {"required_runtime_features": ("mlx", "unrecognized-runtime")},
            "capability_unsupported",
        ),
        ({"model_id": "OpenMed/Synthetic-Other"}, "model_identity_mismatch"),
        ({"revision": "a" * 40}, "model_identity_mismatch"),
        ({"quantization": {"scheme": "int4", "bits": 4}}, "configuration_mismatch"),
        (
            {
                "context_limits": {
                    "max_context_tokens": 9000,
                    "max_input_tokens": 6952,
                    "max_output_tokens": 2048,
                }
            },
            "configuration_mismatch",
        ),
    ],
)
def test_trusted_pin_does_not_bypass_package_capabilities(local_runner, changes, code):
    calls, state = local_runner
    repin_package(
        state["root"], replace(state["manifest"], manifest_digest=None, **changes)
    )
    with pytest.raises(backends.LocalSummarizerPackageError) as caught:
        backends.MLXSummarizerBackend().summarize("Synthetic note.")
    assert caught.value.code == code
    assert "load" not in calls


def test_unsupported_platform_fails_before_manifest_read_or_load(
    local_runner, monkeypatch
):
    calls, _ = local_runner
    monkeypatch.setattr(manifests, "_HAS_SECURE_LOCAL_READ", False)
    with pytest.raises(backends.LocalSummarizerPackageError) as caught:
        summarize_deidentified(deidentified(), model="mlx")
    assert caught.value.code == "platform_unsupported"
    assert "load" not in calls


def test_probe_receives_verified_metadata(local_runner, monkeypatch):
    _, state = local_runner
    probe = backends.probe_clinical_slm_capabilities
    observed = []
    manifest = replace(
        state["manifest"],
        context_limits={
            "max_context_tokens": 4096,
            "max_input_tokens": 3584,
            "max_output_tokens": 512,
        },
        manifest_digest=None,
    )
    repin_package(state["root"], manifest)

    def inspect(payload, **kwargs):
        observed.append(payload)
        return probe(payload, **kwargs)

    monkeypatch.setattr(backends, "probe_clinical_slm_capabilities", inspect)
    state["max_tokens"] = 512
    summarize_deidentified(deidentified(), model="mlx")
    assert observed[0]["context_limits"] == dict(manifest.context_limits)
    assert observed[0]["quantization"] == manifest.quantization.to_dict()
    assert observed[0]["supported_tasks"] == list(manifest.supported_tasks)
    assert observed[0]["required_runtime_features"] == ["mlx"]


@pytest.mark.parametrize(
    "name,payload,code",
    [
        ("config.json", b"{invalid json", "configuration_mismatch"),
        ("config.json", b'{"quantization":[]}', "configuration_mismatch"),
        (
            "templates.json",
            b'[{"role":"user","content":"synthetic changed template"}]',
            "template_mismatch",
        ),
    ],
)
def test_verified_components_must_match_runtime_contract(
    local_runner, name, payload, code
):
    calls, state = local_runner
    (state["root"] / name).write_bytes(payload)
    components = tuple(
        replace(
            item, sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)
        )
        if item.path == name
        else item
        for item in state["manifest"].components
    )
    repin_package(
        state["root"],
        replace(state["manifest"], components=components, manifest_digest=None),
    )
    with pytest.raises(backends.LocalSummarizerPackageError) as caught:
        backends.MLXSummarizerBackend().summarize("Synthetic note.")
    assert caught.value.code == code
    assert caught.value.__context__ is None
    assert "load" not in calls


def test_custom_package_refusal_is_sanitized():
    def custom(text):
        error = backends.LocalSummarizerPackageError("package_unpinned")
        error.code = text
        raise error

    with pytest.raises(backends.LocalSummarizerError) as caught:
        summarize_deidentified(deidentified(), model=custom)
    assert type(caught.value) is backends.LocalSummarizerError
    assert "cough" not in str(caught.value)
    assert caught.value.__context__ is None


def test_config_changed_after_verification_is_checked_on_read(
    local_runner, monkeypatch
):
    calls, state = local_runner
    verify = backends.verify_clinical_slm_package

    def mutate(*args, **kwargs):
        result = verify(*args, **kwargs)
        config = state["root"] / "config.json"
        config.write_bytes(b" " + config.read_bytes()[1:])
        return result

    monkeypatch.setattr(backends, "verify_clinical_slm_package", mutate)
    with pytest.raises(backends.LocalSummarizerPackageError) as caught:
        backends.MLXSummarizerBackend().summarize("Synthetic note.")
    assert caught.value.code == "component_digest_mismatch"
    assert "load" not in calls


def test_extra_file_added_during_verification_prevents_loading(
    local_runner, monkeypatch
):
    calls, state = local_runner
    original = manifests._hash_local_artifact

    def mutate(root, relative):
        result = original(root, relative)
        (state["root"] / "undeclared-private-file").write_text("synthetic payload")
        return result

    monkeypatch.setattr(manifests, "_hash_local_artifact", mutate)
    with pytest.raises(backends.LocalSummarizerPackageError) as caught:
        backends.MLXSummarizerBackend().summarize("Synthetic note.")
    assert caught.value.code == "undeclared_component"
    assert "load" not in calls


def test_provider_refusal_subclass_cannot_leak_exception_properties(
    local_runner, monkeypatch
):
    class UnsafeRefusal(backends.LocalSummarizerPackageError):
        @property
        def code(self):
            raise RuntimeError("synthetic private exception property")

    def load(path):
        error = UnsafeRefusal.__new__(UnsafeRefusal)
        RuntimeError.__init__(error, "synthetic private upstream payload")
        raise error

    monkeypatch.setattr(backends, "_load_model", load)
    with pytest.raises(backends.LocalSummarizerError) as caught:
        backends.MLXSummarizerBackend().summarize("Synthetic note.")
    assert caught.value.reason == "execution_failed"
    assert str(caught.value) == "local summarizer execution failed"
    assert caught.value.__context__ is None


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
        ("package", "package_unpinned"),
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
    if failure == "package":
        registry.clear_summarizer_package()
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
    actual_reason = (
        caught.value.code
        if type(caught.value) is backends.LocalSummarizerPackageError
        else caught.value.reason
    )
    assert actual_reason == reason
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


@pytest.mark.parametrize("boundary", ["load", "generate", "failure", "success"])
def test_owned_mlx_runner_released_at_all_terminal_boundaries(
    local_runner, monkeypatch, boundary
):
    from openmed.clinical.brief_cancellation import BriefCancellation, BriefInterrupted

    cancellation = BriefCancellation()
    original_load = backends._load_model
    owned = []

    def load(path):
        runner = original_load(path)
        runner.model = object()
        original_generate = runner.generate

        def generate(**kwargs):
            if boundary != "success":
                cancellation.cancel()
            if boundary == "failure":
                raise RuntimeError("SYNTHETIC_PRIVATE_PROVIDER_FAILURE")
            return original_generate(**kwargs)

        runner.generate = generate
        owned.append(runner)
        if boundary == "load":
            cancellation.cancel()
        return runner

    monkeypatch.setattr(backends, "_load_model", load)
    if boundary == "success":
        result = summarize_deidentified(
            deidentified(), model="mlx", cancellation=cancellation
        )
        assert result.summary == "A cough is present."
    else:
        with pytest.raises(BriefInterrupted, match="cancelled"):
            summarize_deidentified(
                deidentified(), model="mlx", cancellation=cancellation
            )
    assert len(owned) == 1
    assert owned[0].model is owned[0].tokenizer is None


@pytest.mark.parametrize("surface", ["summary", "mlx", "brief"])
def test_foreign_reason_property_is_not_read_or_retained(monkeypatch, surface):
    from openmed import clinical
    from tests.unit.clinical.test_brief import fixture_context

    reads = []
    marker = "".join(("SYNTHETIC", "_PRIVATE", "_REASON"))

    class ForeignError(backends.LocalSummarizerError):
        def __init__(self):
            RuntimeError.__init__(self, marker)

        @property
        def reason(self):
            reads.append(1)
            raise RuntimeError(marker)

    def fail(*args, **kwargs):
        raise ForeignError()

    if surface == "brief":
        monkeypatch.setattr(clinical, "summarize_deidentified", fail)
        value, context = fixture_context()
        brief = clinical.build_clinical_brief(
            value, context=context, model="extractive"
        )
        assert brief.refusal_reason.value == "stage_failed"
        assert brief.summary == ""
        assert marker not in json.dumps(brief.to_dict())
    else:
        if surface == "mlx":
            monkeypatch.setattr(backends, "_require_runtime", lambda: None)
            backend = backends.MLXSummarizerBackend()
            monkeypatch.setattr(backend, "_generate", fail)
            invoke = lambda: backend.summarize("Synthetic evidence.")
        else:
            invoke = lambda: summarize_deidentified(deidentified(), model=fail)
        with pytest.raises(backends.LocalSummarizerError) as caught:
            invoke()
        assert type(caught.value) is backends.LocalSummarizerError
        assert caught.value.reason == "execution_failed"
        assert caught.value.__context__ is caught.value.__cause__ is None
        assert marker not in str(caught.value)
    assert reads == []


def test_invalid_unicode_output_has_no_retained_decoder_input():
    marker = "".join(("SYNTHETIC", "_PRIVATE", "_OUTPUT"))
    with pytest.raises(backends.LocalSummarizerError) as caught:
        summarize_deidentified(deidentified(), model=lambda _: marker + "\ud800")
    assert caught.value.reason == "invalid_output"
    assert caught.value.__context__ is caught.value.__cause__ is None
    assert marker not in str(caught.value)
