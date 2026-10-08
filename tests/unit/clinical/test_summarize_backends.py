"""Offline runtime, privacy and fail-closed summarizer acceptance tests."""

import hashlib
import json
import socket
from dataclasses import replace
from datetime import datetime
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
    assert str(caught.value) == "local summarizer admission or inference failed"
    assert caught.value.__context__ is None
