"""Offline provenance regression tests for CoreML package manifests."""

from __future__ import annotations

import importlib
import json
import shutil
import types
from pathlib import Path

import pytest

from .test_coreml_convert import _install_conversion_stubs


def _converter(monkeypatch: pytest.MonkeyPatch):
    module = importlib.import_module("openmed.coreml.convert")
    state = _install_conversion_stubs(
        monkeypatch,
        model_type="bert",
        architectures=["BertForTokenClassification"],
    )
    return module, state


def test_canonical_package_preserves_observations_in_parent_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, _ = _converter(monkeypatch)
    output = tmp_path / "model.mlpackage"
    clock = iter(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"])
    latency = iter(
        [
            {"measured": True, "p50": 1.0, "p95": 2.0, "iterations": 3},
            {"measured": True, "p50": 4.0, "p95": 5.0, "iterations": 3},
        ]
    )
    monkeypatch.setattr(module, "_utc_now", lambda: next(clock))
    monkeypatch.setattr(
        module, "_measure_prediction_latency", lambda *a, **k: next(latency)
    )

    module.convert("OpenMed/tiny-stub", output, max_seq_length=16)
    manifest = output / module.COREML_MANIFEST_FILENAME
    first = manifest.read_bytes()
    first_parent = json.loads((tmp_path / "model_coreml_manifest.json").read_text())
    module.convert("OpenMed/tiny-stub", output, max_seq_length=16)
    second = manifest.read_bytes()
    second_parent = json.loads((tmp_path / "model_coreml_manifest.json").read_text())

    assert first == second
    canonical = json.loads(first)
    assert canonical["schema_version"] == 2
    assert "generated_at" not in canonical
    for variant in canonical["variants"]:
        assert variant["path"] == "."
        assert not {"latency_ms", "residency", "parity"} & variant.keys()
    assert first_parent["schema_version"] == second_parent["schema_version"] == 1
    assert first_parent["generated_at"] != second_parent["generated_at"]
    assert first_parent["variants"][0]["latency_ms"]["p50"] == 1.0
    assert second_parent["variants"][0]["latency_ms"]["p50"] == 4.0
    assert second_parent["variants"][0]["residency"]["artifact_path"] == output.name
    assert (tmp_path / "model_id2label.json").is_file()
    assert not (output / "id2label.json").exists()


def test_custom_parent_and_quantized_roots_each_get_own_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, _ = _converter(monkeypatch)
    output = tmp_path / "models" / "model.mlpackage"
    quantized = tmp_path / "quantized" / "model-int8.mlpackage"
    parent = tmp_path / "reports" / "conversion.json"
    module.convert(
        "OpenMed/tiny-stub",
        output,
        quantize="int8",
        quantized_output_path=quantized,
        conversion_manifest_path=parent,
        latency_iterations=0,
    )
    assert parent.is_file()
    for package, expected_name in [(output, "coreml-fp16"), (quantized, "coreml-int8")]:
        manifest = json.loads((package / module.COREML_MANIFEST_FILENAME).read_text())
        assert manifest["schema_version"] == 2
        assert len(manifest["variants"]) == 1
        assert manifest["variants"][0]["name"] == expected_name
        assert manifest["variants"][0]["path"] == "."
        assert manifest["provenance"]["files"]
        assert all(
            not row["path"].startswith("/") for row in manifest["provenance"]["files"]
        )
        from openmed.core.export_provenance import verify_export_provenance

        moved = tmp_path / "moved" / package.name
        shutil.copytree(package, moved)
        assert (
            verify_export_provenance(moved, module.COREML_MANIFEST_FILENAME)
            == manifest["provenance"]
        )


@pytest.mark.parametrize("local_source", [False, True])
def test_resolve_once_pins_loaders_and_separates_public_source_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, local_source: bool
) -> None:
    module, state = _converter(monkeypatch)
    from openmed.core.export_provenance import resolve_source_context

    load_reference = str(tmp_path / "private-user" / "snapshot")
    local_digest = None
    identity = "OpenMed/tiny-stub"
    if local_source:
        checkpoint = Path(load_reference)
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text('{"model_type":"bert","license":"MIT"}')
        local_context = resolve_source_context(load_reference)
        local_digest = local_context.source_digest
        identity = local_context.source_model_id
    requested = load_reference if local_source else "OpenMed/tiny-stub"
    calls = []

    def resolve(model_id, *, cache_dir=None, revision=None):
        calls.append((model_id, cache_dir, revision))
        return types.SimpleNamespace(
            source_model_id=identity,
            load_reference=load_reference,
            revision=None if local_source else "b" * 40,
            source_digest=local_digest,
            license="MIT",
        )

    monkeypatch.setattr(module, "resolve_source_context", resolve, raising=False)
    output = tmp_path / "model.mlpackage"
    module.convert(requested, output, cache_dir="synthetic-cache", latency_iterations=0)

    assert calls == [(requested, "synthetic-cache", None)]
    assert state.config_references == [load_reference]
    assert state.tokenizer_references == [load_reference]
    assert state.model_references == [load_reference]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["user_defined_metadata"]["source_model"] == identity
    assert load_reference not in json.dumps(metadata)
    canonical = json.loads((output / module.COREML_MANIFEST_FILENAME).read_text())
    assert canonical["source_model_id"] == identity
    assert canonical["provenance"]["source"] == {
        "model_id": identity,
        "revision": None if local_source else "b" * 40,
        "content_digest": local_digest,
        "license": "MIT",
    }


@pytest.mark.parametrize("mutation", ["modify", "add", "delete"])
def test_explicit_verifier_checks_package_members_without_changing_legacy_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    from openmed.core.export_provenance import verify_export_provenance

    module, _ = _converter(monkeypatch)
    output = tmp_path / "model.mlpackage"
    module.convert("OpenMed/tiny-stub", output, latency_iterations=0)
    verified = verify_export_provenance(output, module.COREML_MANIFEST_FILENAME)
    assert [row["path"] for row in verified["files"]] == [
        "metadata.json",
        "weights.bin",
    ]
    manifest = json.loads((output / module.COREML_MANIFEST_FILENAME).read_text())
    assert manifest["variants"][0]["size_bytes"] == sum(
        row["size_bytes"] for row in verified["files"]
    )
    weights = output / "weights.bin"
    if mutation == "modify":
        original = weights.read_bytes()
        weights.write_bytes(b"1" + original[1:])
    elif mutation == "delete":
        weights.unlink()
    else:
        nested = output / "nested" / module.COREML_MANIFEST_FILENAME
        nested.parent.mkdir()
        nested.write_text("{}")
    # This reader continues to enforce its documented optional-resource contract.
    assert module.validate_coreml_bundle(output)["schema_version"] == 2
    with pytest.raises(ValueError):
        verify_export_provenance(output, module.COREML_MANIFEST_FILENAME)


@pytest.mark.parametrize("schema_version", [1, 2])
def test_both_manifest_versions_keep_real_segmenter_resource_validation(
    tmp_path: Path, schema_version: int
) -> None:
    from openmed.processing.tokenization import package_segmenter_resources

    module = importlib.import_module("openmed.coreml.convert")
    output = tmp_path / "model.mlpackage"
    output.mkdir()
    segmenter = package_segmenter_resources(output, "openmed-cjk-indic-v1")
    manifest = {
        "format": "openmed-coreml",
        "schema_version": schema_version,
        "segmenter": segmenter,
    }
    (output / module.COREML_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert module.validate_coreml_bundle(output)["schema_version"] == schema_version
    resource = output / "segmenter" / "han_words.txt"
    original = resource.read_bytes()
    resource.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="hash mismatch"):
        module.validate_coreml_bundle(output)


def test_package_finalizer_rejects_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, _ = _converter(monkeypatch)
    output = tmp_path / "model.mlpackage"
    output.mkdir()
    target = tmp_path / "external.bin"
    target.write_bytes(b"synthetic-external")
    try:
        (output / "linked.bin").symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        module.convert("OpenMed/tiny-stub", output, latency_iterations=0)


def test_observation_report_must_remain_outside_package_before_source_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, state = _converter(monkeypatch)
    output = tmp_path / "model.mlpackage"
    with pytest.raises(ValueError, match="outside CoreML packages"):
        module.convert(
            "OpenMed/tiny-stub",
            output,
            conversion_manifest_path=output / "observations.json",
        )
    assert state.model_loads == state.config_loads == state.tokenizer_loads == 0


def test_direct_manifest_writer_uses_available_metadata_without_source_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from openmed.core.export_provenance import verify_export_provenance

    module = importlib.import_module("openmed.coreml.convert")
    output = tmp_path / "model.mlpackage"
    output.mkdir()
    (output / "weights.bin").write_bytes(b"synthetic-weights")

    def forbidden(*args, **kwargs):
        raise AssertionError("direct writer must not resolve remote sources")

    monkeypatch.setattr(module, "resolve_source_context", forbidden)
    module._write_coreml_conversion_manifest(
        tmp_path / "observations.json",
        model_id="OpenMed/tiny-stub",
        model_type="bert",
        compute_units="cpuOnly",
        max_seq_length=16,
        optimize_for_ane=False,
        variants=[
            module.CoreMLVariantRecord(
                name="coreml-fp16",
                path=output.name,
                precision="float16",
                quantization="none",
                size_bytes=17,
            )
        ],
    )
    provenance = verify_export_provenance(output, module.COREML_MANIFEST_FILENAME)
    assert provenance["source"]["model_id"] == "OpenMed/tiny-stub"
    assert provenance["source"]["revision"] is None


def test_conversion_passes_pinned_reference_to_parity_without_serializing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, _ = _converter(monkeypatch)
    private_reference = str(tmp_path / "private-snapshot")
    monkeypatch.setattr(
        module,
        "resolve_source_context",
        lambda *a, **k: types.SimpleNamespace(
            source_model_id="OpenMed/tiny-stub",
            load_reference=private_reference,
            revision="a" * 40,
            source_digest=None,
            license="MIT",
        ),
    )
    output = tmp_path / "model.mlpackage"
    calls = []

    def parity_report(**kwargs):
        calls.append(kwargs)
        return {
            "variants": [{"format": "coreml-fp16", "path": str(output), "passed": True}]
        }

    monkeypatch.setattr(module, "write_coreml_variant_parity_report", parity_report)
    module.convert(
        "OpenMed/tiny-stub",
        output,
        eval_suite_path="synthetic-suite.json",
        latency_iterations=0,
    )
    assert calls[0]["source_load_reference"] == private_reference
    assert calls[0]["source_model_id"] == "OpenMed/tiny-stub"
    parent = json.loads((tmp_path / "model_coreml_manifest.json").read_text())
    assert parent["variants"][0]["parity"]["passed"] is True
    assert parent["variants"][0]["parity"]["path"] == output.name
    assert private_reference not in json.dumps(parent)


@pytest.mark.parametrize(
    "report_option", ["parity_report_path", "swift_parity_corpus_path"]
)
@pytest.mark.parametrize("inside_quantized", [False, True])
def test_dynamic_report_paths_reject_package_roots_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    report_option: str,
    inside_quantized: bool,
) -> None:
    module, state = _converter(monkeypatch)
    output = tmp_path / "model.mlpackage"
    quantized = tmp_path / "model-int8.mlpackage"
    report_root = quantized if inside_quantized else output
    monkeypatch.setattr(
        module, "write_coreml_variant_parity_report", lambda **kwargs: {"variants": []}
    )
    with pytest.raises(
        ValueError, match=f"{report_option} must be outside CoreML packages"
    ):
        module.convert(
            "OpenMed/tiny-stub",
            output,
            quantize="int8",
            quantized_output_path=quantized,
            eval_suite_path="synthetic-suite.json",
            **{report_option: report_root / "dynamic.json"},
        )
    assert state.config_loads == state.model_loads == state.tokenizer_loads == 0


def test_default_parity_report_rejects_an_enclosing_variant_package_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, state = _converter(monkeypatch)
    quantized = tmp_path / "enclosing.mlpackage"
    output = quantized / "model.mlpackage"
    monkeypatch.setattr(
        module, "write_coreml_variant_parity_report", lambda **kwargs: {"variants": []}
    )
    with pytest.raises(
        ValueError, match="parity_report_path must be outside CoreML packages"
    ):
        module.convert(
            "OpenMed/tiny-stub",
            output,
            quantize="int8",
            quantized_output_path=quantized,
            conversion_manifest_path=tmp_path / "observations.json",
            eval_suite_path="synthetic-suite.json",
        )
    assert state.config_loads == state.model_loads == state.tokenizer_loads == 0


def test_loader_mutation_of_local_source_is_rejected_before_manifest_finalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from openmed.core.export_provenance import resolve_source_context

    module, _ = _converter(monkeypatch)
    source = tmp_path / "private-source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type": "bert"}')
    weights = source / "weights.bin"
    weights.write_bytes(b"AA")
    monkeypatch.setattr(module, "resolve_source_context", resolve_source_context)
    transformers = sys.modules["transformers"]
    original_loader = transformers.AutoModelForTokenClassification.from_pretrained

    def modifying_loader(*args, **kwargs):
        weights.write_bytes(b"BB")
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(
        transformers.AutoModelForTokenClassification,
        "from_pretrained",
        modifying_loader,
    )
    output = tmp_path / "model.mlpackage"
    with pytest.raises(
        ValueError, match="local source changed during conversion"
    ) as error:
        module.convert(str(source), output, latency_iterations=0)
    assert str(source) not in str(error.value)
    assert not (output / module.COREML_MANIFEST_FILENAME).exists()
    assert not (tmp_path / "model_coreml_manifest.json").exists()


def test_default_parity_report_is_written_outside_canonical_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from .test_coreml_convert import _perfect_runner, _write_fixture

    module, _ = _converter(monkeypatch)
    monkeypatch.setattr(
        module, "_hf_token_classification_runner", lambda *a, **k: _perfect_runner
    )
    monkeypatch.setattr(
        module, "_coreml_artifact_runner", lambda *a, **k: _perfect_runner
    )
    fixture = _write_fixture(tmp_path / "synthetic-suite.json")
    output = tmp_path / "model.mlpackage"
    module.convert(
        "OpenMed/tiny-stub", output, eval_suite_path=fixture, latency_iterations=0
    )
    assert (tmp_path / "coreml_parity.json").is_file()
    canonical = json.loads((output / module.COREML_MANIFEST_FILENAME).read_text())
    assert all(
        row["path"] != "coreml_parity.json" for row in canonical["provenance"]["files"]
    )
