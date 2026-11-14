"""Offline tests for Maple ONNX Runtime bundle integrity."""

from __future__ import annotations

import json
import zipfile

import pytest

from openmed.onnx.maple_bundle import (
    _MAX_MANIFEST_BYTES,
    MAPLE_BUNDLE_FILENAME,
    MapleBundleError,
    build_maple_onnx_bundle,
    validate_maple_onnx_bundle,
)


def _write_fake_export(root):
    files = {
        "decoder_model.ort": b"prefill-graph",
        "decoder_with_past_model.ort": b"decode-graph",
        "tokenizer.json": b'{"model":{"type":"BPE"}}',
        "tokenizer_config.json": b'{"model_max_length":131072}',
        "config.json": b'{"architectures":["MapleForCausalLM"]}',
    }
    for name, payload in files.items():
        (root / name).write_bytes(payload)


def _valid_manifest(tmp_path):
    """Build a valid bundle and return it with its manifest JSON text."""

    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    built = build_maple_onnx_bundle(source, tmp_path / "maple-valid.zip")
    with zipfile.ZipFile(built.bundle_path) as archive:
        return built, archive.read(MAPLE_BUNDLE_FILENAME).decode("utf-8")


def _replace_manifest(tmp_path, built, payload):
    """Copy a valid bundle, replacing only its manifest entry."""

    hostile = tmp_path / "maple-hostile.zip"
    with zipfile.ZipFile(built.bundle_path) as source_archive:
        with zipfile.ZipFile(hostile, "w", compression=zipfile.ZIP_STORED) as output:
            output.writestr(MAPLE_BUNDLE_FILENAME, payload)
            for entry in source_archive.infolist()[1:]:
                output.writestr(entry.filename, source_archive.read(entry))
    return hostile


def test_builds_android_bundle_with_manifest_first_and_exact_integrity(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(source, tmp_path / "maple.ommaple.zip")

    assert result.manifest["architecture"] == "MapleForCausalLM"
    assert result.manifest["runtime"] == "onnxruntime-mobile"
    assert result.manifest["quantization"] == "qmoe-4bit-blockwise-128"
    assert result.total_size_bytes > 0
    with zipfile.ZipFile(result.bundle_path) as archive:
        assert archive.namelist()[0] == MAPLE_BUNDLE_FILENAME
        assert all(
            entry.compress_type == zipfile.ZIP_STORED for entry in archive.infolist()
        )
    assert validate_maple_onnx_bundle(result.bundle_path).manifest == result.manifest


def test_builds_web_bundle_with_single_decoder_graph(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(
        source,
        tmp_path / "maple-web.zip",
        prefill_path="decoder_model.ort",
        decode_path=None,
        runtime="onnxruntime-web",
    )

    assert result.manifest["runtime"] == "onnxruntime-web"
    assert result.manifest["graphs"]["decode_path"] is None
    assert result.manifest["cache"] is None


def test_builds_stateless_android_bundle_accepted_by_mobile_contract(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(
        source,
        tmp_path / "maple-mobile-stateless.zip",
        prefill_path="decoder_model.ort",
        decode_path=None,
        runtime="onnxruntime-mobile",
    )

    assert result.manifest["runtime"] == "onnxruntime-mobile"
    assert result.manifest["graphs"]["decode_path"] is None
    assert result.manifest["cache"] is None


def test_builds_unified_cached_decoder_without_duplicate_payload(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(
        source,
        tmp_path / "maple-mobile-unified.zip",
        prefill_path="decoder_model.ort",
        decode_path="decoder_model.ort",
        runtime="onnxruntime-mobile",
    )

    assert result.manifest["graphs"]["prefill_path"] == "decoder_model.ort"
    assert result.manifest["graphs"]["decode_path"] == "decoder_model.ort"
    assert result.manifest["cache"] is not None
    declared_paths = [item["path"] for item in result.manifest["files"]]
    assert declared_paths.count("decoder_model.ort") == 1
    with zipfile.ZipFile(result.bundle_path) as archive:
        assert archive.namelist().count("decoder_model.ort") == 1


def test_refuses_overwrite_and_path_traversal(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    output = tmp_path / "maple.zip"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="overwrite"):
        build_maple_onnx_bundle(source, output)
    with pytest.raises(MapleBundleError, match="unsafe bundle path"):
        build_maple_onnx_bundle(source, tmp_path / "new.zip", prefill_path="../x.ort")


def test_rejects_tampered_payload(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    original = build_maple_onnx_bundle(source, tmp_path / "maple.zip")
    tampered = tmp_path / "tampered.zip"

    with zipfile.ZipFile(original.bundle_path) as source_archive:
        with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as output:
            for entry in source_archive.infolist():
                payload = source_archive.read(entry)
                if entry.filename == "decoder_model.ort":
                    payload = b"tampered-data"
                output.writestr(entry.filename, payload)

    with pytest.raises(MapleBundleError, match="checksum mismatch"):
        validate_maple_onnx_bundle(tampered)


def test_rejects_undeclared_archive_entry(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    original = build_maple_onnx_bundle(source, tmp_path / "maple.zip")
    invalid = tmp_path / "invalid.zip"

    with zipfile.ZipFile(original.bundle_path) as source_archive:
        manifest = json.loads(source_archive.read(MAPLE_BUNDLE_FILENAME))
        with zipfile.ZipFile(invalid, "w", compression=zipfile.ZIP_STORED) as output:
            output.writestr(MAPLE_BUNDLE_FILENAME, json.dumps(manifest))
            for entry in source_archive.infolist()[1:]:
                output.writestr(entry.filename, source_archive.read(entry))
            output.writestr("undeclared.bin", b"not allowed")

    with pytest.raises(MapleBundleError, match="exactly match"):
        validate_maple_onnx_bundle(invalid)


def test_rejects_duplicate_manifest_keys(tmp_path):
    built, text = _valid_manifest(tmp_path)
    duplicate = text.replace(
        '"runtime": "onnxruntime-mobile"',
        '"runtime": "onnxruntime-mobile",\n    "runtime": "sentinel-runtime"',
        1,
    )
    assert duplicate != text
    hostile = _replace_manifest(tmp_path, built, duplicate.encode("utf-8"))

    with pytest.raises(MapleBundleError, match="repeat object keys") as excinfo:
        validate_maple_onnx_bundle(hostile)
    assert "sentinel-runtime" not in str(excinfo.value)


def test_rejects_duplicate_keys_inside_a_nested_manifest_object(tmp_path):
    built, text = _valid_manifest(tmp_path)
    duplicate = text.replace(
        '"graphs": {',
        '"graphs": {\n      "prefill_path": "sentinel.ort",',
        1,
    )
    assert duplicate != text
    hostile = _replace_manifest(tmp_path, built, duplicate.encode("utf-8"))

    with pytest.raises(MapleBundleError, match="repeat object keys") as excinfo:
        validate_maple_onnx_bundle(hostile)
    assert "sentinel.ort" not in str(excinfo.value)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_rejects_non_finite_manifest_numbers(tmp_path, literal):
    built, text = _valid_manifest(tmp_path)
    hostile_text = text.replace(
        '"max_context_tokens": 4096',
        f'"max_context_tokens": {literal}',
        1,
    )
    assert hostile_text != text
    hostile = _replace_manifest(tmp_path, built, hostile_text.encode("utf-8"))

    with pytest.raises(MapleBundleError, match="non-finite"):
        validate_maple_onnx_bundle(hostile)


def test_rejects_manifest_over_the_size_limit(tmp_path):
    built, text = _valid_manifest(tmp_path)
    oversized = text.encode("utf-8") + b" " * (_MAX_MANIFEST_BYTES + 1 - len(text))
    hostile = _replace_manifest(tmp_path, built, oversized)

    with pytest.raises(MapleBundleError, match="too large"):
        validate_maple_onnx_bundle(hostile)


def test_rejects_unknown_runtime(tmp_path):
    built, text = _valid_manifest(tmp_path)
    manifest = json.loads(text)
    manifest["runtime"] = "sentinel-runtime"
    hostile = _replace_manifest(tmp_path, built, json.dumps(manifest).encode("utf-8"))

    with pytest.raises(MapleBundleError, match="runtime must be") as excinfo:
        validate_maple_onnx_bundle(hostile)
    assert "sentinel-runtime" not in str(excinfo.value)


def test_rejects_malformed_sha256_digest(tmp_path):
    built, text = _valid_manifest(tmp_path)
    manifest = json.loads(text)
    manifest["files"][0]["sha256"] = "sentinel-digest"
    hostile = _replace_manifest(tmp_path, built, json.dumps(manifest).encode("utf-8"))

    with pytest.raises(MapleBundleError, match="valid SHA-256") as excinfo:
        validate_maple_onnx_bundle(hostile)
    assert "sentinel-digest" not in str(excinfo.value)
    assert manifest["files"][0]["path"] not in str(excinfo.value)


def test_rejects_short_source_revision(tmp_path):
    built, text = _valid_manifest(tmp_path)
    manifest = json.loads(text)
    manifest["source_revision"] = "sentinel-revision"
    hostile = _replace_manifest(tmp_path, built, json.dumps(manifest).encode("utf-8"))

    with pytest.raises(MapleBundleError, match="immutable lowercase commit") as excinfo:
        validate_maple_onnx_bundle(hostile)
    assert "sentinel-revision" not in str(excinfo.value)
