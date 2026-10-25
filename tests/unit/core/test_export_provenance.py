"""Synthetic, offline coverage for portable export provenance."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from fnmatch import fnmatch
from pathlib import Path
from unittest.mock import Mock

import pytest

from openmed.core.export_provenance import (
    SourceContext,
    build_provenance,
    context_from_metadata,
    resolve_source_context,
    verify_export_provenance,
    write_canonical_manifest,
)

MANIFEST = "openmed-test.json"
REVISION = "a" * 40


def _checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "bert", "license": "apache-2.0"}),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"synthetic weights")
    return root


def _artifact(root: Path) -> Path:
    root.mkdir()
    (root / "model.onnx").write_bytes(b"synthetic graph")
    nested = root / "model.mlpackage" / "Data"
    nested.mkdir(parents=True)
    (nested / "weights.bin").write_bytes(b"external weights")
    (root / "empty.txt").touch()
    return root


def _write(root: Path) -> Path:
    return write_canonical_manifest(
        root,
        MANIFEST,
        {"format": "synthetic"},
        source=SourceContext("OpenMed/synthetic", "internal", REVISION),
        backend="onnx",
    )


def test_local_source_identity_depends_on_content_not_location(tmp_path: Path) -> None:
    first = resolve_source_context(str(_checkpoint(tmp_path / "first")))
    second_root = _checkpoint(tmp_path / "second")
    second = resolve_source_context(str(second_root))
    assert first.source_digest == second.source_digest
    assert first.source_model_id == f"local:{first.source_digest}"
    assert first.revision is None
    assert first.license == "apache-2.0"
    assert first.load_reference != second.load_reference
    (second_root / "model.safetensors").write_bytes(b"Synthetic weights")
    assert resolve_source_context(str(second_root)).source_digest != first.source_digest


@pytest.mark.parametrize("mutation", ["add", "delete"])
def test_local_source_inventory_binds_additions_and_deletions(
    tmp_path: Path, mutation: str
) -> None:
    root = _checkpoint(tmp_path / "source")
    before = resolve_source_context(str(root)).source_digest
    if mutation == "add":
        (root / "new.txt").write_bytes(b"new")
    else:
        (root / "model.safetensors").unlink()
    assert resolve_source_context(str(root)).source_digest != before


def test_local_source_symlinks_are_rejected(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path / "source")
    (root / "alias").symlink_to(root / "config.json")
    with pytest.raises(ValueError, match="symlink"):
        resolve_source_context(str(root))


def test_remote_resolves_once_and_allows_hf_cache_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _checkpoint(tmp_path / REVISION)
    blob = tmp_path / "blob"
    blob.write_bytes(b"cached HF weight blob")
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(blob)
    download = Mock(return_value=str(snapshot))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=download),
    )
    context = resolve_source_context(
        "OpenMed/synthetic", cache_dir="synthetic-cache", revision="main"
    )
    download.assert_called_once()
    assert download.call_args.kwargs["repo_id"] == "OpenMed/synthetic"
    assert download.call_args.kwargs["cache_dir"] == "synthetic-cache"
    assert download.call_args.kwargs["revision"] == "main"
    assert context.revision == REVISION
    assert context.source_digest is None
    assert context.load_reference == str(snapshot)
    assert context.license == "apache-2.0"


def test_remote_rejects_mutable_snapshot_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _checkpoint(tmp_path / "main")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=Mock(return_value=str(snapshot))),
    )
    with pytest.raises(ValueError, match="revision"):
        resolve_source_context("OpenMed/synthetic")


@pytest.mark.parametrize(
    "model_id",
    [
        "https://user:secret@example.test/model",
        "../private",
        "/missing/path",
        "user/repo?token=secret",
    ],
)
def test_credential_or_path_model_references_are_rejected(model_id: str) -> None:
    with pytest.raises(ValueError):
        context_from_metadata(model_id)


def test_metadata_context_does_not_resolve_or_echo_private_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            snapshot_download=Mock(side_effect=AssertionError("network"))
        ),
    )
    context = context_from_metadata(
        "OpenMed/synthetic",
        {
            "_commit_hash": REVISION,
            "license": "mit",
            "_name_or_path": "/private/user/token",
        },
    )
    assert context == SourceContext(
        "OpenMed/synthetic", "OpenMed/synthetic", REVISION, license="mit"
    )


def test_readme_scalar_license_only(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path / "source")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "README.md").write_text(
        "---\nlicense: 'bsd-3-clause'\n---\nsecret raw model card\n", encoding="utf-8"
    )
    assert resolve_source_context(str(root)).license == "bsd-3-clause"
    (root / "README.md").write_text("license: mit\n", encoding="utf-8")
    assert resolve_source_context(str(root)).license is None


def test_canonical_manifest_and_nested_inventory_are_reproducible(
    tmp_path: Path,
) -> None:
    first_root = _artifact(tmp_path / "first")
    second_root = _artifact(tmp_path / "second")
    first = _write(first_root)
    second = _write(second_root)
    assert first.read_bytes() == second.read_bytes()
    before = first.read_bytes()
    _write(first_root)
    assert first.read_bytes() == before
    assert before.endswith(b"\n")
    provenance = verify_export_provenance(first_root, MANIFEST)
    assert provenance["schema_version"] == 1
    assert [item["path"] for item in provenance["files"]] == [
        "empty.txt",
        "model.mlpackage/Data/weights.bin",
        "model.onnx",
    ]
    record = provenance["files"][-1]
    assert (
        record["sha256"] == "sha256:" + hashlib.sha256(b"synthetic graph").hexdigest()
    )
    assert record["size_bytes"] == len(b"synthetic graph")
    assert str(tmp_path) not in before.decode()
    assert "internal" not in before.decode()


@pytest.mark.parametrize("mutation", ["modify", "add", "delete", "nested_add"])
def test_verifier_detects_content_and_exact_inventory_changes(
    tmp_path: Path, mutation: str
) -> None:
    root = _artifact(tmp_path / "artifact")
    _write(root)
    if mutation == "modify":
        (root / "model.onnx").write_bytes(b"Synthetic graph")
    elif mutation == "add":
        (root / "extra.bin").write_bytes(b"extra")
    elif mutation == "nested_add":
        (root / "model.mlpackage" / "Data" / "extra.bin").write_bytes(b"extra")
    else:
        (root / "model.onnx").unlink()
    with pytest.raises(ValueError):
        verify_export_provenance(root, MANIFEST)


@pytest.mark.parametrize("directory", [False, True])
def test_output_symlinks_are_rejected(tmp_path: Path, directory: bool) -> None:
    root = _artifact(tmp_path / "artifact")
    target = tmp_path / "outside"
    target.mkdir()
    (target / "secret").write_bytes(b"secret")
    (root / "alias").symlink_to(target if directory else target / "secret")
    with pytest.raises(ValueError, match="symlink"):
        _write(root)


@pytest.mark.parametrize(
    "filename",
    [
        "../escape.json",
        "/absolute.json",
        "nested/file.json",
        "a\\b.json",
        ".",
        "C:bad.json",
    ],
)
def test_manifest_filename_cannot_escape_root(tmp_path: Path, filename: str) -> None:
    with pytest.raises(ValueError):
        write_canonical_manifest(
            tmp_path,
            filename,
            {},
            source=context_from_metadata("OpenMed/test"),
            backend="mlx",
        )
    with pytest.raises(ValueError):
        verify_export_provenance(tmp_path, filename)


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "a/../b", "a\\b", "a//b", "C:/secret"]
)
def test_verifier_rejects_unsafe_declared_paths(tmp_path: Path, path: str) -> None:
    root = _artifact(tmp_path / "artifact")
    manifest = _write(root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["provenance"]["files"][0]["path"] = path
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_export_provenance(root, MANIFEST)


@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate",
        "negative_size",
        "boolean_size",
        "bad_digest",
        "missing",
        "wrong_schema",
    ],
)
def test_verifier_rejects_invalid_provenance(tmp_path: Path, invalid: str) -> None:
    root = _artifact(tmp_path / "artifact")
    manifest = _write(root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    if invalid == "duplicate":
        provenance["files"].append(dict(provenance["files"][0]))
    elif invalid == "negative_size":
        provenance["files"][0]["size_bytes"] = -1
    elif invalid == "boolean_size":
        provenance["files"][0]["size_bytes"] = True
    elif invalid == "bad_digest":
        provenance["files"][0]["sha256"] = "not-a-digest"
    elif invalid == "wrong_schema":
        provenance["schema_version"] = 2
    else:
        del payload["provenance"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_export_provenance(root, MANIFEST)


def test_missing_versions_are_null_and_no_backend_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib import metadata

    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    result = build_provenance(
        tmp_path,
        source=context_from_metadata("OpenMed/test"),
        manifest_filename=MANIFEST,
        backend="coreml",
    )
    assert result["toolchain"]["coremltools"] is None
    assert result["toolchain"]["openmed"] is None
    assert result["source"]["license"] is None
    assert "numpy" not in result["toolchain"]


@pytest.mark.parametrize("target", ["manifest", "root"])
def test_existing_output_manifest_or_root_symlink_rejected(
    tmp_path: Path, target: str
) -> None:
    root = _artifact(tmp_path / "artifact")
    if target == "manifest":
        outside = tmp_path / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        (root / MANIFEST).symlink_to(outside)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    with pytest.raises(ValueError, match="symlink"):
        _write(root)
    with pytest.raises(ValueError, match="symlink"):
        verify_export_provenance(root, MANIFEST)


def test_unclosed_readme_frontmatter_is_not_license_metadata(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path / "source")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "README.md").write_text("---\nlicense: mit\n", encoding="utf-8")
    assert resolve_source_context(str(root)).license is None


def test_source_private_paths_are_hashed_but_never_serialized(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path / "private-user-checkpoint")
    (source / "config.json").write_text(
        json.dumps({"_name_or_path": "/private/user/token-secret"}), encoding="utf-8"
    )
    context = resolve_source_context(str(source))
    root = _artifact(tmp_path / "artifact")
    manifest = write_canonical_manifest(
        root, MANIFEST, {}, source=context, backend="mlx"
    )
    data = manifest.read_text(encoding="utf-8")
    assert "private-user" not in data
    assert "token-secret" not in data
    assert (
        verify_export_provenance(root, MANIFEST)["source"]["content_digest"]
        == context.source_digest
    )


def test_duplicate_json_keys_and_nonfinite_payload_are_rejected(tmp_path: Path) -> None:
    root = _artifact(tmp_path / "artifact")
    manifest = _write(root)
    manifest.write_text('{"provenance": {}, "provenance": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        verify_export_provenance(root, MANIFEST)
    with pytest.raises(ValueError):
        write_canonical_manifest(
            root,
            MANIFEST,
            {"measurement": float("nan")},
            source=context_from_metadata("OpenMed/test"),
            backend="onnx",
        )


@pytest.mark.parametrize("model_id", ["OpenMed/model_", "OpenMed/_model", "_org/model"])
def test_ordinary_repository_ids_allow_underscores_at_edges(model_id: str) -> None:
    assert context_from_metadata(model_id).source_model_id == model_id


@pytest.mark.parametrize(
    "model_id", ["OpenMed/模型", "OpenMed/modèle", "组织/模型", "a" * 97 + "/model"]
)
def test_repository_validation_preserves_standard_unicode_and_namespace_ids(
    model_id: str,
) -> None:
    assert context_from_metadata(model_id).source_model_id == model_id


@pytest.mark.parametrize("model_id", ["OpenMed/model.git", "OpenMed/" + "a" * 97])
def test_invalid_repository_suffix_or_length_is_rejected(model_id: str) -> None:
    with pytest.raises(ValueError):
        context_from_metadata(model_id)


@pytest.mark.parametrize(
    "scalar",
    ["mit # SPDX identifier", "'mit' # SPDX identifier", '"mit" # SPDX identifier'],
)
def test_frontmatter_scalar_license_allows_yaml_inline_comments(
    tmp_path: Path, scalar: str
) -> None:
    root = _checkpoint(tmp_path / "source")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "README.md").write_text(f"---\nlicense: {scalar}\n---\n", encoding="utf-8")
    assert resolve_source_context(str(root)).license == "mit"


@pytest.mark.parametrize("mutation", ["modify", "add", "delete"])
def test_local_source_stability_guard_rejects_changes(
    tmp_path: Path, mutation: str
) -> None:
    from openmed.core import export_provenance

    root = _checkpoint(tmp_path / "private-source")
    context = resolve_source_context(str(root))
    if mutation == "modify":
        (root / "model.safetensors").write_bytes(b"Synthetic weights")
    elif mutation == "add":
        (root / "extra.bin").write_bytes(b"extra")
    else:
        (root / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="local source changed") as raised:
        export_provenance.assert_source_unchanged(context)
    assert str(root) not in str(raised.value)


def test_local_source_stability_guard_is_content_based(tmp_path: Path) -> None:
    from openmed.core import export_provenance

    first = _checkpoint(tmp_path / "first")
    second = _checkpoint(tmp_path / "second")
    context = resolve_source_context(str(first))
    moved = SourceContext(
        context.source_model_id,
        str(second),
        source_digest=context.source_digest,
        license=context.license,
    )
    export_provenance.assert_source_unchanged(context)
    export_provenance.assert_source_unchanged(moved)


def test_remote_source_stability_guard_does_not_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openmed.core import export_provenance

    download = Mock(side_effect=AssertionError("network"))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=download),
    )
    export_provenance.assert_source_unchanged(
        SourceContext("OpenMed/synthetic", "/private/cache/nonexistent", REVISION)
    )
    download.assert_not_called()


def test_remote_snapshot_patterns_select_checkpoint_inputs_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _checkpoint(tmp_path / REVISION)
    download = Mock(return_value=str(snapshot))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=download),
    )
    resolve_source_context("OpenMed/synthetic")
    patterns = download.call_args.kwargs["allow_patterns"]
    assert isinstance(patterns, list)

    def selected(name: str) -> bool:
        return any(fnmatch(name, pattern) for pattern in patterns)

    for name in (
        "config.json",
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model.bin.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "spiece.model",
        "spm.model",
        "tokenizer.model",
        "README.md",
        "LICENSE",
        "LICENSE.md",
    ):
        assert selected(name), name
    for name in (
        "onnx/model.onnx",
        "onnx/model.safetensors",
        "coreml/model.mlpackage/Data/weights.bin",
        "datasets/train.json",
        "data.parquet",
        "training_args.bin",
        "tf_model.h5",
        "flax_model.msgpack",
    ):
        assert not selected(name), name
