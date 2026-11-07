"""Synthetic, offline coverage for portable export provenance."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
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


def test_observation_open_failure_keeps_previous_file_without_creating_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openmed.core import export_provenance

    observation = tmp_path / "artifact_onnx_observations.json"
    observation.write_text('{"validation": {"passed": true}}\n')
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def failed_open(*args, **kwargs):
        raise OSError("synthetic open failure")

    with monkeypatch.context() as patch:
        patch.setattr(export_provenance.os, "open", failed_open)
        with pytest.raises(OSError, match="synthetic open failure"):
            export_provenance.write_observation_report(observation, {"updated": True})
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_observations_reject_hardlinks_without_changing_either_file(
    tmp_path: Path,
) -> None:
    from openmed.core.export_provenance import write_observation_report

    runtime = tmp_path / "config.json"
    original = b'{"model_type": "bert"}\n'
    runtime.write_bytes(original)
    observation = tmp_path / "artifact_mlx_observations.json"
    try:
        observation.hardlink_to(runtime)
    except (NotImplementedError, OSError):
        pytest.skip("hard links unavailable")
    with pytest.raises(ValueError, match="observations"):
        write_observation_report(observation, {"generated_at": "2026-01-01T00:00:00Z"})
    assert runtime.read_bytes() == original
    assert observation.read_bytes() == original


def test_observation_json_policy_preserves_previous_file_on_serialization_error(
    tmp_path: Path,
) -> None:
    from openmed.core.export_provenance import write_observation_report

    observation = tmp_path / "artifact_onnx_observations.json"
    payload = {"label": "synthetic λ", "latency": float("nan")}
    write_observation_report(observation, payload)
    before = observation.read_bytes()
    assert b"\\u03bb" in before
    assert b"NaN" in before
    assert before.endswith(b"\n")
    with pytest.raises(ValueError):
        write_observation_report(observation, payload, allow_nan=False)
    assert observation.read_bytes() == before
    assert list(tmp_path.iterdir()) == [observation]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission and umask semantics")
@pytest.mark.parametrize(
    ("existing_mode", "mask"),
    [
        (None, 0o022),
        (None, 0o002),
        (None, 0o077),
        (None, 0o027),
        (0o640, 0o022),
        (0o640, 0o077),
        (0o664, 0o022),
        (0o664, 0o077),
        (0o600, 0o022),
        (0o600, 0o077),
    ],
)
def test_observation_permissions_match_legacy_writes_in_child_process(
    tmp_path: Path, existing_mode: int | None, mask: int
) -> None:
    # Keep the pytest process and any concurrent threads' umask unchanged.
    script = """
import json
import os
from pathlib import Path
import stat
import sys

from openmed.core.export_provenance import write_observation_report

root = Path(sys.argv[1])
mask, existing_mode = json.loads(sys.argv[2])
os.umask(mask)
paths = [root / name for name in ("observation.json", "pathlib.json", "open.json")]
if existing_mode is not None:
    for path in paths:
        path.write_text('{}')
        path.chmod(existing_mode)
    original_ids = [(path.stat().st_uid, path.stat().st_gid) for path in paths]
write_observation_report(paths[0], {"updated": True})
paths[1].write_text('{"updated": true}')
with paths[2].open('w') as handle:
    handle.write('{"updated": true}')
if existing_mode is not None:
    assert [(path.stat().st_uid, path.stat().st_gid) for path in paths] == original_ids
print(json.dumps([stat.S_IMODE(path.stat().st_mode) for path in paths]))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            json.dumps([mask, existing_mode]),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    actual, pathlib_mode, open_mode = json.loads(result.stdout)
    expected = existing_mode if existing_mode is not None else 0o666 & ~mask
    assert pathlib_mode == open_mode == expected
    assert actual == expected
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "observation.json",
        "open.json",
        "pathlib.json",
    ]


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="non-root POSIX write-permission semantics",
)
def test_readonly_observations_refuse_replacement_without_changing_file(
    tmp_path: Path,
) -> None:
    from openmed.core.export_provenance import write_observation_report

    observation = tmp_path / "artifact_onnx_observations.json"
    original = b'{"validation": {"passed": true}}\n'
    observation.write_bytes(original)
    observation.chmod(0o440)
    with pytest.raises(PermissionError):
        observation.write_text('{"updated": true}')
    with pytest.raises(PermissionError):
        write_observation_report(observation, {"updated": True})
    assert observation.read_bytes() == original
    assert observation.stat().st_mode & 0o777 == 0o440
    assert list(tmp_path.iterdir()) == [observation]


@pytest.mark.parametrize("target_kind", ["file", "symlink", "dangling"])
def test_new_observations_reject_target_created_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_kind: str
) -> None:
    from openmed.core import export_provenance

    observation = tmp_path / "artifact_onnx_observations.json"
    original = b"synthetic racing writer"
    runtime = tmp_path / "config.json"
    runtime.write_bytes(b'{"model_type": "bert"}\n')
    runtime_before = runtime.read_bytes()
    absent = tmp_path / "absent.json"
    native_open = os.open

    def racing_open(path, flags, mode=0o777):
        if target_kind == "file":
            observation.write_bytes(original)
        else:
            try:
                observation.symlink_to(runtime if target_kind == "symlink" else absent)
            except (NotImplementedError, OSError):
                pytest.skip("symlinks unavailable")
        return native_open(path, flags, mode)

    monkeypatch.setattr(export_provenance.os, "open", racing_open)
    with pytest.raises(FileExistsError):
        export_provenance.write_observation_report(observation, {"updated": True})
    assert runtime.read_bytes() == runtime_before
    assert not absent.exists()
    if target_kind == "file":
        assert observation.read_bytes() == original
    else:
        assert observation.is_symlink()


@pytest.mark.parametrize("same_target", [False, True])
def test_observations_reject_a_symlink_swap_without_nofollow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_target: bool
) -> None:
    from openmed.core import export_provenance

    observation = tmp_path / "artifact_onnx_observations.json"
    original = b'{"validation": {"passed": true}}\n'
    observation.write_bytes(original)
    saved = tmp_path / "original-observation.json"
    runtime = tmp_path / "config.json"
    runtime.write_bytes(b'{"model_type": "bert"}\n')
    runtime_before = runtime.read_bytes()
    native_open = os.open
    opened = []

    def racing_open(path, flags, mode=0o777):
        observation.rename(saved)
        try:
            observation.symlink_to(saved if same_target else runtime)
        except (NotImplementedError, OSError):
            pytest.skip("symlinks unavailable")
        descriptor = native_open(path, flags, mode)
        opened.append(descriptor)
        return descriptor

    with monkeypatch.context() as patch:
        patch.delattr(export_provenance.os, "O_NOFOLLOW", raising=False)
        patch.setattr(export_provenance.os, "open", racing_open)
        with pytest.raises(ValueError, match="observations"):
            export_provenance.write_observation_report(observation, {"updated": True})
    assert saved.read_bytes() == original
    assert runtime.read_bytes() == runtime_before
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_observation_fifo_swap_is_rejected_without_blocking(tmp_path: Path) -> None:
    script = """
import errno
import os
from pathlib import Path
import sys
from unittest.mock import patch

from openmed.core import export_provenance

root = Path(sys.argv[1])
observation = root / 'observation.json'
saved = root / 'original.json'
observation.write_text('{}')
native_open = os.open
def swap_to_fifo(path, flags, mode=0o777):
    observation.rename(saved)
    os.mkfifo(observation)
    return native_open(path, flags, mode)
with patch.object(export_provenance.os, 'open', swap_to_fifo):
    try:
        export_provenance.write_observation_report(observation, {'updated': True})
    except OSError as error:
        assert error.errno in {errno.ENXIO, errno.EAGAIN}
    else:
        raise AssertionError('FIFO must be rejected')
assert saved.read_text() == '{}'
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )


@pytest.mark.parametrize("existing", [False, True])
def test_observation_write_failure_closes_descriptor_and_cleans_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    from openmed.core import export_provenance

    observation = tmp_path / "artifact_onnx_observations.json"
    if existing:
        observation.write_text("{}")
    primary = OSError("synthetic full device")
    opened = []

    class FailedWriter:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, content):
            raise primary

    def failing_writer(descriptor, *args, **kwargs):
        opened.append(descriptor)
        return FailedWriter()

    monkeypatch.setattr(export_provenance.os, "fdopen", failing_writer)
    with pytest.raises(OSError, match="synthetic full device") as caught:
        export_provenance.write_observation_report(observation, {"updated": True})
    assert caught.value is primary
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
    if existing:
        # Match legacy open('w'): a subsequent write failure may leave truncation.
        assert observation.read_bytes() == b""
    elif os.name == "nt" and observation.exists():
        # Closing a Windows handle may refresh generation metadata; retain it safely.
        assert observation.read_bytes() == b""
    else:
        assert not observation.exists()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX unlink of an open file")
def test_new_observation_write_failure_does_not_delete_racing_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openmed.core import export_provenance

    observation = tmp_path / "artifact_onnx_observations.json"
    replacement = b"synthetic racing replacement"
    opened = []

    class RacingWriter:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, content):
            observation.unlink()
            observation.write_bytes(replacement)
            raise OSError("synthetic write interruption")

    def failing_writer(descriptor, *args, **kwargs):
        opened.append(descriptor)
        return RacingWriter()

    monkeypatch.setattr(export_provenance.os, "fdopen", failing_writer)
    with pytest.raises(OSError, match="synthetic write interruption"):
        export_provenance.write_observation_report(observation, {"updated": True})
    assert observation.read_bytes() == replacement
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.parametrize(
    ("platform", "cleanup_case"),
    [
        pytest.param(
            "linux",
            "normal",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="requires POSIX unlink of an open file"
            ),
        ),
        ("win32", "normal"),
        pytest.param(
            "linux",
            "denied",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="POSIX cleanup-order simulation"
            ),
        ),
        ("win32", "denied"),
        ("win32", "replacement"),
        ("win32", "partial"),
        ("win32", "unknown_generation"),
    ],
)
def test_observation_failure_cleanup_preserves_primary_error_and_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, cleanup_case: str
) -> None:
    from openmed.core import export_provenance

    observation = tmp_path / "artifact_onnx_observations.json"
    primary = OSError("synthetic full device")
    replacement = b"synthetic racing replacement"
    opened = []
    failed = []
    native_close = os.close
    native_unlink = Path.unlink
    native_fstat = os.fstat

    class FailedWriter:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, content):
            if cleanup_case == "partial":
                os.write(opened[0], b"partial")
            failed.append(True)
            raise primary

    def failing_writer(descriptor, *args, **kwargs):
        opened.append(descriptor)
        return FailedWriter()

    def close_descriptor(descriptor):
        native_close(descriptor)
        if cleanup_case == "replacement" and descriptor == opened[0]:
            native_unlink(observation, missing_ok=True)
            observation.write_bytes(replacement)

    def file_state(descriptor):
        info = native_fstat(descriptor)
        if failed and cleanup_case == "unknown_generation":
            return types.SimpleNamespace(
                st_dev=info.st_dev,
                st_ino=info.st_ino,
                st_mode=info.st_mode,
                st_nlink=info.st_nlink,
                st_size=info.st_size,
            )
        return info

    def platform_unlink(path, *args, **kwargs):
        if path == observation:
            if cleanup_case == "denied":
                raise PermissionError("synthetic cleanup permission denial")
            try:
                os.fstat(opened[0])
            except OSError:
                descriptor_open = False
            else:
                descriptor_open = True
            if platform == "win32" and descriptor_open:
                raise PermissionError("simulated Windows file sharing violation")
            if platform == "linux":
                assert descriptor_open, "POSIX cleanup must keep the inode pinned"
        return native_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(export_provenance.sys, "platform", platform)
        patch.setattr(export_provenance.os, "fdopen", failing_writer)
        patch.setattr(export_provenance.os, "close", close_descriptor)
        patch.setattr(export_provenance.os, "fstat", file_state)
        patch.setattr(Path, "unlink", platform_unlink)
        with pytest.raises(OSError) as caught:
            export_provenance.write_observation_report(observation, {"updated": True})
    assert caught.value is primary
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
    if cleanup_case == "replacement":
        assert observation.read_bytes() == replacement
    elif cleanup_case in {"denied", "unknown_generation"}:
        assert observation.read_bytes() == b""
    elif os.name == "nt" and observation.exists():
        # Only real Windows may refresh its own file's metadata during close.
        assert observation.read_bytes() == (
            b"partial" if cleanup_case == "partial" else b""
        )
    else:
        assert not observation.exists()


@pytest.fixture
def posix_observation_acl():
    """Use libacl only for synthetic fixtures; do not require ACL command tools."""
    import ctypes
    import ctypes.util
    import errno

    if sys.platform != "linux":
        pytest.skip("Linux POSIX ACL fixture")
    library = ctypes.util.find_library("acl")
    if library is None:
        pytest.skip("libacl unavailable")
    api = ctypes.CDLL(library, use_errno=True)
    api.acl_from_text.argtypes = [ctypes.c_char_p]
    api.acl_from_text.restype = ctypes.c_void_p
    api.acl_set_file.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
    api.acl_set_file.restype = ctypes.c_int
    api.acl_get_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
    api.acl_get_file.restype = ctypes.c_void_p
    api.acl_to_text.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ssize_t)]
    api.acl_to_text.restype = ctypes.c_void_p
    api.acl_free.argtypes = [ctypes.c_void_p]
    api.acl_free.restype = ctypes.c_int

    def set_acl(path, text, *, default=False):
        handle = api.acl_from_text(text.encode())
        if not handle:
            raise OSError(ctypes.get_errno(), "acl_from_text failed")
        try:
            if api.acl_set_file(
                os.fsencode(path), 0x4000 if default else 0x8000, handle
            ):
                code = ctypes.get_errno()
                if code in {errno.ENOTSUP, errno.EOPNOTSUPP}:
                    pytest.skip("fixture filesystem lacks POSIX ACLs")
                raise OSError(code, "acl_set_file failed")
        finally:
            api.acl_free(handle)

    def get_acl(path):
        handle = api.acl_get_file(os.fsencode(path), 0x8000)
        if not handle:
            raise OSError(ctypes.get_errno(), "acl_get_file failed")
        try:
            length = ctypes.c_ssize_t()
            text = api.acl_to_text(handle, ctypes.byref(length))
            if not text:
                raise OSError(ctypes.get_errno(), "acl_to_text failed")
            try:
                return ctypes.string_at(text, length.value)
            finally:
                api.acl_free(text)
        finally:
            api.acl_free(handle)

    return set_acl, get_acl


@pytest.mark.parametrize(
    "case", ["existing_acl", "existing_parent_acl", "new_parent_acl"]
)
def test_observation_acl_matches_legacy_writes(
    tmp_path: Path, posix_observation_acl, case: str
) -> None:
    from openmed.core.export_provenance import write_observation_report

    set_acl, get_acl = posix_observation_acl
    observation, legacy = (
        tmp_path / name for name in ("observation.json", "legacy.json")
    )
    existing = case != "new_parent_acl"
    if existing:
        for path in (observation, legacy):
            path.write_text("{}")
            path.chmod(0o640)
    if case == "existing_acl":
        for path in (observation, legacy):
            set_acl(
                path,
                f"user::rw-,user:{os.getuid()}:r--,group::---,mask::r--,other::---",
            )
    else:
        set_acl(
            tmp_path,
            f"user::rwx,user:{os.getuid()}:r--,group::r--,mask::r--,other::---",
            default=True,
        )
    before = observation.stat() if existing else None
    original_acl = get_acl(observation) if existing else None
    legacy.write_text('{"updated": true}\n')
    write_observation_report(observation, {"updated": True})
    assert get_acl(observation) == get_acl(legacy)
    assert observation.stat().st_mode == legacy.stat().st_mode
    if existing:
        assert get_acl(observation) == original_acl
        assert (observation.stat().st_dev, observation.stat().st_ino) == (
            before.st_dev,
            before.st_ino,
        )


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="non-root POSIX directory permissions",
)
def test_observations_update_writable_file_in_readonly_parent(tmp_path: Path) -> None:
    from openmed.core.export_provenance import write_observation_report

    observation = tmp_path / "observation.json"
    legacy = tmp_path / "legacy.json"
    for path in (observation, legacy):
        path.write_text("{}")
        path.chmod(0o640)
    before = observation.stat()
    tmp_path.chmod(0o550)
    try:
        legacy.write_text('{"updated": true}\n')
        write_observation_report(observation, {"updated": True})
        assert json.loads(observation.read_text()) == json.loads(legacy.read_text())
        assert (observation.stat().st_dev, observation.stat().st_ino) == (
            before.st_dev,
            before.st_ino,
        )
    finally:
        tmp_path.chmod(0o700)
