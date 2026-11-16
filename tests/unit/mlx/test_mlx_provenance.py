"""Offline provenance acceptance tests using tiny synthetic MLX artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from openmed.mlx import convert as convert_module
from openmed.mlx.artifact import find_tokenizer_files, write_manifest


@pytest.mark.parametrize(
    "manifest_name", ["openmed-mlx.json", "openmed-onnx.json", "openmed-coreml.json"]
)
def test_shared_tokenizer_finder_excludes_export_metadata_and_declared_reports(
    tmp_path, manifest_name
):
    for name in (
        "tokenizer.json",
        "custom-tokenizer-asset.json",
        "recall_delta.json",
        "custom-recall.json",
        "certification.json",
        "nested.json",
        "openmed-mlx.json",
        "openmed-onnx.json",
        "openmed-coreml.json",
    ):
        (tmp_path / name).write_text("{}")
    (tmp_path / manifest_name).write_text(
        json.dumps(
            {
                "recall_delta_path": "custom-recall.json",
                "certification": {"report_path": "certification.json"},
            }
        )
    )
    assert find_tokenizer_files(tmp_path) == [
        "tokenizer.json",
        "custom-tokenizer-asset.json",
        "nested.json",
    ]


@pytest.mark.parametrize(
    "descriptor",
    [
        [],
        {"recall_delta_path": ["custom.json"]},
        {"recall_delta_path": "../custom.json"},
        {"recall_delta_path": "/custom.json"},
        {"certification": ["custom.json"]},
        {"certification": {"report_path": "reports/custom.json"}},
    ],
)
def test_shared_tokenizer_finder_keeps_unrelated_fallback_assets(tmp_path, descriptor):
    (tmp_path / "custom.json").write_text("{}")
    (tmp_path / "metadata.json").write_text('{"recall_delta_path":"custom.json"}')
    (tmp_path / "openmed-mlx.json").write_text(json.dumps(descriptor))
    assert find_tokenizer_files(tmp_path) == ["custom.json", "metadata.json"]


def test_shared_tokenizer_finder_does_not_follow_manifest_symlinks(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "custom.json").write_text("{}")
    outside = tmp_path / "outside.json"
    outside.write_text('{"recall_delta_path":"custom.json"}')
    (artifact / "openmed-onnx.json").symlink_to(outside)
    assert find_tokenizer_files(artifact) == ["custom.json"]


def _write_bundle(root: Path) -> dict:
    root.mkdir(parents=True)
    config = {
        "model_type": "bert",
        "_mlx_weights_format": "safetensors",
        "_commit_hash": "a" * 40,
        "license": "apache-2.0",
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "weights.safetensors").write_bytes(b"synthetic weights only")
    (root / "vocab.txt").write_text("[PAD]\n[UNK]\nfixture\n")
    write_manifest(root, source_model_id="fixture/tiny-bert", config=config)
    return json.loads((root / "openmed-mlx.json").read_text())


def _stub_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    class Tokenizer:
        def __init__(self, reference):
            self.reference = reference

        def save_pretrained(self, output_dir):
            Path(output_dir, "tokenizer.json").write_text("{}")
            Path(output_dir, "tokenizer_config.json").write_text(
                json.dumps({"name_or_path": self.reference})
            )

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda reference, **kwargs: Tokenizer(reference)
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "mlx.core", None)


def test_mlx_manifest_records_source_license_versions_and_file_digests(tmp_path):
    manifest = _write_bundle(tmp_path / "artifact")
    provenance = manifest["provenance"]
    assert provenance["source"] == {
        "model_id": "fixture/tiny-bert",
        "revision": "a" * 40,
        "content_digest": None,
        "license": "apache-2.0",
    }
    assert provenance["toolchain"]["python"]
    assert "openmed" in provenance["toolchain"]
    files = provenance["files"]
    assert [item["path"] for item in files] == [
        "config.json",
        "vocab.txt",
        "weights.safetensors",
    ]
    for item in files:
        contents = (tmp_path / "artifact" / item["path"]).read_bytes()
        assert item["size_bytes"] == len(contents)
        assert item["sha256"] == "sha256:" + hashlib.sha256(contents).hexdigest()


def test_mlx_local_manifest_omits_source_absolute_path(tmp_path):
    checkpoint = tmp_path / "SYNTHETIC_USER" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type":"bert"}')
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "weights.safetensors").write_bytes(b"synthetic")
    path = write_manifest(
        output, source_model_id=str(checkpoint), config={"model_type": "bert"}
    )
    assert str(checkpoint) not in path.read_text()
    assert "SYNTHETIC_USER" not in path.read_text()
    assert json.loads(path.read_text())["provenance"]["source"]["content_digest"]


def test_mlx_convert_tracks_local_source_bytes_through_real_save(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    _stub_tokenizer(monkeypatch)
    checkpoint = tmp_path / "SYNTHETIC_USER" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type":"bert","license":"MIT"}')
    weight_path = checkpoint / "weights.bin"
    weight_path.write_bytes(b"\x01\x02")
    loaded = []

    def load_synthetic(model_id, cache_dir=None):
        loaded.append(model_id)
        config = json.loads(Path(model_id, "config.json").read_text())
        config["_name_or_path"] = model_id
        weights = Path(model_id, "weights.bin").read_bytes()
        return {
            "classifier.weight": np.frombuffer(weights, dtype=np.uint8).copy()
        }, config

    monkeypatch.setattr(convert_module, "convert_weights", load_synthetic)
    first = convert_module.convert(str(checkpoint), tmp_path / "first")
    second = convert_module.convert(str(checkpoint), tmp_path / "second")
    first_bytes = (first / "openmed-mlx.json").read_bytes()
    assert first_bytes == (second / "openmed-mlx.json").read_bytes()
    before = json.loads(first_bytes)["provenance"]["source"]["content_digest"]
    weight_path.write_bytes(b"\x03\x04")
    third = convert_module.convert(str(checkpoint), tmp_path / "third")
    after = json.loads((third / "openmed-mlx.json").read_text())["provenance"][
        "source"
    ]["content_digest"]
    assert before != after
    assert loaded == [str(checkpoint)] * 3
    for root in (first, second, third):
        for name in ("config.json", "tokenizer_config.json", "openmed-mlx.json"):
            assert str(checkpoint) not in (root / name).read_text()
            assert "SYNTHETIC_USER" not in (root / name).read_text()


@pytest.mark.parametrize("mutation", ["modify", "add", "delete"])
def test_mlx_explicit_provenance_verifier_detects_bundle_changes(tmp_path, mutation):
    from openmed.core.export_provenance import verify_export_provenance

    root = tmp_path / "artifact"
    _write_bundle(root)
    verify_export_provenance(root, "openmed-mlx.json")
    weights = root / "weights.safetensors"
    if mutation == "modify":
        contents = weights.read_bytes()
        weights.write_bytes(b"X" + contents[1:])
    elif mutation == "add":
        (root / "extra.bin").write_bytes(b"unexpected synthetic file")
    else:
        weights.unlink()
    with pytest.raises(ValueError):
        verify_export_provenance(root, "openmed-mlx.json")


def test_mlx_convert_loads_resolved_snapshot_once(tmp_path, monkeypatch):
    from openmed.core.export_provenance import SourceContext

    np = pytest.importorskip("numpy")
    _stub_tokenizer(monkeypatch)
    snapshot = tmp_path / "private-cache" / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    calls = []

    def resolve(model_id, *, cache_dir=None, revision=None):
        calls.append((model_id, cache_dir))
        return SourceContext(
            source_model_id="fixture/tiny-bert",
            load_reference=str(snapshot),
            revision="b" * 40,
            license="MIT",
        )

    loaded = []

    def load(model_id, cache_dir=None):
        loaded.append(model_id)
        return {"weight": np.array([1.0])}, {
            "model_type": "bert",
            "_name_or_path": model_id,
        }

    monkeypatch.setattr(convert_module, "resolve_source_context", resolve)
    monkeypatch.setattr(convert_module, "convert_weights", load)
    output = convert_module.convert("fixture/tiny-bert", tmp_path / "artifact")
    assert calls == [("fixture/tiny-bert", None)]
    assert loaded == [str(snapshot)]
    manifest = json.loads((output / "openmed-mlx.json").read_text())
    assert manifest["source_model_id"] == "fixture/tiny-bert"
    assert manifest["provenance"]["source"]["revision"] == "b" * 40
    assert str(snapshot) not in (output / "config.json").read_text()
    assert str(snapshot) not in (output / "tokenizer_config.json").read_text()


def test_mlx_convert_rejects_source_changed_by_loader(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    _stub_tokenizer(monkeypatch)
    checkpoint = tmp_path / "SYNTHETIC_USER" / "checkpoint"
    checkpoint.mkdir(parents=True)
    weight_path = checkpoint / "weights.bin"
    weight_path.write_bytes(b"AA")
    output = tmp_path / "artifact"

    def load_and_modify_source(model_id, cache_dir=None):
        Path(model_id, "weights.bin").write_bytes(b"BB")
        return {"weight": np.array([1.0])}, {"model_type": "bert"}

    monkeypatch.setattr(convert_module, "convert_weights", load_and_modify_source)
    with pytest.raises(
        ValueError, match="local source changed during conversion"
    ) as exc:
        convert_module.convert(str(checkpoint), output)
    assert "SYNTHETIC_USER" not in str(exc.value)
    assert not (output / "openmed-mlx.json").exists()
