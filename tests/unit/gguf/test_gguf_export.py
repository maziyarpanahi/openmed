"""Smoke tests for embedding-backbone GGUF export."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import pytest

from openmed.gguf.convert import UnsupportedGgufModelError, convert

convert_module = importlib.import_module("openmed.gguf.convert")


def test_convert_accepts_embedding_stub_and_records_gguf_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _write_model_config(
        tmp_path / "sapbert-stub",
        {
            "architectures": ["BertModel"],
            "model_type": "bert",
            "pipeline_tag": "feature-extraction",
            "_commit_hash": "abc123",
        },
    )
    converter = _write_converter_stub(tmp_path)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        output = Path(command[command.index("--outfile") + 1])
        outtype = command[command.index("--outtype") + 1]
        output.write_bytes(b"GGUF" + outtype.encode("ascii"))
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 3600.0
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(convert_module.subprocess, "run", fake_run)

    output = tmp_path / "export"
    result = convert(
        model,
        output,
        converter_path=converter,
        source_model_id="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    )

    assert result.formats == ["gguf"]
    assert [artifact.path.name for artifact in result.artifacts] == [
        "model-f16.gguf",
        "model-q8_0.gguf",
    ]
    assert [command[command.index("--outtype") + 1] for command in commands] == [
        "f16",
        "q8_0",
    ]
    assert all(
        artifact.path.read_bytes().startswith(b"GGUF") for artifact in result.artifacts
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "openmed-gguf"
    assert manifest["format_version"] == 1
    assert manifest["formats"] == ["gguf"]
    assert manifest["task"] == "feature-extraction"
    assert manifest["family"] == "bert"
    assert manifest["architecture"] == "BertModel"
    assert manifest["source_model_id"] == (
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    assert manifest["source_revision"] == "abc123"
    assert manifest["converter"]["outtypes"] == ["f16", "q8_0"]
    assert manifest["artifacts"] == [
        {
            "format": "gguf",
            "path": "model-f16.gguf",
            "precision": "float16",
            "quantization": "F16",
        },
        {
            "format": "gguf",
            "path": "model-q8_0.gguf",
            "precision": "q8_0",
            "quantization": "Q8_0",
        },
    ]
    assert json.loads((output / "config.json").read_text()) == json.loads(
        (model / "config.json").read_text()
    )


@pytest.mark.parametrize(
    "config",
    [
        {
            "architectures": ["BertForTokenClassification"],
            "model_type": "bert",
        },
        {
            "architectures": ["BertModel"],
            "model_type": "bert",
            "pipeline_tag": "token-classification",
        },
    ],
)
def test_convert_rejects_token_classification_before_running_converter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: dict[str, object],
) -> None:
    model = _write_model_config(tmp_path / "token-classifier", config)
    converter = _write_converter_stub(tmp_path)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("converter must not run for token-classification")

    monkeypatch.setattr(convert_module.subprocess, "run", fail_if_called)

    with pytest.raises(
        UnsupportedGgufModelError,
        match="does not support token-classification heads.*classifier head",
    ):
        convert(model, tmp_path / "export", converter_path=converter)


def test_convert_requires_both_outputs_before_publishing_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _write_model_config(
        tmp_path / "embedding",
        {"architectures": ["BertModel"], "model_type": "bert"},
    )
    converter = _write_converter_stub(tmp_path)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        outtype = command[command.index("--outtype") + 1]
        if outtype == "f16":
            Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(convert_module.subprocess, "run", fake_run)

    output = tmp_path / "export"
    with pytest.raises(
        RuntimeError,
        match="q8_0 conversion did not write model-q8_0.gguf",
    ):
        convert(model, output, converter_path=converter)

    assert not (output / "model-f16.gguf").exists()
    assert not (output / "openmed-gguf.json").exists()


def _write_model_config(path: Path, config: dict[str, object]) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    return path


def _write_converter_stub(tmp_path: Path) -> Path:
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("# stub\n", encoding="utf-8")
    return converter
