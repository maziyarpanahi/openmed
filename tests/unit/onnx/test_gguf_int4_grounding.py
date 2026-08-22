"""Tests for Q4_K_M GGUF grounding export and subprocess runtime."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import pytest


def _export_module():
    return importlib.import_module("openmed.onnx.gguf_int4_export")


def _runtime_module():
    return importlib.import_module("openmed.onnx.gguf_embed_runtime")


class _MappingEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


class _FlippingEmbedder:
    def __init__(
        self, first: dict[str, list[float]], second: dict[str, list[float]]
    ) -> None:
        self._vectors = (first, second)
        self._calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._vectors[min(self._calls, 1)]
        self._calls += 1
        return [vectors[text] for text in texts]


@pytest.fixture
def retrieval_fixture():
    queries = ["synthetic query one", "synthetic query two"]
    passages = [
        "synthetic passage one",
        "synthetic passage two",
        "synthetic passage three",
        "synthetic passage four",
    ]
    fp16_vectors = {
        queries[0]: [1.0, 0.0],
        queries[1]: [0.0, 1.0],
        passages[0]: [1.0, 0.0],
        passages[1]: [0.9, 0.1],
        passages[2]: [0.0, 1.0],
        passages[3]: [0.1, 0.9],
    }
    rejected_vectors = {
        queries[0]: [1.0, 0.0],
        queries[1]: [0.0, 1.0],
        passages[0]: [0.0, 1.0],
        passages[1]: [0.1, 0.9],
        passages[2]: [1.0, 0.0],
        passages[3]: [0.9, 0.1],
    }
    return queries, passages, fp16_vectors, rejected_vectors


def _clock(*values: float):
    samples = iter(values)
    return lambda: next(samples)


def test_subprocess_runtime_returns_vectors_without_in_process_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runtime_module()
    model = tmp_path / "model-q4_k_m.gguf"
    model.write_bytes(b"GGUF")
    executable = tmp_path / "llama-embedding"
    executable.write_bytes(b"stub")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"embedding": [1.0, 2.0, 3.0]}',
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_stdin_prompt_path", lambda: "/dev/stdin")
    runtime = module.LlamaCppEmbeddingRuntime(model, executable)

    vectors = runtime.encode(["synthetic query", "synthetic passage"])

    assert vectors == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert len(calls) == 2
    command, kwargs = calls[0]
    assert command[0] == str(executable)
    assert command[command.index("--model") + 1] == str(model)
    assert "--embeddings" not in command
    assert command[command.index("--embd-output-format") + 1] == "raw"
    assert command[command.index("--file") + 1] == "/dev/stdin"
    assert "synthetic query" not in command
    assert kwargs["input"] == "synthetic query"
    assert kwargs["check"] is False
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert "shell" not in kwargs


def test_subprocess_runtime_parses_llama_style_bracket_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runtime_module()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    executable = tmp_path / "llama-embedding"
    executable.write_bytes(b"stub")

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="embedding: [-0.25, 0.5, 1e-1]\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(module, "_stdin_prompt_path", lambda: "/dev/stdin")

    assert module.LlamaCppEmbeddingRuntime(model, executable).embed("synthetic") == [
        -0.25,
        0.5,
        0.1,
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"command": ["embedding"], "timeout_seconds": float("nan")},
        {"command": ["embedding"], "context_size": True},
        {"command": ["embedding"], "batch_size": 65_537},
        {"command": "embedding"},
        {"command": ["embedding"], "extra_args": "--verbose"},
    ],
)
def test_subprocess_runtime_rejects_unbounded_or_ambiguous_options(
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    module = _runtime_module()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")

    with pytest.raises(ValueError):
        module.LlamaCppEmbeddingRuntime(model, **kwargs)


def test_subprocess_runtime_rejects_nul_text_and_oversized_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runtime_module()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    runtime = module.LlamaCppEmbeddingRuntime(model, command=["embedding"])

    with pytest.raises(ValueError, match="NUL"):
        runtime.embed("synthetic\0prompt")

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="x" * (module.MAX_EMBEDDING_OUTPUT_CHARS + 1),
            stderr="",
        ),
    )
    with pytest.raises(module.GgufEmbeddingRuntimeError, match="finite vector"):
        runtime.embed("synthetic prompt")


def test_recall_certificate_passes_for_equivalent_rankings(retrieval_fixture) -> None:
    module = _export_module()
    queries, passages, fp16_vectors, _ = retrieval_fixture

    certification = module.certify_gguf_grounding(
        _MappingEmbedder(fp16_vectors),
        _MappingEmbedder(fp16_vectors),
        queries=queries,
        passages=passages,
        top_k=2,
        recall_delta_tolerance=0.0,
        clock=_clock(1.0, 1.006, 2.0, 2.003),
    )

    assert certification.gate.passed is True
    assert certification.gate.gate == "G4"
    assert certification.gate.deterministic is True
    assert certification.gate.mean_top_k_overlap == 1.0
    assert certification.gate.recall_delta == 0.0
    assert certification.fp16_latency.count == 6
    assert certification.int4_latency.p50_ms == pytest.approx(0.5)


def test_recall_certificate_rejects_rank_drift_and_nondeterminism(
    retrieval_fixture,
) -> None:
    module = _export_module()
    queries, passages, fp16_vectors, rejected_vectors = retrieval_fixture

    rejected = module.certify_gguf_grounding(
        _MappingEmbedder(fp16_vectors),
        _MappingEmbedder(rejected_vectors),
        queries=queries,
        passages=passages,
        top_k=2,
        recall_delta_tolerance=0.05,
        clock=_clock(0.0, 0.0, 0.0, 0.0),
    )
    assert rejected.gate.passed is False
    assert rejected.gate.recall_delta == 1.0

    flipping = _FlippingEmbedder(fp16_vectors, rejected_vectors)
    nondeterministic = module.certify_gguf_grounding(
        _MappingEmbedder(fp16_vectors),
        flipping,
        queries=queries,
        passages=passages,
        top_k=2,
        recall_delta_tolerance=1.0,
        clock=_clock(0.0, 0.0, 0.0, 0.0),
    )
    assert nondeterministic.gate.passed is False
    assert nondeterministic.gate.deterministic is False


def test_recall_certificate_rejects_ambiguous_limits_and_vectors(
    retrieval_fixture,
) -> None:
    module = _export_module()
    queries, passages, fp16_vectors, _ = retrieval_fixture
    embedder = _MappingEmbedder(fp16_vectors)

    with pytest.raises(ValueError, match="top_k"):
        module.certify_gguf_grounding(
            embedder,
            embedder,
            queries=queries,
            passages=passages,
            top_k=True,
        )
    with pytest.raises(ValueError, match="finite"):
        module.certify_gguf_grounding(
            embedder,
            embedder,
            queries=queries,
            passages=passages,
            recall_delta_tolerance=float("nan"),
        )

    invalid_vectors = dict(fp16_vectors)
    invalid_vectors[queries[0]] = [True, 0.0]
    with pytest.raises(ValueError, match="numeric"):
        module.certify_gguf_grounding(
            _MappingEmbedder(invalid_vectors),
            embedder,
            queries=queries,
            passages=passages,
        )


def test_export_quantizes_staged_om195_bundle_and_writes_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retrieval_fixture,
) -> None:
    module = _export_module()
    queries, passages, fp16_vectors, _ = retrieval_fixture
    model = tmp_path / "synthetic-grounding-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["BertModel"],
                "model_type": "bert",
                "_commit_hash": "synthetic-revision",
            }
        ),
        encoding="utf-8",
    )
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("# synthetic converter\n", encoding="utf-8")
    quantizer = tmp_path / "llama-quantize"
    quantizer.write_bytes(b"synthetic quantizer")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        commands.append(command)
        if "--outtype" in command:
            output = Path(command[command.index("--outfile") + 1])
            output.write_bytes(
                b"GGUF-" + command[command.index("--outtype") + 1].encode()
            )
        else:
            assert command[-1] == "Q4_K_M"
            Path(command[2]).write_bytes(b"GGUF-Q4_K_M")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output = tmp_path / "export"
    result = module.export_gguf_int4(
        model,
        output,
        converter_path=converter,
        quantizer_path=quantizer,
        source_model_id="local/synthetic-grounding-model",
        fp16_embedder=_MappingEmbedder(fp16_vectors),
        int4_embedder=_MappingEmbedder(fp16_vectors),
        queries=queries,
        passages=passages,
        top_k=2,
        recall_delta_tolerance=0.0,
        clock=_clock(1.0, 1.006, 2.0, 2.003),
    )

    assert [command[-1] for command in commands] == ["f16", "q8_0", "Q4_K_M"]
    assert result.q4_k_m_path.read_bytes() == b"GGUF-Q4_K_M"
    assert result.recall_gate.passed is True
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["quantization"]["scheme"] == "Q4_K_M"
    assert manifest["certification"]["gate"] == "G4"
    q4_record = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["path"] == module.GGUF_INT4_FILENAME
    )
    expected_sha256 = hashlib.sha256(b"GGUF-Q4_K_M").hexdigest()
    assert q4_record["sha256"] == expected_sha256
    assert q4_record["size_bytes"] == len(b"GGUF-Q4_K_M")
    report = json.loads(result.benchmark_report_path.read_text(encoding="utf-8"))
    assert report["metadata"]["certified"] is True
    assert report["metrics"]["resources"]["model_size_bytes"] > 0
    assert report["metrics"]["resources"]["artifact_sha256"] == expected_sha256
    assert q4_record["sha256"] == report["metadata"]["artifact_sha256"]
    assert q4_record["size_bytes"] == report["metrics"]["resources"]["model_size_bytes"]
    module.validate_gguf_int4_artifact(output)

    result.q4_k_m_path.write_bytes(b"GGUF-tampered")
    with pytest.raises(module.GgufInt4Rejected, match="consistent passing G4"):
        module.validate_gguf_int4_artifact(output)


def test_export_rejects_failed_gate_before_publishing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retrieval_fixture,
) -> None:
    module = _export_module()
    queries, passages, fp16_vectors, rejected_vectors = retrieval_fixture
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"architectures": ["BertModel"], "model_type": "bert"}),
        encoding="utf-8",
    )
    converter = tmp_path / "converter.py"
    converter.write_text("# synthetic\n", encoding="utf-8")
    quantizer = tmp_path / "quantizer"
    quantizer.write_bytes(b"synthetic")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        if "--outtype" in command:
            Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF")
        else:
            Path(command[2]).write_bytes(b"GGUF-Q4")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output = tmp_path / "export"
    with pytest.raises(module.GgufInt4Rejected, match="rejected"):
        module.export_gguf_int4(
            model,
            output,
            converter_path=converter,
            quantizer_path=quantizer,
            fp16_embedder=_MappingEmbedder(fp16_vectors),
            int4_embedder=_MappingEmbedder(rejected_vectors),
            queries=queries,
            passages=passages,
            top_k=2,
            recall_delta_tolerance=0.05,
            clock=_clock(0.0, 0.0, 0.0, 0.0),
        )

    assert not (output / module.GGUF_INT4_FILENAME).exists()
    assert not (output / "openmed-gguf.json").exists()


def test_loader_fails_closed_when_report_is_tampered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retrieval_fixture,
) -> None:
    module = _export_module()
    queries, passages, fp16_vectors, _ = retrieval_fixture
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"architectures": ["BertModel"], "model_type": "bert"}),
        encoding="utf-8",
    )
    converter = tmp_path / "converter.py"
    converter.write_text("# synthetic\n", encoding="utf-8")
    quantizer = tmp_path / "quantizer"
    quantizer.write_bytes(b"synthetic")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        if "--outtype" in command:
            Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF")
        else:
            Path(command[2]).write_bytes(b"GGUF-Q4")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output = tmp_path / "export"
    module.export_gguf_int4(
        model,
        output,
        converter_path=converter,
        quantizer_path=quantizer,
        fp16_embedder=_MappingEmbedder(fp16_vectors),
        int4_embedder=_MappingEmbedder(fp16_vectors),
        queries=queries,
        passages=passages,
        top_k=2,
        recall_delta_tolerance=0.0,
        clock=_clock(0.0, 0.0, 0.0, 0.0),
    )
    report_path = output / module.GGUF_INT4_BENCHMARK_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["metrics"]["retrieval"]["passed"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(module.GgufInt4Rejected, match="certification"):
        module.load_gguf_grounding_embedder(
            output,
            command=["synthetic-llama-embedding"],
        )


def test_export_requires_recall_evidence_or_local_runtime(tmp_path: Path) -> None:
    module = _export_module()

    with pytest.raises(module.GgufInt4Rejected, match="requires fp16 and int4"):
        module.export_gguf_int4(tmp_path / "model", tmp_path / "output")


def test_quantizer_rejects_non_gguf_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _export_module()
    quantizer = tmp_path / "quantizer"
    input_path = tmp_path / "input.gguf"
    output_path = tmp_path / "output.gguf"
    quantizer.write_bytes(b"stub")
    input_path.write_bytes(b"GGUF-input")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        Path(command[2]).write_bytes(b"NOPE")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(module.GgufInt4ExportError, match="invalid GGUF header"):
        module._run_quantizer(
            quantizer,
            input_path=input_path,
            output_path=output_path,
            timeout_seconds=1.0,
        )


def test_bundle_publish_restores_previous_outputs_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _export_module()
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    for name in ("a", "b"):
        (staging / name).write_text(f"new-{name}", encoding="utf-8")
        (destination / name).write_text(f"old-{name}", encoding="utf-8")

    real_replace = module.os.replace
    failed = False

    def fail_second_publish(source: Path, target: Path) -> None:
        nonlocal failed
        if source == staging / "b" and target == destination / "b" and not failed:
            failed = True
            raise OSError("synthetic publish failure")
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_second_publish)
    with pytest.raises(module.GgufInt4ExportError, match="previous outputs restored"):
        module._publish_staged_bundle(
            staging,
            destination,
            {"a", "b"},
            overwrite=True,
        )

    assert (destination / "a").read_text(encoding="utf-8") == "old-a"
    assert (destination / "b").read_text(encoding="utf-8") == "old-b"


def test_certification_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    module = _export_module()
    payload = tmp_path / "evidence.json"
    payload.write_text('{"passed": true, "passed": false}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        module._read_json(payload)


def test_runtime_rejects_args_that_can_bypass_model_or_prompt(
    tmp_path: Path,
) -> None:
    module = _runtime_module()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    executable = tmp_path / "llama-embedding"
    executable.write_bytes(b"stub")

    for extra_args in (
        ["--model", "other.gguf"],
        ["--model=other.gguf"],
        ["--prompt", "other text"],
        ["--log-file", "prompt.log"],
        ["--log-prompts-dir", "prompt-logs"],
        ["--rpc", "remote.example:50052"],
        ["--rpc=remote.example:50052"],
        ["--"],
    ):
        with pytest.raises(ValueError, match="protected option"):
            module.LlamaCppEmbeddingRuntime(
                model,
                executable,
                extra_args=extra_args,
            )


def test_runtime_removes_inherited_llama_argument_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runtime_module()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    executable = tmp_path / "llama-embedding"
    executable.write_bytes(b"stub")
    captured_environment: dict[str, str] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        captured_environment.update(environment)
        return subprocess.CompletedProcess(command, 0, stdout="1 2", stderr="")

    monkeypatch.setenv("OPENMED_GGUF_TEST_MARKER", "preserved")
    monkeypatch.setenv("LLAMA_ARG_RPC", "remote.example:50052")
    monkeypatch.setenv("LLAMA_ARG_LOG_FILE", str(tmp_path / "prompt.log"))
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_stdin_prompt_path", lambda: "/dev/stdin")

    module.LlamaCppEmbeddingRuntime(model, executable).embed("synthetic")

    assert captured_environment["OPENMED_GGUF_TEST_MARKER"] == "preserved"
    assert not any(key.startswith("LLAMA_ARG_") for key in captured_environment)


def test_runtime_rejects_inconsistent_vector_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runtime_module()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    executable = tmp_path / "llama-embedding"
    executable.write_bytes(b"stub")
    outputs = iter(("1 2", "1 2 3"))

    monkeypatch.setattr(module, "_stdin_prompt_path", lambda: "/dev/stdin")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=next(outputs),
            stderr="",
        ),
    )

    with pytest.raises(module.GgufEmbeddingRuntimeError, match="dimensions"):
        module.LlamaCppEmbeddingRuntime(model, executable).encode(["one", "two"])


def test_certification_rejects_overflowing_vectors(retrieval_fixture) -> None:
    module = _export_module()
    queries, passages, fp16_vectors, _ = retrieval_fixture
    overflowing = dict(fp16_vectors)
    overflowing[queries[0]] = [10**400, 1]

    with pytest.raises(ValueError, match="numeric values"):
        module.certify_gguf_grounding(
            _MappingEmbedder(overflowing),
            _MappingEmbedder(fp16_vectors),
            queries=queries,
            passages=passages,
            top_k=2,
        )


def test_loader_rejects_replaced_q4_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retrieval_fixture,
) -> None:
    module = _export_module()
    queries, passages, fp16_vectors, _ = retrieval_fixture
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"architectures": ["BertModel"], "model_type": "bert"}),
        encoding="utf-8",
    )
    converter = tmp_path / "converter.py"
    converter.write_text("# synthetic\n", encoding="utf-8")
    quantizer = tmp_path / "quantizer"
    quantizer.write_bytes(b"synthetic")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        if "--outtype" in command:
            Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF")
        else:
            Path(command[2]).write_bytes(b"GGUF-Q4")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output = tmp_path / "export"
    result = module.export_gguf_int4(
        model,
        output,
        converter_path=converter,
        quantizer_path=quantizer,
        fp16_embedder=_MappingEmbedder(fp16_vectors),
        int4_embedder=_MappingEmbedder(fp16_vectors),
        queries=queries,
        passages=passages,
        top_k=2,
        recall_delta_tolerance=0.0,
    )
    result.q4_k_m_path.write_bytes(b"replacement")

    with pytest.raises(module.GgufInt4Rejected, match="certification"):
        module.validate_gguf_int4_artifact(output)
