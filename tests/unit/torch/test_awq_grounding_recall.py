"""Tests for AWQ grounding quantization and retrieval recall certification."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    return importlib.import_module("openmed.torch.awq_grounding")


class _MappingEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


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


def _install_fake_quantizer(monkeypatch, module, captured: dict[str, object]):
    from openmed.torch.quantize_awq import AwqQuantizationResult, write_quant_config

    def fake_quantize_awq(
        model_name: str,
        calibration_texts: list[str],
        out_dir: str | Path,
        **kwargs,
    ) -> AwqQuantizationResult:
        captured["calibration_texts"] = list(calibration_texts)
        captured["quantize_kwargs"] = kwargs
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model.safetensors").write_bytes(b"certified-awq-artifact")
        quant_config_path = write_quant_config(
            output_dir,
            source_model_id=model_name,
            source_revision="pinned-source-sha",
            w_bit=4,
            group_size=int(kwargs["group_size"]),
            calibration_texts=calibration_texts,
            autoawq_quant_config={
                "zero_point": True,
                "q_group_size": int(kwargs["group_size"]),
                "w_bit": 4,
                "version": "GEMM",
            },
            config={"model_type": "synthetic-embedder"},
        )
        return AwqQuantizationResult(
            output_dir=output_dir,
            quant_config_path=quant_config_path,
            source_model_id=model_name,
            source_revision="pinned-source-sha",
            calibration_sample_count=len(calibration_texts),
        )

    monkeypatch.setattr(module, "quantize_awq", fake_quantize_awq)


def test_top_k_overlap_certificate_passes_for_equivalent_rankings(
    retrieval_fixture,
) -> None:
    module = _module()
    queries, passages, fp16_vectors, _ = retrieval_fixture

    certification = module.certify_grounding_recall(
        _MappingEmbedder(fp16_vectors),
        _MappingEmbedder(fp16_vectors),
        queries=queries,
        passages=passages,
        top_k=2,
        recall_delta_tolerance=0.0,
        clock=_clock(1.0, 1.006, 2.0, 2.003),
    )

    assert certification.gate.passed is True
    assert certification.gate.mean_top_k_overlap == 1.0
    assert certification.gate.recall_delta == 0.0
    assert certification.gate.per_query_overlap == (1.0, 1.0)
    assert certification.fp16_latency.count == 6
    assert certification.fp16_latency.p50_ms == pytest.approx(1.0)
    assert certification.awq_latency.p50_ms == pytest.approx(0.5)


def test_top_k_overlap_certificate_fails_closed_on_rank_drift(
    retrieval_fixture,
) -> None:
    module = _module()
    queries, passages, fp16_vectors, rejected_vectors = retrieval_fixture

    certification = module.certify_grounding_recall(
        _MappingEmbedder(fp16_vectors),
        _MappingEmbedder(rejected_vectors),
        queries=queries,
        passages=passages,
        top_k=2,
        recall_delta_tolerance=0.05,
        clock=_clock(1.0, 1.0, 2.0, 2.0),
    )

    assert certification.gate.passed is False
    assert certification.gate.mean_top_k_overlap == 0.0
    assert certification.gate.recall_delta == 1.0


@pytest.mark.parametrize(
    "vectors, message",
    [
        ([[1.0, 0.0]], "output count"),
        ([[1.0, 0.0]] * 5 + [[float("nan"), 0.0]], "finite"),
        ([[0.0, 0.0]] * 6, "non-zero norm"),
    ],
)
def test_certificate_rejects_invalid_embedding_evidence(vectors, message) -> None:
    module = _module()

    with pytest.raises(ValueError, match=message):
        module.certify_grounding_recall(
            lambda texts: vectors,
            lambda texts: [[1.0, 0.0]] * len(texts),
            queries=["query one", "query two"],
            passages=["passage one", "passage two", "passage three", "passage four"],
            clock=_clock(0.0, 0.0),
        )


def test_certificate_rejects_mismatched_fp16_and_awq_dimensions() -> None:
    module = _module()
    texts = ["query", "passage one", "passage two", "passage three"]

    with pytest.raises(ValueError, match="same vector dimension"):
        module.certify_grounding_recall(
            lambda values: [[1.0, 0.0]] * len(values),
            lambda values: [[1.0, 0.0, 0.0]] * len(values),
            queries=texts[:1],
            passages=texts[1:],
            clock=_clock(0.0, 0.0, 0.0, 0.0),
        )


def test_recipe_uses_only_shared_synthetic_calibration_and_writes_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retrieval_fixture,
) -> None:
    module = _module()
    queries, passages, fp16_vectors, _ = retrieval_fixture
    captured: dict[str, object] = {}
    calibration_texts = [
        "Synthetic calibration passage alpha.",
        "Synthetic calibration passage beta.",
    ]
    monkeypatch.setattr(
        module,
        "load_awq_calibration_texts",
        lambda limit=None: calibration_texts[:limit],
    )
    _install_fake_quantizer(monkeypatch, module, captured)

    result = module.quantize_awq_grounding(
        "OpenMed/synthetic-grounding-embedder",
        tmp_path,
        calibration_limit=2,
        certification_queries=queries,
        certification_passages=passages,
        top_k=2,
        recall_delta_tolerance=0.0,
        fp16_embedder=_MappingEmbedder(fp16_vectors),
        awq_embedder=_MappingEmbedder(fp16_vectors),
        clock=_clock(1.0, 1.006, 2.0, 2.003),
    )

    assert captured["calibration_texts"] == calibration_texts
    assert result.recall_gate.passed is True
    assert result.source_revision == "pinned-source-sha"
    assert result.calibration_sha256 == module.calibration_texts_sha256(
        calibration_texts
    )
    benchmark = json.loads(result.benchmark_report_path.read_text(encoding="utf-8"))
    quant_config = json.loads(result.quant_config_path.read_text(encoding="utf-8"))
    assert benchmark["suite"] == "grounding-awq-retrieval"
    assert benchmark["metrics"]["retrieval"]["passed"] is True
    assert benchmark["metrics"]["retrieval"]["metric"] == "top_k_overlap"
    assert benchmark["metrics"]["resources"]["model_size_bytes"] > 0
    assert benchmark["metrics"]["latency"]["fp16"]["count"] == 6
    assert benchmark["metadata"]["calibration"]["source"] == (
        module.QUANTIZATION_CALIBRATION_SOURCE
    )
    assert benchmark["metadata"]["calibration"]["sha256"] == (result.calibration_sha256)
    assert quant_config["task"] == "feature-extraction"
    assert quant_config["grounding"]["certified"] is True
    assert quant_config["grounding"]["benchmark_report_path"] == (
        "grounding_awq_benchmark.json"
    )
    serialized_metadata = result.quant_config_path.read_text(
        encoding="utf-8"
    ) + result.benchmark_report_path.read_text(encoding="utf-8")
    assert calibration_texts[0] not in serialized_metadata


def test_recipe_persists_failed_gate_and_rejects_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retrieval_fixture,
) -> None:
    module = _module()
    queries, passages, fp16_vectors, rejected_vectors = retrieval_fixture
    captured: dict[str, object] = {}
    _install_fake_quantizer(monkeypatch, module, captured)

    with pytest.raises(module.GroundingAwqRejected) as excinfo:
        module.quantize_awq_grounding(
            "OpenMed/synthetic-grounding-embedder",
            tmp_path,
            certification_queries=queries,
            certification_passages=passages,
            top_k=2,
            fp16_embedder=_MappingEmbedder(fp16_vectors),
            awq_embedder=_MappingEmbedder(rejected_vectors),
            clock=_clock(1.0, 1.0, 2.0, 2.0),
        )

    assert excinfo.value.gate is not None
    assert excinfo.value.gate.passed is False
    benchmark = json.loads(
        (tmp_path / module.GROUNDING_BENCHMARK_FILENAME).read_text(encoding="utf-8")
    )
    quant_config = json.loads(
        (tmp_path / module.QUANT_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    assert benchmark["metadata"]["certified"] is False
    assert quant_config["grounding"]["certified"] is False
    with pytest.raises(module.GroundingAwqRejected, match="passing recall evidence"):
        module.load_awq_grounding_embedder(tmp_path)


def test_loader_admits_only_consistently_certified_grounding_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retrieval_fixture,
) -> None:
    module = _module()
    queries, passages, fp16_vectors, _ = retrieval_fixture
    captured: dict[str, object] = {}
    _install_fake_quantizer(monkeypatch, module, captured)
    module.quantize_awq_grounding(
        "OpenMed/synthetic-grounding-embedder",
        tmp_path,
        certification_queries=queries,
        certification_passages=passages,
        top_k=2,
        fp16_embedder=_MappingEmbedder(fp16_vectors),
        awq_embedder=_MappingEmbedder(fp16_vectors),
        clock=_clock(1.0, 1.0, 2.0, 2.0),
    )
    sentinel = object()
    monkeypatch.setattr(
        module,
        "_load_awq_grounding_embedder_unchecked",
        lambda *args, **kwargs: sentinel,
    )

    assert module.load_awq_grounding_embedder(tmp_path) is sentinel

    benchmark_path = tmp_path / module.GROUNDING_BENCHMARK_FILENAME
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark["metrics"]["retrieval"]["recall_delta"] = 0.5
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    with pytest.raises(module.GroundingAwqRejected, match="passing recall evidence"):
        module.load_awq_grounding_embedder(tmp_path)


def test_fixture_digest_is_ordered_and_reproducible() -> None:
    module = _module()
    queries = ["synthetic query a", "synthetic query b"]
    passages = ["synthetic passage a", "synthetic passage b"]

    first = module.grounding_fixture_sha256(queries, passages)
    second = module.grounding_fixture_sha256(queries, passages)

    assert first == second
    assert first != module.grounding_fixture_sha256(list(reversed(queries)), passages)
    assert first != module.grounding_fixture_sha256(queries, list(reversed(passages)))


def test_loader_rejects_missing_certificate_before_optional_imports(
    tmp_path: Path,
) -> None:
    module = _module()

    with pytest.raises(module.GroundingAwqRejected, match="quant_config.json"):
        module.load_awq_grounding_embedder(tmp_path)


def test_unchecked_loader_disables_fixed_batch_fused_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    captured: dict[str, object] = {}
    sentinel_model = object()

    class FakeAutoAWQForCausalLM:
        @classmethod
        def from_quantized(cls, artifact_dir: str, **kwargs):
            captured["artifact_dir"] = artifact_dir
            captured["model_kwargs"] = kwargs
            return sentinel_model

    tokenizer = SimpleNamespace(pad_token_id=0)
    monkeypatch.setitem(
        sys.modules,
        "awq",
        SimpleNamespace(AutoAWQForCausalLM=FakeAutoAWQForCausalLM),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=object()),
        ),
    )
    monkeypatch.setattr(
        module,
        "get_tokenizer_with_loader",
        lambda *args, **kwargs: tokenizer,
    )

    embedder = module._load_awq_grounding_embedder_unchecked(
        tmp_path,
        trust_remote_code=False,
        device_map="auto",
        max_length=128,
    )

    assert embedder.model is sentinel_model
    assert captured["artifact_dir"] == str(tmp_path)
    assert captured["model_kwargs"] == {
        "trust_remote_code": False,
        "safetensors": True,
        "fuse_layers": False,
        "device_map": "auto",
    }
