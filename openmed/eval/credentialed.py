"""Offline aggregate-only evaluation of user-supplied de-identification data."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from openmed.eval.aggregate_report import project_aggregate_report
from openmed.eval.datasets.i2b2 import I2B2_DUA_NAME, load_i2b2_deid
from openmed.eval.harness import ModelRunner, run_benchmark


class CredentialedEvalError(ValueError):
    """A controlled reason code safe to print in a credentialed evaluation."""


def _output_file(output: str | Path) -> Path:
    path = Path(output).expanduser()
    if path.is_symlink() or path.exists():
        raise CredentialedEvalError("output_exists")
    path = path.resolve()
    if not path.parent.is_dir() or any(
        (parent / ".git").exists() or parent.name == ".git" for parent in path.parents
    ):
        raise CredentialedEvalError("output_location_refused")
    return path


def _cached_snapshot(model_id: str, revision: str) -> Path:
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.utils import validate_repo_id

    try:
        validate_repo_id(model_id)
    except ValueError:
        raise CredentialedEvalError("invalid_model_id") from None
    if not re.fullmatch(r"[\w.-]+", revision) or revision in {".", ".."}:
        raise CredentialedEvalError("invalid_revision")
    config = try_to_load_from_cache(model_id, "config.json", revision=revision)
    if not isinstance(config, str):
        raise CredentialedEvalError("model_not_cached")
    snapshot = Path(config).parent
    if not re.fullmatch(r"[a-f0-9]{40}", snapshot.name):
        raise CredentialedEvalError("invalid_cached_revision")

    return snapshot


def _load_cached_model(model_id: str, revision: str) -> tuple[str, ModelRunner]:
    snapshot = _cached_snapshot(model_id, revision)

    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    options = {"local_files_only": True, "trust_remote_code": False}
    try:
        tokenizer = AutoTokenizer.from_pretrained(snapshot, **options)
        model = AutoModelForTokenClassification.from_pretrained(
            snapshot, use_safetensors=True, **options
        )
        # Some tokenizers use a sentinel for unbounded length. The encoder
        # still has a finite context and long records must be scored in full.
        tokenizer.model_max_length = min(
            tokenizer.model_max_length,
            getattr(
                model.config, "max_position_embeddings", tokenizer.model_max_length
            ),
        )
        width = tokenizer.model_max_length - tokenizer.num_special_tokens_to_add()
        if not tokenizer.is_fast or not 1 <= width < 2**31:
            raise ValueError
        classifier = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            aggregation_strategy="simple",
            stride=min(64, width // 2),
        )
    except (OSError, ValueError, TypeError):
        raise CredentialedEvalError("cached_model_unusable") from None

    def predict(fixture, model_name, device):
        return classifier(fixture.text)

    return hashlib.sha256(snapshot.name.encode("ascii")).hexdigest(), predict


_VARIANTS = {"onnx-fp32", "onnx-fp16", "onnx-int8", "mlx-fp", "mlx-4bit", "mlx-8bit"}


def _format_specs(formats, model_id, revision):
    specs = {}
    for spec in formats:
        name, separator, ref = spec.partition("=")
        if name not in _VARIANTS or name in specs:
            raise CredentialedEvalError("invalid_format")
        repo, pin = model_id, revision
        if separator:
            repo, _, pin = ref.rpartition("@")
            if not repo or not re.fullmatch(r"[0-9a-f]{40}", pin):
                raise CredentialedEvalError("invalid_format_revision")
        specs[name] = (repo, pin)
    return specs


def _mlx_spans(classifier, text):
    """Decode one document after combining overlapping native token windows."""
    import mlx.core as mx

    width = classifier.max_length
    if not width or width <= classifier.tokenizer.num_special_tokens_to_add():
        raise CredentialedEvalError("cached_model_unusable")
    stride = min(64, (width - classifier.tokenizer.num_special_tokens_to_add()) // 2)
    windows = classifier.tokenizer(
        text,
        max_length=width,
        stride=stride,
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    best = {}
    start = 0
    for ids, mask, offsets, special in zip(
        windows["input_ids"],
        windows["attention_mask"],
        windows["offset_mapping"],
        windows["special_tokens_mask"],
        strict=True,
    ):
        logits = classifier.model(
            mx.array([ids]), attention_mask=mx.array([mask], dtype=mx.float32)
        )
        probabilities = mx.softmax(logits[0], axis=-1).tolist()
        content = [i for i, marker in enumerate(special) if not marker]
        for index, pos in enumerate(content):
            score = min(index, len(content) - 1 - index)
            key = start + index
            if key not in best or best[key][0] < score:
                vector = probabilities[pos]
                best[key] = (
                    score,
                    max(range(len(vector)), key=vector.__getitem__),
                    vector,
                    offsets[pos],
                )
        start += len(content) - stride
    tokens = [best[k] for k in sorted(best)]
    return classifier._decode_grouped(
        [r[1] for r in tokens], [r[2] for r in tokens], [r[3] for r in tokens], text
    )


def _load_format(name, repo, revision, model_id, reference_digest):
    snapshot = _cached_snapshot(repo, revision)
    digest = hashlib.sha256(snapshot.name.encode("ascii")).hexdigest()
    family, variant = name.split("-", 1)
    manifest_file = snapshot / f"openmed-{family}.json"
    manifest = json.loads(manifest_file.read_bytes()) if manifest_file.is_file() else {}
    tokenizer_snapshot = snapshot
    if repo != model_id or digest != reference_digest:
        source = manifest.get("provenance", {}).get("source", {})
        source_pin = source.get("revision")
        if (
            source.get("model_id") != model_id
            or not isinstance(source_pin, str)
            or hashlib.sha256(source_pin.encode()).hexdigest() != reference_digest
        ):
            raise CredentialedEvalError("source_revision_unverified")
        if family == "mlx" and not (snapshot / "tokenizer.json").is_file():
            tokenizer_snapshot = _cached_snapshot(model_id, source_pin)
    if family == "onnx":
        from openmed.onnx.inference import MODEL_FILENAMES, OnnxModel

        if not any(
            (directory / MODEL_FILENAMES[variant]).is_file()
            for directory in (snapshot, snapshot / "onnx")
        ):
            raise CredentialedEvalError("model_not_cached")
        engine = OnnxModel(snapshot, variant=variant)

        def predict(fixture, _model, _device):
            return [entity.to_dict() for entity in engine.predict(fixture.text)]
    else:
        if name not in manifest.get("formats", []) or not any(
            (snapshot / file).is_file()
            for file in ("weights.npz", "weights.safetensors")
        ):
            raise CredentialedEvalError("model_not_cached")
        from openmed.mlx.inference import MLXTokenClassificationPipeline

        if not (tokenizer_snapshot / "tokenizer.json").is_file():
            raise CredentialedEvalError("tokenizer_not_cached")
        engine = MLXTokenClassificationPipeline(
            snapshot, tokenizer_name=str(tokenizer_snapshot)
        )

        def predict(fixture, _model, _device):
            return _mlx_spans(engine, fixture.text)

    return digest, predict


def _score(fixtures, suite, year, model_id, runner):
    benchmark = run_benchmark(
        fixtures, suite=suite, model_name=model_id, runner=runner, cache_dir=None
    )
    return project_aggregate_report(benchmark, fixtures, corpus=suite, year=year)


def _format_report(name, spec, model_id, reference, fixtures, suite, year):
    try:
        digest, runner = _load_format(
            name, *spec, model_id, reference["revision_digest"]
        )
    except CredentialedEvalError as exc:
        if str(exc) not in {
            "model_not_cached",
            "tokenizer_not_cached",
            "source_revision_unverified",
        }:
            raise
        return {"status": "skipped", "reason": str(exc)}
    except ImportError:
        return {"status": "skipped", "reason": "runtime_unavailable"}
    cells = _score(fixtures, suite, year, model_id, runner)["by_label"]
    deltas = {}
    for label, base in reference["by_label"].items():
        other = cells.get(label)
        if base["status"] == "reported" and other and other["status"] == "reported":
            deltas[label] = other["recall"]["rate"] - base["recall"]["rate"]
    return {
        "status": "scored",
        "artifact_id": spec[0],
        "revision_digest": digest,
        "by_label": cells,
        "recall_delta_by_label": deltas,
    }


def run_credentialed(
    *,
    suite: str,
    year: int,
    model_id: str,
    output: str | Path,
    revision: str = "main",
    formats: Sequence[str] = (),
) -> None:
    """Score a credentialed corpus offline and write a new private aggregate file.

    Args:
        suite: Credentialed de-identification corpus family.
        year: Corpus edition.
        model_id: Hub identifier of a locally cached token classifier.
        output: New file under an existing directory outside Git worktrees.
        revision: Cached ref or immutable revision of the classifier.
        formats: Additional cached format names or FORMAT=REPOSITORY@COMMIT entries.
    """
    try:
        target = _output_file(output)
        if suite != "i2b2" or year not in (2006, 2014):
            raise CredentialedEvalError("unsupported_corpus")
        specs = _format_specs(formats, model_id, revision)
        digest, runner = _load_cached_model(model_id, revision)
        try:
            fixtures = load_i2b2_deid(year=year)
        except (OSError, ValueError, RuntimeError):
            raise CredentialedEvalError("corpus_unavailable") from None
        report = _score(fixtures, suite, year, model_id, runner)
        del runner
        report.update(dua=I2B2_DUA_NAME, model_id=model_id, revision_digest=digest)
        if specs:
            report["reference_format"] = "safetensors"
            report["formats"] = {
                name: _format_report(
                    name, spec, model_id, report, fixtures, suite, year
                )
                for name, spec in specs.items()
            }
        encoded = json.dumps(report, sort_keys=True, allow_nan=False).encode() + b"\n"
        with tempfile.NamedTemporaryFile(dir=target.parent) as temp:
            temp.write(encoded)
            temp.flush()
            os.link(temp.name, target)
    except CredentialedEvalError:
        raise
    except FileExistsError:
        raise CredentialedEvalError("output_exists") from None
    except Exception:
        # Third-party model and corpus errors can include input text and paths.
        raise CredentialedEvalError("credentialed_evaluation_failed") from None
