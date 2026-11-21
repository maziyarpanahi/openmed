"""Offline aggregate-only evaluation of user-supplied de-identification data."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from openmed.eval.aggregate_report import project_aggregate_report
from openmed.eval.datasets.cegs_ngrid import CEGS_NGRID_DUA, load_cegs_ngrid
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


def _load_cached_model(model_id: str, revision: str) -> tuple[str, ModelRunner]:
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


def run_credentialed(
    *,
    suite: str,
    year: int | None,
    model_id: str,
    output: str | Path,
    revision: str = "main",
) -> None:
    """Score a credentialed corpus offline and write a new private aggregate file.

    Args:
        suite: Credentialed de-identification corpus family.
        year: Corpus edition. The CEGS suite name selects 2016 by default.
        model_id: Hub identifier of a locally cached token classifier.
        output: New file under an existing directory outside Git worktrees.
        revision: Cached ref or immutable revision of the classifier.
    """
    try:
        target = _output_file(output)
        if suite == "cegs-ngrid-2016" and year in (None, 2016):
            corpus, dua, year = "cegs-n-grid", CEGS_NGRID_DUA, 2016
        elif suite == "i2b2" and year in (2006, 2014):
            corpus, dua = "i2b2", I2B2_DUA_NAME
        else:
            raise CredentialedEvalError("unsupported_corpus")
        digest, runner = _load_cached_model(model_id, revision)
        try:
            fixtures = (
                load_cegs_ngrid()
                if suite == "cegs-ngrid-2016"
                else load_i2b2_deid(year=year)
            )
        except (OSError, ValueError, RuntimeError):
            raise CredentialedEvalError("corpus_unavailable") from None
        benchmark = run_benchmark(
            fixtures, suite=suite, model_name=model_id, runner=runner, cache_dir=None
        )
        report = project_aggregate_report(benchmark, fixtures, corpus=corpus, year=year)
        report.update(
            dua=dua,
            model_id=model_id,
            revision_digest=digest,
            postprocessing_profile="transformers-simple-v1",
        )
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
