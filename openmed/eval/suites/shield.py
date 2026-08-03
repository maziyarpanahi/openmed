"""SHIELD clinical PHI comparison corpus loader.

The suite is intentionally loaded by reference from the public dataset mirror.
No corpus rows are stored in this repository.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import quote
from urllib.request import urlopen

from openmed.core.audit import stable_hash
from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    DATE,
    ID_NUM,
    LOCATION,
    ORGANIZATION,
    PERSON,
    PHONE,
    URL,
)
from openmed.eval.cache import eval_code_hash, hash_fixture_set
from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    run_benchmark,
)
from openmed.eval.metrics import EvalSpan
from openmed.eval.report import BenchmarkReport

SHIELD = "shield"
CORPUS_ROLE = "comparison"
SUITE_ANNOTATION = "comparison corpus, not a high-recall gate target"
IS_HIGH_RECALL_GATE_TARGET = False

PUBLIC_SAMPLE_REPOSITORY = "tds-research-tech/shield-sample"
FULL_REPOSITORY = "tds-research-tech/shield"
PUBLIC_SAMPLE_NOTES_CONFIG = "sample_notes"
PUBLIC_SAMPLE_SPANS_CONFIG = "sample_spans"
FULL_NOTES_CONFIG = "full_notes"
FULL_SPANS_CONFIG = "full_spans"
DEFAULT_SPLIT = "train"
VERIFIED_LICENSE = "data-use-agreement"
VERIFIED_LICENSE_DATE = "2026-06-12"

SHIELD_LABEL_TO_CANONICAL: dict[str, str] = {
    "age": AGE,
    "date": DATE,
    "doctor": PERSON,
    "hospital": ORGANIZATION,
    "id": ID_NUM,
    "location": LOCATION,
    "patient": PERSON,
    "phone": PHONE,
    "web": URL,
}

RowsLoader = Callable[[str, str, str], list[Mapping[str, Any]]]
CheckpointManifest = Mapping[str, Any] | str | Path
_SAFE_MANIFEST_REF = re.compile(r"[A-Za-z0-9._~:/?#@!$&()*+,;=%-]{1,2048}")
_SAFE_SOURCE_REVISION = re.compile(r"[A-Za-z0-9._~:/@+-]{1,256}")


@dataclass(frozen=True)
class ShieldSource:
    """Dataset mirror coordinates for one SHIELD variant."""

    repository: str
    notes_config: str
    spans_config: str
    split: str
    variant: str
    requires_approval: bool


PUBLIC_SAMPLE_SOURCE = ShieldSource(
    repository=PUBLIC_SAMPLE_REPOSITORY,
    notes_config=PUBLIC_SAMPLE_NOTES_CONFIG,
    spans_config=PUBLIC_SAMPLE_SPANS_CONFIG,
    split=DEFAULT_SPLIT,
    variant="public_sample",
    requires_approval=False,
)

FULL_SOURCE = ShieldSource(
    repository=FULL_REPOSITORY,
    notes_config=FULL_NOTES_CONFIG,
    spans_config=FULL_SPANS_CONFIG,
    split=DEFAULT_SPLIT,
    variant="full_access_controlled",
    requires_approval=True,
)


def map_shield_label(label: str) -> str:
    """Map a SHIELD PHI category onto OpenMed's canonical label taxonomy."""
    canonical = SHIELD_LABEL_TO_CANONICAL.get(label.strip().lower())
    if canonical is None:
        allowed = ", ".join(sorted(SHIELD_LABEL_TO_CANONICAL))
        raise ValueError(f"unknown SHIELD label {label!r}; expected one of: {allowed}")
    return canonical


def shield_suite_metadata(*, use_sample: bool = True) -> dict[str, Any]:
    """Return source, license, and role metadata for SHIELD benchmark reports."""
    source = _source_for(use_sample=use_sample)
    return {
        "access": (
            "public sample is available without approval; full corpus requires "
            "approved access and a signed data-use agreement"
        ),
        "annotation": SUITE_ANNOTATION,
        "corpus_role": CORPUS_ROLE,
        "full_repository": FULL_REPOSITORY,
        "gate_target": IS_HIGH_RECALL_GATE_TARGET,
        "label_mapping": dict(sorted(SHIELD_LABEL_TO_CANONICAL.items())),
        "license": VERIFIED_LICENSE,
        "license_verified_at": VERIFIED_LICENSE_DATE,
        "notes_config": source.notes_config,
        "redistribution": "not vendored; loaded by reference",
        "repository": source.repository,
        "requires_approval": source.requires_approval,
        "source_url": f"https://huggingface.co/datasets/{source.repository}",
        "span_count_paper": 10505,
        "spans_config": source.spans_config,
        "split": source.split,
        "suite": SHIELD,
        "variant": source.variant,
    }


def load_shield_fixtures(
    *,
    use_sample: bool = True,
    rows_loader: RowsLoader | None = None,
) -> list[BenchmarkFixture]:
    """Load SHIELD notes and spans as benchmark fixtures.

    The default uses the public sample mirror. Set ``use_sample=False`` only
    on an approved machine with access to the full corpus.
    """
    source = _source_for(use_sample=use_sample)
    loader = rows_loader or _load_dataset_rows
    notes = loader(source.repository, source.notes_config, source.split)
    spans = loader(source.repository, source.spans_config, source.split)
    return fixtures_from_rows(notes, spans, source=source)


def fixtures_from_rows(
    notes: Iterable[Mapping[str, Any]],
    spans: Iterable[Mapping[str, Any]],
    *,
    source: ShieldSource = PUBLIC_SAMPLE_SOURCE,
) -> list[BenchmarkFixture]:
    """Build benchmark fixtures from SHIELD note and span table rows."""
    spans_by_note: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for span in spans:
        note_id = str(span.get("note_id", ""))
        if note_id:
            spans_by_note[note_id].append(span)

    metadata = shield_suite_metadata(use_sample=source.variant == "public_sample")
    fixtures: list[BenchmarkFixture] = []
    for note in notes:
        note_id = str(note.get("note_id", ""))
        text = str(note.get("note_text", ""))
        note_type = str(note.get("note_type") or "")
        gold_spans = tuple(
            _span_from_row(span, text=text)
            for span in sorted(
                spans_by_note.get(note_id, []),
                key=lambda row: (
                    int(row.get("span_start", 0)),
                    str(row.get("span_id", "")),
                ),
            )
        )
        fixture_metadata = dict(metadata)
        fixture_metadata.update(
            {
                "note_type": note_type,
                "source_note_id": note_id,
            }
        )
        fixtures.append(
            BenchmarkFixture(
                fixture_id=note_id,
                text=text,
                gold_spans=gold_spans,
                language="en",
                metadata=fixture_metadata,
            )
        )
    return fixtures


def run_clinical_phi_shield_benchmark(
    fixtures: Sequence[BenchmarkFixture],
    *,
    checkpoint_manifest: CheckpointManifest,
    checkpoint_manifest_ref: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Benchmark the named clinical-PHI flagship as SHIELD comparison evidence.

    The checkpoint manifest may be one JSON object, a list of model-manifest
    rows, or a JSONL model manifest. ``checkpoint_manifest_ref`` is the stable
    repository or publication link recorded in the report. Corpus text and raw
    fixture identifiers are never copied into the resulting metadata.

    Args:
        fixtures: Public-sample SHIELD fixtures loaded by reference. Tests may
            supply synthetic SHIELD-shaped rows through :func:`fixtures_from_rows`.
        checkpoint_manifest: Checkpoint metadata or a path containing it.
        checkpoint_manifest_ref: Stable link to the committed or published
            checkpoint manifest.
        device: Device label recorded in the benchmark report.
        runner: Optional model runner; defaults to the standard eval runner.
        generated_at: Optional caller-supplied report timestamp.

    Returns:
        A ``BenchmarkReport`` with explicit aggregate and per-label SHIELD
        comparison metrics plus reproducibility metadata.

    Raises:
        ValueError: If the checkpoint or fixtures do not identify the named
            flagship and public SHIELD comparison source.
    """

    from openmed.eval.datasets.clinical_phi import (
        CLINICAL_PHI_MANIFEST_ID,
        CLINICAL_PHI_MANIFEST_REF,
        CLINICAL_PRIVACY_MODEL_ID,
        clinical_phi_manifest_hash,
        load_clinical_phi_manifest,
    )

    if not fixtures:
        raise ValueError("clinical PHI SHIELD benchmark requires fixtures")
    _validate_public_sample_fixtures(fixtures)
    manifest_ref = _validate_manifest_ref(checkpoint_manifest_ref)
    checkpoint = _checkpoint_manifest_row(
        checkpoint_manifest,
        model_id=CLINICAL_PRIVACY_MODEL_ID,
    )
    checkpoint_metadata = _checkpoint_metadata(
        checkpoint,
        manifest_ref=manifest_ref,
        model_id=CLINICAL_PRIVACY_MODEL_ID,
    )

    dataset_manifest = load_clinical_phi_manifest()
    public_source = dataset_manifest.source("shield_public_sample")
    metadata = shield_suite_metadata()
    metadata.update(
        {
            "benchmark_domain": "clinical_phi",
            "checkpoint_manifest": checkpoint_metadata,
            "comparison_evidence_only": True,
            "dataset_manifest": {
                "manifest_hash": clinical_phi_manifest_hash(dataset_manifest),
                "manifest_id": CLINICAL_PHI_MANIFEST_ID,
                "manifest_ref": CLINICAL_PHI_MANIFEST_REF,
            },
            "fixture_ids": [
                _sha256_value(stable_hash({"fixture_id": fixture.fixture_id}))
                for fixture in fixtures
            ],
            "public_corpus_reference": {
                "dataset": public_source.dataset,
                "license_id": public_source.license_id,
                "loader_ref": public_source.loader_ref,
                "redistribution": public_source.redistribution,
                "source_id": public_source.source_id,
                "source_url": public_source.source_url,
                "split": public_source.split,
            },
            "reproducibility": {
                "eval_code_hash": _sha256_value(
                    eval_code_hash(
                        (
                            "openmed.eval.harness",
                            "openmed.eval.metrics",
                            "openmed.eval.suites.shield",
                        )
                    )
                ),
                "fixture_set_hash": _sha256_value(hash_fixture_set(fixtures)),
                "runner_ref": ("openmed.eval.suites:run_clinical_phi_shield_benchmark"),
            },
        }
    )

    report = run_benchmark(
        fixtures,
        suite=SHIELD,
        model_name=CLINICAL_PRIVACY_MODEL_ID,
        device=device,
        runner=runner,
        generated_at=generated_at,
        metadata=metadata,
    )
    metrics = dict(report.metrics)
    metrics["shield_comparison"] = _shield_comparison_metrics(metrics)
    return replace(report, metrics=metrics)


def _span_from_row(row: Mapping[str, Any], *, text: str) -> EvalSpan:
    raw_label = str(row.get("span_label", ""))
    canonical_label = map_shield_label(raw_label)
    start = _read_required_int(row, "span_start")
    end = _read_required_int(row, "span_end")
    if start < 0 or end < start or end > len(text):
        raise ValueError(
            f"invalid SHIELD span offsets {start}:{end} for text length {len(text)}"
        )
    return EvalSpan(
        start=start,
        end=end,
        label=canonical_label,
        text=text[start:end],
        language="en",
        metadata={
            "canonical_label": canonical_label,
            "shield_label": raw_label.strip().lower(),
            "span_id": str(row.get("span_id") or ""),
        },
    )


def _checkpoint_manifest_row(
    checkpoint_manifest: CheckpointManifest,
    *,
    model_id: str,
) -> Mapping[str, Any]:
    if isinstance(checkpoint_manifest, Mapping):
        payload: Any = checkpoint_manifest
    else:
        path = Path(checkpoint_manifest)
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"failed to read checkpoint manifest: {path}") from exc
        try:
            payload = json.loads(contents)
        except json.JSONDecodeError:
            try:
                payload = [
                    json.loads(line) for line in contents.splitlines() if line.strip()
                ]
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"checkpoint manifest is not valid JSON or JSONL: {path}"
                ) from exc

    if isinstance(payload, Mapping):
        nested = payload.get("models") or payload.get("checkpoints")
        rows: Sequence[Any] = (
            nested
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes))
            else (payload,)
        )
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        rows = payload
    else:
        raise ValueError("checkpoint manifest must contain model metadata")

    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("model_id") or row.get("repo_id") or "") == model_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "checkpoint manifest must contain exactly one "
            f"{model_id!r} row; found {len(matches)}"
        )
    return matches[0]


def _checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    manifest_ref: str,
    model_id: str,
) -> dict[str, str]:
    metadata = {
        "manifest_content_hash": _sha256_value(stable_hash(checkpoint)),
        "manifest_ref": manifest_ref,
        "model_id": model_id,
    }
    reproducibility_hash = checkpoint.get("reproducibility_hash")
    if reproducibility_hash is not None:
        value = str(reproducibility_hash)
        if not _is_sha256(value):
            raise ValueError(
                "checkpoint reproducibility_hash must be sha256:<64 lowercase hex>"
            )
        metadata["reproducibility_hash"] = value

    provenance = checkpoint.get("provenance")
    source_revision = checkpoint.get("source_revision")
    if source_revision is None and isinstance(provenance, Mapping):
        source_revision = provenance.get("source_revision")
    if source_revision is not None:
        revision = str(source_revision)
        if _SAFE_SOURCE_REVISION.fullmatch(revision) is None:
            raise ValueError("checkpoint source_revision must be a safe revision id")
        metadata["source_revision"] = revision
    return metadata


def _shield_comparison_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    recall = metrics.get("recall_slices")
    leakage = metrics.get("leakage")
    exact = metrics.get("exact_span_f1")
    if not all(isinstance(value, Mapping) for value in (recall, leakage, exact)):
        raise ValueError(
            "SHIELD benchmark lacks required recall, leakage, or F1 metrics"
        )

    labels = sorted(set(SHIELD_LABEL_TO_CANONICAL.values()))
    recall_by_label = recall.get("by_label")
    leakage_by_label = leakage.get("by_label")
    if not isinstance(recall_by_label, Mapping) or not isinstance(
        leakage_by_label, Mapping
    ):
        raise ValueError("SHIELD benchmark lacks per-label recall or leakage")

    return {
        "aggregate": {
            "exact_span_f1": float(exact["f1"]),
            "exact_span_precision": float(exact["precision"]),
            "exact_span_recall": float(exact["recall"]),
            "leakage": float(leakage["overall"]),
            "recall": float(recall["overall"]),
        },
        "by_label": {
            label: {
                "leakage": float(leakage_by_label[label]),
                "recall": float(recall_by_label[label]),
            }
            for label in labels
        },
        "evidence_role": "comparison",
        "high_recall_release_gate": False,
    }


def _validate_public_sample_fixtures(fixtures: Sequence[BenchmarkFixture]) -> None:
    for fixture in fixtures:
        repository = str(fixture.metadata.get("repository") or "")
        variant = str(fixture.metadata.get("variant") or "")
        if repository != PUBLIC_SAMPLE_REPOSITORY or variant != "public_sample":
            raise ValueError(
                "clinical PHI SHIELD benchmark requires public-sample fixtures"
            )


def _validate_manifest_ref(value: str) -> str:
    reference = str(value).strip()
    if _SAFE_MANIFEST_REF.fullmatch(reference) is None:
        raise ValueError("checkpoint_manifest_ref must be a safe repository link")
    return reference


def _is_sha256(value: str) -> bool:
    prefix = "sha256:"
    digest = value.removeprefix(prefix)
    return (
        value.startswith(prefix)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _sha256_value(value: str) -> str:
    return value if value.startswith("sha256:") else f"sha256:{value}"


def _load_dataset_rows(
    repository: str, config: str, split: str
) -> list[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return _load_dataset_rows_via_server(repository, config, split)

    try:
        dataset = load_dataset(repository, config, split=split)
    except Exception as exc:
        if repository == PUBLIC_SAMPLE_REPOSITORY:
            return _load_dataset_rows_via_server(repository, config, split)
        raise RuntimeError(
            f"failed to load approved SHIELD rows for {repository}/{config}/{split}: {exc}"
        ) from exc
    return [dict(row) for row in dataset]


def _load_dataset_rows_via_server(
    repository: str,
    config: str,
    split: str,
    *,
    page_size: int = 100,
) -> list[Mapping[str, Any]]:
    encoded_repository = quote(repository, safe="")
    encoded_config = quote(config, safe="")
    encoded_split = quote(split, safe="")
    rows: list[Mapping[str, Any]] = []
    offset = 0

    while True:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={encoded_repository}"
            f"&config={encoded_config}"
            f"&split={encoded_split}"
            f"&offset={offset}"
            f"&length={page_size}"
        )
        try:
            with urlopen(url, timeout=30) as response:  # nosec: trusted fixed host
                payload = json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            raise RuntimeError(
                f"failed to load SHIELD rows for {repository}/{config}/{split}: {exc}"
            ) from exc

        page_rows = [item["row"] for item in payload.get("rows", [])]
        rows.extend(page_rows)
        total = int(payload.get("num_rows_total") or len(rows))
        if not page_rows or len(rows) >= total:
            return rows
        offset += len(page_rows)


def _read_required_int(row: Mapping[str, Any], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"SHIELD span row missing integer {key!r}: {row!r}") from None


def _source_for(*, use_sample: bool) -> ShieldSource:
    return PUBLIC_SAMPLE_SOURCE if use_sample else FULL_SOURCE


_invalid_mapping = {
    label: canonical
    for label, canonical in SHIELD_LABEL_TO_CANONICAL.items()
    if canonical not in CANONICAL_LABELS
}
if _invalid_mapping:
    raise RuntimeError(
        f"SHIELD mapping contains non-canonical labels: {_invalid_mapping}"
    )


__all__ = [
    "SHIELD",
    "CORPUS_ROLE",
    "SUITE_ANNOTATION",
    "IS_HIGH_RECALL_GATE_TARGET",
    "PUBLIC_SAMPLE_REPOSITORY",
    "FULL_REPOSITORY",
    "VERIFIED_LICENSE",
    "SHIELD_LABEL_TO_CANONICAL",
    "ShieldSource",
    "PUBLIC_SAMPLE_SOURCE",
    "FULL_SOURCE",
    "map_shield_label",
    "shield_suite_metadata",
    "load_shield_fixtures",
    "fixtures_from_rows",
    "run_clinical_phi_shield_benchmark",
]
