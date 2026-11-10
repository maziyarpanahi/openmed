#!/usr/bin/env python3
"""Measure a small, rights-clean synthetic SHIELD-schema rules baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.labels import AGE, DATE, ID_NUM, PHONE, URL
from openmed.core.repro_hash import compute_reproducibility_hash
from openmed.eval.cache import hash_fixture_set
from openmed.eval.harness import BenchmarkFixture, run_benchmark
from openmed.eval.metrics import EvalSpan
from openmed.eval.report import BenchmarkReport
from openmed.eval.suites.shield import map_shield_label

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "openmed/eval/fixtures/shield_synthetic_baseline.json"
DEFAULT_REPORT = ROOT / "docs/benchmarks/shield-synthetic.report.json"
MODEL_ID = "openmed-synthetic-regex-phi-baseline-v1"
SUITE = "shield-synthetic"
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (DATE, re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")),
    (ID_NUM, re.compile(r"\bSYN-\d{4}\b")),
    (PHONE, re.compile(r"\b555-\d{4}\b")),
    (URL, re.compile(r"\b[a-z0-9.-]+\.invalid\b")),
    (AGE, re.compile(r"\b\d{1,3}\b(?= years old\b)")),
)


def text_file_digest(path: Path) -> str:
    """Hash UTF-8 source text with checkout newlines normalized to LF.

    Git may materialize CRLF on Windows. Only newline encoding is normalized;
    all source content, whitespace, and the final newline remain bound.
    """
    content = path.read_text(encoding="utf-8").encode("utf-8")
    return "sha256:" + hashlib.sha256(content).hexdigest()


def load_synthetic_fixtures(path: Path = FIXTURE_PATH) -> list[BenchmarkFixture]:
    """Construct gold spans from committed synthetic text segments."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "openmed.shield_synthetic_baseline.v1":
        raise ValueError("unsupported synthetic SHIELD fixture schema")
    if (
        payload.get("source_rights")
        != "OpenMed-generated synthetic fixture; Apache-2.0"
    ):
        raise ValueError("synthetic SHIELD fixture has no approved source rights")

    fixtures: list[BenchmarkFixture] = []
    for case in payload["cases"]:
        text = ""
        gold: list[EvalSpan] = []
        for segment in case["segments"]:
            value = str(segment["text"])
            start = len(text)
            text += value
            label = segment.get("shield_label")
            if label is not None:
                gold.append(
                    EvalSpan(
                        start=start,
                        end=len(text),
                        label=map_shield_label(str(label)),
                        text=value,
                        language="en",
                    )
                )
        fixtures.append(
            BenchmarkFixture(
                fixture_id=str(case["id"]),
                text=text,
                gold_spans=tuple(gold),
                language="en",
                metadata={"synthetic": True},
            )
        )
    if not fixtures or any(not fixture.gold_spans for fixture in fixtures):
        raise ValueError("synthetic SHIELD baseline requires annotated fixtures")
    return fixtures


def regex_baseline(
    fixture: BenchmarkFixture, model_name: str, device: str
) -> list[Mapping[str, Any]]:
    """Return fixed rule predictions without loading a model or remote data."""
    if model_name != MODEL_ID or device != "cpu":
        raise ValueError("synthetic baseline requires its pinned CPU configuration")
    return [
        {"start": match.start(), "end": match.end(), "label": label}
        for label, pattern in RULES
        for match in pattern.finditer(fixture.text)
    ]


def generate_report(
    *,
    source_revision: str,
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Run the baseline and bind the exact data, rules, and source revision."""
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("source_revision must be a full 40-character Git SHA")
    fixtures = load_synthetic_fixtures()
    fixture_digest = text_file_digest(FIXTURE_PATH)
    script_digest = text_file_digest(Path(__file__))
    recipe = {
        "script_sha256": script_digest,
        "rules_revision": "v1",
        "suite": SUITE,
        "harness": "openmed.eval.harness.run_benchmark",
        "text_digest_format": "utf8-lf-v1",
    }
    data_manifest = {
        "fixture_sha256": fixture_digest,
        "fixture_set_hash": hash_fixture_set(fixtures),
        "fixture_count": len(fixtures),
    }
    base_model = {"id": MODEL_ID, "revision": "v1", "kind": "fixed regex rules"}
    reproduction = compute_reproducibility_hash(
        recipe=recipe,
        data_manifest=data_manifest,
        base_model=base_model,
        git_sha=source_revision,
    )
    metadata = {
        "synthetic": True,
        "model_family": "Synthetic rules control",
        "release_tag": "not-released",
        "publication_role": "synthetic_shield_baseline",
        "source_rights": "OpenMed-generated synthetic fixture; Apache-2.0",
        "fixture_provenance": "openmed/eval/fixtures/shield_synthetic_baseline.json",
        "fixture_sha256": fixture_digest,
        "fixture_set_hash": data_manifest["fixture_set_hash"],
        "model_revision": "v1",
        "config_revision": "v1",
        "script_sha256": script_digest,
        "text_digest_format": "utf8-lf-v1",
        "source_revision": source_revision,
        "reproducibility_hash": reproduction,
        "evidence_path": "shield-synthetic.report.json",
        "limitations": (
            "Two OpenMed-generated synthetic notes using SHIELD label names; "
            "no SHIELD public-sample or restricted records were used. "
            "This is a rules smoke baseline, not clinical model performance "
            "or a high-recall release gate."
        ),
    }
    return run_benchmark(
        fixtures,
        suite=SUITE,
        model_name=MODEL_ID,
        device="cpu",
        runner=regex_baseline,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        metadata=metadata,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write inspectable JSON evidence for the synthetic baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    report = generate_report(source_revision=args.source_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
