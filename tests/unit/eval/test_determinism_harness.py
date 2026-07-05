"""Determinism regression tests for the public de-identification API (OM-822).

These assert that :func:`openmed.deidentify`, :func:`openmed.extract_pii`, and
:func:`openmed.analyze_text` produce byte-identical spans, labels, and applied
replacements across repeated in-process runs, across fresh interpreter
processes, and against a checked-in PHI-free golden file.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from openmed.core.repro_hash import (
    canonicalize_span_records,
    compute_span_set_hash,
)
from openmed.eval.determinism import (
    DETERMINISM_CORPUS_VERSION,
    CorpusItem,
    DivergenceError,
    SyntheticSpan,
    build_corpus_signature,
    default_corpus,
    run_api_once,
    run_determinism_check,
)

GOLDEN_PATH = Path("openmed/eval/golden/determinism_golden.json")
_APIS = ("extract_pii", "deidentify", "analyze_text")

# Keep the reference corpus well under the 60s CI budget: the harness is the
# only slow part, so we cap end-to-end wall time here.
CI_TIME_BUDGET_SECONDS = 60.0


def _load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_corpus_is_synthetic_and_spans_match_text() -> None:
    corpus = default_corpus()

    assert corpus, "determinism corpus must not be empty"
    ids = [item.item_id for item in corpus]
    assert len(ids) == len(set(ids)), "corpus item ids must be unique"

    for item in corpus:
        # validate() raises if any declared span does not match the text.
        item.validate()
        for span in item.spans:
            assert item.text[span.start : span.end] == span.word


def test_run_to_run_identity_in_process() -> None:
    """Each API run twice over each item must produce identical span sets."""
    for item in default_corpus():
        for api_name in _APIS:
            first = run_api_once(item, api_name)
            second = run_api_once(item, api_name)
            assert first["hash"] == second["hash"], (
                f"{api_name} diverged run-to-run for {item.item_id}"
            )
            assert first["spans"] == second["spans"]


def test_determinism_check_passes_and_matches_golden() -> None:
    mapping = run_determinism_check(iterations=5)

    assert mapping["schema_version"] == DETERMINISM_CORPUS_VERSION
    assert mapping["corpus_signature"] == build_corpus_signature(default_corpus())
    assert mapping == _load_golden(), (
        "determinism golden hashes drifted; regenerate the golden file only if "
        "the pinned corpus or deterministic pipeline output changed on purpose"
    )


def test_golden_file_contains_no_raw_phi() -> None:
    """The golden file must carry offsets/labels/hashes only -- never raw text."""
    raw = GOLDEN_PATH.read_text(encoding="utf-8")

    # No synthetic identifier from the corpus may appear in the golden file.
    for item in default_corpus():
        for span in item.spans:
            assert span.word not in raw
        # The document text itself must not be embedded either.
        assert item.text not in raw

    payload = json.loads(raw)
    # Every leaf value under items is a sha256 content hash, nothing else.
    for api_hashes in payload["items"].values():
        for value in api_hashes.values():
            assert value.startswith("sha256:")


def test_repro_hash_stable_across_100_iterations() -> None:
    """Same input + seed must yield the identical span-set hash 100 times."""
    item = next(
        candidate
        for candidate in default_corpus()
        if candidate.item_id == "replace_seeded_name"
    )

    hashes = {run_api_once(item, "deidentify")["hash"] for _ in range(100)}
    assert len(hashes) == 1, "seeded replacement hash drifted across iterations"


def test_seeded_replacement_and_keyed_shift_are_deterministic() -> None:
    """Stochastic methods are pinned by seed / patient-key + secret."""
    corpus = {item.item_id: item for item in default_corpus()}

    replace_item = corpus["replace_seeded_name"]
    replace_hashes = {
        run_api_once(replace_item, "deidentify")["hash"] for _ in range(8)
    }
    assert len(replace_hashes) == 1

    shift_item = corpus["shift_dates_patient_keyed"]
    shift_hashes = {run_api_once(shift_item, "deidentify")["hash"] for _ in range(8)}
    assert len(shift_hashes) == 1


def test_cross_process_determinism_fresh_interpreter() -> None:
    """A fresh interpreter must emit the same hashes as the in-process run.

    Running in a separate process with a distinct PYTHONHASHSEED catches
    dict/set ordering and hash-seed leaks that an in-process rerun could miss.
    """
    in_process = run_determinism_check(iterations=2)

    completed = subprocess.run(
        [sys.executable, "-m", "openmed.eval.determinism", "--iterations", "2"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=120,
        env={**_child_env(), "PYTHONHASHSEED": "0"},
    )
    assert completed.returncode == 0, completed.stderr
    first_child = json.loads(completed.stdout)

    completed_two = subprocess.run(
        [sys.executable, "-m", "openmed.eval.determinism", "--iterations", "2"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=120,
        env={**_child_env(), "PYTHONHASHSEED": "1"},
    )
    assert completed_two.returncode == 0, completed_two.stderr
    second_child = json.loads(completed_two.stdout)

    assert first_child == in_process
    assert second_child == in_process


def test_unseeded_random_call_is_caught_by_harness() -> None:
    """A deliberately nondeterministic runner must trip the divergence guard."""
    import random

    def flaky_runner(item: CorpusItem, api_name: str) -> dict:
        result = run_api_once(item, api_name)
        # Inject an unseeded, run-to-run varying span to simulate a regression
        # where an ordering-dependent or unseeded-random step leaks into output.
        result["spans"] = list(result["spans"]) + [
            {
                "start": random.randint(0, 1_000_000),
                "end": 1_000_001,
                "label": "NAME",
                "confidence": 0.5,
                "action": "replace",
                "replacement_digest": f"{random.random()}",
            }
        ]
        result["hash"] = compute_span_set_hash(
            [
                {
                    "start": record["start"],
                    "end": record["end"],
                    "label": record["label"],
                    "confidence": record["confidence"],
                    "action": record["action"],
                    "surrogate": record["replacement_digest"],
                }
                for record in result["spans"]
            ],
            method="flaky",
        )
        return result

    with pytest.raises(DivergenceError):
        run_determinism_check(iterations=3, runner=flaky_runner)


def test_divergence_diagnostic_reports_offsets_only_no_raw_text() -> None:
    """The failure message must not contain any synthetic identifier text."""
    forbidden = [span.word for item in default_corpus() for span in item.spans]
    forbidden += [item.text for item in default_corpus()]

    call_count = {"n": 0}

    def drifting_runner(item: CorpusItem, api_name: str) -> dict:
        # Return a stable baseline then a divergent second reading.
        call_count["n"] += 1
        base = run_api_once(item, api_name)
        if call_count["n"] % 2 == 0:
            base = dict(base)
            base["hash"] = base["hash"][:-1] + ("0" if base["hash"][-1] != "0" else "1")
        return base

    with pytest.raises(DivergenceError) as excinfo:
        run_determinism_check(iterations=2, runner=drifting_runner)

    message = str(excinfo.value)
    assert "offsets/labels/hashes only" in message
    for secret in forbidden:
        assert secret not in message


def test_harness_runs_within_ci_time_budget() -> None:
    started = time.perf_counter()
    run_determinism_check(iterations=5)
    elapsed = time.perf_counter() - started
    assert elapsed < CI_TIME_BUDGET_SECONDS, (
        f"determinism harness took {elapsed:.2f}s, over the {CI_TIME_BUDGET_SECONDS}s "
        "CI budget"
    )


def test_span_set_hash_is_order_independent_and_phi_free() -> None:
    """The span-set hash must ignore input ordering and never embed raw text."""
    spans_forward = [
        SyntheticSpan("Ada Stone", "NAME", 7, 16, 0.9),
        SyntheticSpan("Casey Example", "NAME", 30, 43, 0.8),
    ]
    spans_reversed = list(reversed(spans_forward))

    forward_records = canonicalize_span_records(
        {
            "start": span.start,
            "end": span.end,
            "label": span.label,
            "confidence": span.confidence,
        }
        for span in spans_forward
    )
    reversed_records = canonicalize_span_records(
        {
            "start": span.start,
            "end": span.end,
            "label": span.label,
            "confidence": span.confidence,
        }
        for span in spans_reversed
    )
    assert forward_records == reversed_records

    forward_hash = compute_span_set_hash(forward_records, method="test")
    reversed_hash = compute_span_set_hash(reversed_records, method="test")
    assert forward_hash == reversed_hash

    # No raw entity text is present in the canonical records.
    serialized = json.dumps(forward_records)
    assert "Ada Stone" not in serialized
    assert "Casey Example" not in serialized


def _child_env() -> dict:
    """Return an environment for the subprocess that finds the worktree package."""
    import os

    env = dict(os.environ)
    cwd = str(Path.cwd())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{cwd}{os.pathsep}{existing}" if existing else cwd
    return env
