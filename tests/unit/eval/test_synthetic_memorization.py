"""Unit tests for the synthetic memorization audit (issue #2853)."""

from __future__ import annotations

import hashlib
import json
import socket
from typing import Any

import pytest

from openmed.eval.synthetic_memorization import (
    DEFAULT_MAX_FINDINGS,
    MAX_CANDIDATES,
    MAX_REFERENCES,
    MAX_TEXT_CHARS,
    SYNTHETIC_MEMORIZATION_SCHEMA_VERSION,
    MemorizationFinding,
    MemorizationReasonCode,
    MemorizationSignal,
    MemorizationVerdict,
    ProtectedReferenceFingerprint,
    ReferenceFragmentFingerprint,
    SyntheticMemorizationError,
    SyntheticMemorizationPolicy,
    SyntheticMemorizationReport,
    assert_release_allowed,
    audit_synthetic_memorization,
    fingerprint_reference,
    normalize_text,
)

REFERENCE_TEXT = (
    "Patient reports severe chest pain radiating to the left arm since this morning."
)
PARAPHRASE_TEXT = (
    "The patient has severe chest pain that radiates toward the left arm "
    "since this morning."
)
CLEAN_TEXT = "Routine annual review; blood pressure within normal limits."
SENTINEL = "synthetic_private_sentinel_2853"

_GOLDEN_REPORT_DIGEST = (
    "eb18c08cfb91a1986efa17a580ae97c7b7df99695be30a8b8f8926cac8789ea9"
)


def _reference(policy: SyntheticMemorizationPolicy | None = None):
    return fingerprint_reference("protected-001", REFERENCE_TEXT, policy=policy)


def _audit(candidate_text: str, *, policy=None, candidate_id="candidate-001"):
    return audit_synthetic_memorization(
        {candidate_id: candidate_text},
        (_reference(),),
        policy=policy,
    )


def _exact_finding(**overrides: Any) -> MemorizationFinding:
    payload: dict[str, Any] = {
        "candidate_id": "candidate-001",
        "reference_id": "protected-001",
        "signal": MemorizationSignal.EXACT,
        "reason": MemorizationReasonCode.EXACT_FRAGMENT_MATCH,
        "fragment_digest": "sha256:" + "a" * 64,
        "candidate_offset": 0,
        "candidate_length": 44,
        "score": 1.0,
    }
    payload.update(overrides)
    return MemorizationFinding(**payload)


# --- normalization and fingerprinting ----------------------------------------


def test_normalize_text_collapses_whitespace_and_case() -> None:
    assert normalize_text("  Chest   PAIN\n radiating  ") == "chest pain radiating"


def test_normalize_text_rejects_non_string() -> None:
    with pytest.raises(SyntheticMemorizationError):
        normalize_text(None)  # type: ignore[arg-type]


def test_reference_fingerprint_is_value_free() -> None:
    reference = _reference()
    payload = json.dumps(reference.to_dict())
    assert reference.reference_id == "protected-001"
    assert reference.digest.startswith("sha256:")
    assert len(reference.digest) == len("sha256:") + 64
    assert reference.char_length == len(normalize_text(REFERENCE_TEXT))
    assert reference.ngram_size > 0
    assert "chest" not in payload
    assert "patient" not in payload
    assert reference.fragments
    assert [fragment.digest for fragment in reference.fragments] == sorted(
        fragment.digest for fragment in reference.fragments
    )
    assert reference.ngram_digests == tuple(sorted(set(reference.ngram_digests)))


def test_reference_digest_matches_normalized_text() -> None:
    reference = _reference()
    expected = (
        "sha256:"
        + hashlib.sha256(normalize_text(REFERENCE_TEXT).encode("utf-8")).hexdigest()
    )
    assert reference.digest == expected


def test_reference_fingerprint_rejects_short_text() -> None:
    with pytest.raises(SyntheticMemorizationError) as error:
        fingerprint_reference("protected-002", "short note")
    assert "short note" not in str(error.value)


def test_reference_fingerprint_rejects_non_string_text() -> None:
    with pytest.raises(SyntheticMemorizationError):
        fingerprint_reference("protected-002", 5)  # type: ignore[arg-type]


def test_reference_fingerprint_rejects_bad_identifier() -> None:
    with pytest.raises(SyntheticMemorizationError):
        fingerprint_reference("Protected 002", REFERENCE_TEXT)


def test_reference_fingerprint_honours_policy_granularity() -> None:
    coarse = fingerprint_reference(
        "protected-003",
        REFERENCE_TEXT,
        policy=SyntheticMemorizationPolicy(exposure_ngram_size=64),
    )
    fine = fingerprint_reference("protected-007", REFERENCE_TEXT)
    assert coarse.ngram_size == 64
    assert fine.ngram_size != coarse.ngram_size
    assert 0 < len(coarse.ngram_digests) < len(fine.ngram_digests)


def test_fragment_fingerprint_validates_payload() -> None:
    with pytest.raises(SyntheticMemorizationError):
        ReferenceFragmentFingerprint(
            digest="not-a-digest", token_count=3, shingle_hashes=(1,)
        )
    with pytest.raises(SyntheticMemorizationError):
        ReferenceFragmentFingerprint(
            digest="sha256:" + "b" * 64, token_count=0, shingle_hashes=(1,)
        )
    with pytest.raises(SyntheticMemorizationError):
        ReferenceFragmentFingerprint(
            digest="sha256:" + "b" * 64, token_count=3, shingle_hashes=()
        )
    with pytest.raises(SyntheticMemorizationError):
        ReferenceFragmentFingerprint(
            digest="sha256:" + "b" * 64, token_count=3, shingle_hashes=(2, 1)
        )


def test_reference_fingerprint_validates_ordering_and_uniqueness() -> None:
    fragment = ReferenceFragmentFingerprint(
        digest="sha256:" + "c" * 64, token_count=3, shingle_hashes=(7,)
    )
    with pytest.raises(SyntheticMemorizationError):
        ProtectedReferenceFingerprint(
            reference_id="protected-004",
            digest="sha256:" + "d" * 64,
            char_length=40,
            fragments=(fragment, fragment),
            ngram_digests=(),
            ngram_size=8,
        )


# --- signal behaviour --------------------------------------------------------


def test_identical_candidate_reports_all_three_signals() -> None:
    report = _audit(REFERENCE_TEXT)
    assert report.verdict is MemorizationVerdict.BLOCKED
    assert report.blocked is True
    assert report.ok is False
    assert report.signals == (
        MemorizationSignal.EXACT,
        MemorizationSignal.EXPOSURE,
        MemorizationSignal.FUZZY,
    )
    assert report.signal_counts == {"exact": 1, "exposure": 1, "fuzzy": 1}
    exact = [f for f in report.findings if f.signal is MemorizationSignal.EXACT]
    assert exact[0].score == 1.0
    assert exact[0].reference_id == "protected-001"


def test_exact_finding_offsets_index_the_normalized_candidate() -> None:
    report = _audit(REFERENCE_TEXT)
    finding = next(
        finding
        for finding in report.findings
        if finding.signal is MemorizationSignal.EXACT
    )
    normalized = normalize_text(REFERENCE_TEXT)
    matched = normalized[
        finding.candidate_offset : finding.candidate_offset + finding.candidate_length
    ]
    assert matched
    assert (
        finding.fragment_digest
        == "sha256:" + hashlib.sha256(matched.encode("utf-8")).hexdigest()
    )


def test_paraphrase_triggers_exposure_signal_only() -> None:
    report = _audit(PARAPHRASE_TEXT)
    assert report.blocked is True
    assert report.signal_counts["exact"] == 0
    assert report.signal_counts["fuzzy"] == 0
    assert report.signal_counts["exposure"] >= 1


def test_fuzzy_signal_fires_above_the_similarity_floor() -> None:
    policy = SyntheticMemorizationPolicy(fuzzy_similarity=0.2)
    report = _audit(PARAPHRASE_TEXT, policy=policy)
    assert report.signal_counts["fuzzy"] >= 1
    assert report.signal_counts["exact"] == 0


def test_clean_candidate_clears_the_release_gate() -> None:
    report = _audit(CLEAN_TEXT)
    assert report.verdict is MemorizationVerdict.CLEAR
    assert report.findings == ()
    assert report.blocked is False
    assert report.ok is True
    assert report.signal_counts == {"exact": 0, "exposure": 0, "fuzzy": 0}
    assert_release_allowed(report)


def test_empty_candidate_is_skipped_not_blocked() -> None:
    report = _audit("   \n  ")
    assert report.verdict is MemorizationVerdict.CLEAR
    assert report.findings == ()
    assert report.candidate_count == 1
    assert report.skipped_candidate_count == 1


def test_blocking_signals_control_the_verdict() -> None:
    exact_only = SyntheticMemorizationPolicy(
        blocking_signals=(MemorizationSignal.EXACT,)
    )
    assert _audit(PARAPHRASE_TEXT, policy=exact_only).blocked is False
    assert _audit(REFERENCE_TEXT, policy=exact_only).blocked is True

    fuzzy_only = SyntheticMemorizationPolicy(
        blocking_signals=(MemorizationSignal.FUZZY,)
    )
    assert _audit(REFERENCE_TEXT, policy=fuzzy_only).blocked is True


def test_short_reference_ngrams_produce_no_exposure_finding() -> None:
    policy = SyntheticMemorizationPolicy(min_fragment_chars=4, exposure_ngram_size=64)
    short_text = "severe chest pain radiating to left arm"
    reference = fingerprint_reference("protected-005", short_text, policy=policy)
    assert reference.ngram_digests == ()
    report = audit_synthetic_memorization(
        {"candidate-001": short_text}, (reference,), policy=policy
    )
    assert report.signal_counts["exposure"] == 0
    assert report.signal_counts["exact"] == 1


def test_findings_are_deterministically_ordered() -> None:
    first = fingerprint_reference("protected-001", REFERENCE_TEXT)
    second = fingerprint_reference("protected-002", REFERENCE_TEXT)
    report = audit_synthetic_memorization(
        {
            "candidate-002": REFERENCE_TEXT,
            "candidate-001": PARAPHRASE_TEXT,
        },
        (second, first),
    )
    keys = [
        (
            finding.candidate_id,
            finding.candidate_offset,
            finding.signal.value,
            finding.reference_id,
            finding.fragment_digest,
        )
        for finding in report.findings
    ]
    assert keys == sorted(keys)
    repeat = audit_synthetic_memorization(
        {
            "candidate-002": REFERENCE_TEXT,
            "candidate-001": PARAPHRASE_TEXT,
        },
        (second, first),
    )
    assert report.to_json() == repeat.to_json()


def test_candidate_iterables_are_supported() -> None:
    report = audit_synthetic_memorization(
        [("candidate-002", CLEAN_TEXT), ("candidate-001", REFERENCE_TEXT)],
        (_reference(),),
    )
    assert report.candidate_count == 2
    assert report.findings[0].candidate_id == "candidate-001"


def test_max_findings_truncates_but_keeps_blocking_verdict() -> None:
    policy = SyntheticMemorizationPolicy(max_findings=1)
    report = _audit(REFERENCE_TEXT, policy=policy)
    assert report.truncated is True
    assert len(report.findings) == 1
    assert report.blocked is True
    assert DEFAULT_MAX_FINDINGS > 1


def test_no_findings_is_not_truncated() -> None:
    assert _audit(CLEAN_TEXT).truncated is False


# --- value-free guarantees ---------------------------------------------------


def test_report_json_never_contains_raw_text() -> None:
    reference = fingerprint_reference("protected-006", f"{SENTINEL} {REFERENCE_TEXT}")
    report = audit_synthetic_memorization(
        {"candidate-001": f"{SENTINEL} {REFERENCE_TEXT}"}, (reference,)
    )
    payload = report.to_json()
    assert SENTINEL.lower() not in payload.lower()
    assert "chest" not in payload
    assert SENTINEL.lower() not in json.dumps(reference.to_dict()).lower()


def test_release_gate_message_is_value_free() -> None:
    reference = fingerprint_reference("protected-006", f"{SENTINEL} {REFERENCE_TEXT}")
    report = audit_synthetic_memorization(
        {"candidate-001": f"{SENTINEL} {REFERENCE_TEXT}"}, (reference,)
    )
    with pytest.raises(SyntheticMemorizationError) as error:
        assert_release_allowed(report)
    assert SENTINEL.lower() not in str(error.value).lower()
    assert "finding" in str(error.value)


def test_input_errors_do_not_echo_text() -> None:
    with pytest.raises(SyntheticMemorizationError) as error:
        _audit("x" * (MAX_TEXT_CHARS + 1))
    assert "x" * 32 not in str(error.value)


# --- report round trips and invariants ---------------------------------------


def test_report_round_trip() -> None:
    report = _audit(REFERENCE_TEXT)
    assert SyntheticMemorizationReport.from_json(report.to_json()) == report
    assert SyntheticMemorizationReport.from_dict(report.to_dict()) == report
    assert report.to_json().endswith("\n")
    assert report.schema_version == SYNTHETIC_MEMORIZATION_SCHEMA_VERSION


def test_report_json_is_canonical() -> None:
    report = _audit(PARAPHRASE_TEXT)
    decoded = json.loads(report.to_json())
    assert decoded["signal_counts"] == report.signal_counts
    assert decoded["blocking_signals"] == ["exact", "exposure", "fuzzy"]
    assert decoded["findings"][0]["candidate_id"] == "candidate-001"


def test_report_from_dict_rejects_bad_payloads() -> None:
    payload = _audit(REFERENCE_TEXT).to_dict()
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport.from_dict({**payload, "schema_version": "v0"})
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport.from_dict({**payload, "verdict": "maybe"})
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport.from_dict({**payload, "findings": "nope"})
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport.from_dict({**payload, "candidate_count": -1})
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport.from_dict({**payload, "truncated": "no"})
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport.from_dict({**payload, "blocking_signals": ["nope"]})
    broken = json.loads(json.dumps(payload))
    broken["findings"][0]["reason"] = "exact_fragment_match"
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport.from_dict(
            {**broken, "findings": [{**broken["findings"][0], "signal": "fuzzy"}]}
        )


def test_report_from_json_rejects_non_json() -> None:
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport.from_json("{not json")
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport.from_json("[1, 2]")


def test_report_invariants_reject_inconsistent_state() -> None:
    finding = _exact_finding()
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport(
            verdict=MemorizationVerdict.CLEAR,
            findings=(finding,),
            blocking_signals=(MemorizationSignal.EXACT,),
            candidate_count=1,
            reference_count=1,
            skipped_candidate_count=0,
            truncated=False,
        )
    later = _exact_finding(candidate_id="candidate-002")
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport(
            verdict=MemorizationVerdict.BLOCKED,
            findings=(later, finding),
            blocking_signals=(MemorizationSignal.EXACT,),
            candidate_count=2,
            reference_count=1,
            skipped_candidate_count=0,
            truncated=False,
        )
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport(
            verdict=MemorizationVerdict.CLEAR,
            findings=(),
            blocking_signals=(MemorizationSignal.EXACT,),
            candidate_count=1,
            reference_count=1,
            skipped_candidate_count=2,
            truncated=False,
        )
    with pytest.raises(SyntheticMemorizationError):
        SyntheticMemorizationReport(
            verdict=MemorizationVerdict.CLEAR,
            findings=(),
            blocking_signals=(),
            candidate_count=1,
            reference_count=1,
            skipped_candidate_count=0,
            truncated=False,
        )


def test_assert_release_allowed_validates_input() -> None:
    with pytest.raises(SyntheticMemorizationError):
        assert_release_allowed("blocked")  # type: ignore[arg-type]


# --- finding and policy validation -------------------------------------------


def test_finding_validates_reason_matches_signal() -> None:
    with pytest.raises(SyntheticMemorizationError):
        _exact_finding(reason=MemorizationReasonCode.FUZZY_FRAGMENT_MATCH)
    with pytest.raises(SyntheticMemorizationError):
        _exact_finding(signal="exact")
    with pytest.raises(SyntheticMemorizationError):
        _exact_finding(fragment_digest="sha256:short")
    with pytest.raises(SyntheticMemorizationError):
        _exact_finding(candidate_offset=-1)
    with pytest.raises(SyntheticMemorizationError):
        _exact_finding(candidate_length=0)
    with pytest.raises(SyntheticMemorizationError):
        _exact_finding(score=1.5)
    with pytest.raises(SyntheticMemorizationError):
        _exact_finding(candidate_id="Candidate 001")


def test_policy_defaults_and_validation() -> None:
    policy = SyntheticMemorizationPolicy()
    assert policy.blocking_signals == (
        MemorizationSignal.EXACT,
        MemorizationSignal.EXPOSURE,
        MemorizationSignal.FUZZY,
    )
    invalid: list[dict[str, Any]] = [
        {"min_fragment_chars": 0},
        {"min_fragment_chars": True},
        {"fuzzy_similarity": 0.0},
        {"fuzzy_similarity": 1.5},
        {"shingle_size": 0},
        {"shingle_size": 9},
        {"exposure_ngram_size": 0},
        {"exposure_ngram_size": 65},
        {"exposure_match_ratio": 0.0},
        {"exposure_match_ratio": 2.0},
        {"blocking_signals": ()},
        {"blocking_signals": (MemorizationSignal.FUZZY, MemorizationSignal.EXACT)},
        {"blocking_signals": ("exact",)},
        {"max_findings": 0},
        {"max_candidate_fragments": 0},
    ]
    for payload in invalid:
        with pytest.raises(SyntheticMemorizationError):
            SyntheticMemorizationPolicy(**payload)


def test_audit_validates_inputs() -> None:
    reference = _reference()
    with pytest.raises(SyntheticMemorizationError):
        audit_synthetic_memorization({"Candidate 001": CLEAN_TEXT}, (reference,))
    with pytest.raises(SyntheticMemorizationError):
        audit_synthetic_memorization(
            [("candidate-001", CLEAN_TEXT), ("candidate-001", CLEAN_TEXT)],
            (reference,),
        )
    with pytest.raises(SyntheticMemorizationError):
        audit_synthetic_memorization({"candidate-001": 5}, (reference,))  # type: ignore[dict-item]
    with pytest.raises(SyntheticMemorizationError):
        audit_synthetic_memorization(
            {"candidate-001": CLEAN_TEXT}, (reference, reference)
        )
    with pytest.raises(SyntheticMemorizationError):
        audit_synthetic_memorization({"candidate-001": CLEAN_TEXT}, (object(),))
    with pytest.raises(SyntheticMemorizationError):
        audit_synthetic_memorization(
            {"candidate-001": CLEAN_TEXT},
            [None] * (MAX_REFERENCES + 1),  # type: ignore[list-item]
        )
    with pytest.raises(SyntheticMemorizationError):
        audit_synthetic_memorization(
            {f"candidate-{index}": CLEAN_TEXT for index in range(MAX_CANDIDATES + 1)},
            (reference,),
        )
    with pytest.raises(SyntheticMemorizationError):
        audit_synthetic_memorization(
            {"candidate-001": CLEAN_TEXT},
            (reference,),
            policy=object(),  # type: ignore[arg-type]
        )


def test_audit_rejects_candidates_with_too_many_fragments() -> None:
    text = (
        "The first synthetic fragment is comfortably long. "
        "The second synthetic fragment is comfortably long."
    )
    policy = SyntheticMemorizationPolicy(max_candidate_fragments=1)
    with pytest.raises(SyntheticMemorizationError):
        _audit(text, policy=policy)


def test_audit_is_offline_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("memorization audit must not open a socket")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    first = _audit(PARAPHRASE_TEXT)
    second = _audit(PARAPHRASE_TEXT)
    assert first.to_json() == second.to_json()


def test_golden_report_digest() -> None:
    report = _audit(REFERENCE_TEXT)
    digest = hashlib.sha256(report.to_json().encode("utf-8")).hexdigest()
    assert digest == _GOLDEN_REPORT_DIGEST
