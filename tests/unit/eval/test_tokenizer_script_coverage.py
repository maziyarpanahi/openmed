"""Offline tests for PII tokenizer script coverage auditing."""

from __future__ import annotations

from openmed.core.manifest_schema import SCRIPT_COVERAGE_TARGETS
from openmed.eval.coverage import (
    _is_byte_fallback_token,
    audit_pii_tokenizers,
    audit_tokenizer_scripts,
    update_manifest_script_coverage,
)


class StubTokenizer:
    """Return a fixed token mix without optional tokenizer dependencies."""

    unk_token_id = 0
    unk_token = "[UNK]"

    def __init__(self, *, unknown: int = 0, byte_fallback: int = 0) -> None:
        self.unknown = unknown
        self.byte_fallback = byte_fallback

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert text
        assert add_special_tokens is False
        normal = 100 - self.unknown - self.byte_fallback
        return {
            "input_ids": [0] * self.unknown + [2] * self.byte_fallback + [1] * normal
        }

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [
            "[UNK]" if token_id == 0 else "<0xE0>" if token_id == 2 else "token"
            for token_id in token_ids
        ]


def test_claimed_script_at_exact_threshold_remains_supported() -> None:
    results = audit_tokenizer_scripts(StubTokenizer(unknown=1), languages=["hi"])

    assert set(results) == set(SCRIPT_COVERAGE_TARGETS)
    assert results["devanagari"]["unk_rate"] == 0.01
    assert results["devanagari"]["verdict"] == "supported"
    assert results["telugu"]["verdict"] == "unclaimed"


def test_claimed_script_above_threshold_is_unsupported() -> None:
    results = audit_tokenizer_scripts(
        StubTokenizer(unknown=2, byte_fallback=3),
        languages=["te"],
    )

    assert results["telugu"]["unk_rate"] == 0.02
    assert results["telugu"]["byte_fallback_rate"] == 0.03
    assert results["telugu"]["tokens_per_grapheme"] > 0
    assert results["telugu"]["verdict"] == "unsupported"


def test_byte_level_utf8_fragments_count_as_fallback_only_for_byte_backend() -> None:
    assert _is_byte_fallback_token("à¬", byte_level=True) is True
    assert _is_byte_fallback_token("à¬¦", byte_level=True) is False
    assert _is_byte_fallback_token("à¬", byte_level=False) is False


def test_audit_covers_every_pii_row_and_all_scripts() -> None:
    rows = [
        {"repo_id": "OpenMed/pii-hi", "family": "PII", "languages": ["hi"]},
        {"repo_id": "OpenMed/ner", "family": "NER", "languages": ["hi"]},
        {"repo_id": "OpenMed/pii-te", "family": "PII", "languages": ["te"]},
    ]
    loaded: list[str] = []

    def loader(model_id: str) -> StubTokenizer:
        loaded.append(model_id)
        return StubTokenizer()

    report = audit_pii_tokenizers(rows, tokenizer_loader=loader)

    assert report.model_count == 2
    assert report.script_count == 11
    assert loaded == ["OpenMed/pii-hi", "OpenMed/pii-te"]
    assert set(report.models) == {"OpenMed/pii-hi", "OpenMed/pii-te"}
    assert all(
        set(result["scripts"]) == set(SCRIPT_COVERAGE_TARGETS)
        for result in report.models.values()
    )


def test_resume_refreshes_language_claims_and_verdicts_without_reloading() -> None:
    original = audit_pii_tokenizers(
        [{"repo_id": "OpenMed/pii", "family": "PII", "languages": ["en"]}],
        tokenizer_loader=lambda _model_id: StubTokenizer(unknown=2),
    )

    def unexpected_loader(_model_id: str) -> StubTokenizer:
        raise AssertionError("a complete resumed audit must not reload its tokenizer")

    resumed = audit_pii_tokenizers(
        [{"repo_id": "OpenMed/pii", "family": "PII", "languages": ["hi"]}],
        tokenizer_loader=unexpected_loader,
        existing_models=original.models,
    )

    result = resumed.models["OpenMed/pii"]
    assert result["languages"] == ["hi"]
    assert result["scripts"]["devanagari"]["verdict"] == "unsupported"
    assert result["scripts"]["telugu"]["verdict"] == "unclaimed"


def test_report_populates_manifest_and_flags_threshold_in_markdown() -> None:
    rows = [
        {"repo_id": "OpenMed/pii-hi", "family": "PII", "languages": ["hi"]},
        {"repo_id": "OpenMed/ner", "family": "NER", "languages": ["en"]},
    ]
    report = audit_pii_tokenizers(
        rows,
        tokenizer_loader=lambda _model_id: StubTokenizer(unknown=2),
    )

    updated = update_manifest_script_coverage(rows, report)

    assert set(updated[0]["script_coverage"]) == set(SCRIPT_COVERAGE_TARGETS)
    assert updated[0]["script_coverage"]["devanagari"]["verdict"] == "unsupported"
    assert "script_coverage" not in updated[1]
    markdown = report.to_markdown()
    assert "| unsupported | FLAG |" in markdown
    assert "11 script targets" in markdown
