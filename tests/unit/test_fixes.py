"""Tests for the specific fixes applied to OpenMed."""

import json
import threading
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from openmed.core.pii import (
    DeidentificationResult,
    PIIEntity,
    reidentify,
)
from openmed.core.config import (
    OpenMedConfig,
    get_config,
    set_config,
    _load_toml,
    _load_toml_fallback,
)


# ---------------------------------------------------------------------------
# Fix 1 (CRITICAL): reidentify() position-based replacement
# ---------------------------------------------------------------------------


class TestReidentifyPositionBased:
    """Verify that reidentify() handles overlapping/duplicate placeholders."""

    def test_duplicate_placeholders_restored_correctly(self):
        """Two different names both redacted to [NAME] must round-trip."""
        original = "Patient John Doe met with Jane Smith on Monday."
        mapping = {
            "[NAME_1]": "John Doe",
            "[NAME_2]": "Jane Smith",
        }
        deidentified = "Patient [NAME_1] met with [NAME_2] on Monday."
        result = reidentify(deidentified, mapping)
        assert result == original

    def test_multiple_occurrences_of_same_placeholder(self):
        """Same redacted value appearing multiple times."""
        mapping = {"[X]": "secret"}
        deidentified = "Start [X] middle [X] end"
        result = reidentify(deidentified, mapping)
        assert result == "Start secret middle secret end"

    def test_overlapping_placeholders_handled(self):
        """Placeholders at different positions restored independently."""
        mapping = {
            "[A]": "foo",
            "[B]": "bar",
        }
        deidentified = "[A] and [B]"
        result = reidentify(deidentified, mapping)
        assert result == "foo and bar"

    def test_empty_mapping(self):
        assert reidentify("no changes", {}) == "no changes"

    def test_placeholder_is_substring_of_another(self):
        """Ensure substrings don't cause incorrect replacements."""
        mapping = {
            "[PHONE]": "555-1234",
            "[PHONE_EXT]": "ext 99",
        }
        deidentified = "Call [PHONE] [PHONE_EXT]"
        result = reidentify(deidentified, mapping)
        assert result == "Call 555-1234 ext 99"

    def test_adjacent_placeholders(self):
        mapping = {"[X]": "A", "[Y]": "B"}
        result = reidentify("[X][Y]", mapping)
        assert result == "AB"


# ---------------------------------------------------------------------------
# Fix 4 (HIGH): DeidentificationResult.to_dict() includes mapping
# ---------------------------------------------------------------------------


class TestDeidentificationResultToDict:
    def test_mapping_included_when_present(self):
        entity = PIIEntity(
            text="John", label="NAME", start=0, end=4, confidence=0.95
        )
        result = DeidentificationResult(
            original_text="John went home",
            deidentified_text="[NAME] went home",
            pii_entities=[entity],
            method="mask",
            timestamp=datetime.now(),
            mapping={"[NAME]": "John"},
        )
        d = result.to_dict()
        assert "mapping" in d
        assert d["mapping"] == {"[NAME]": "John"}

    def test_mapping_absent_when_none(self):
        entity = PIIEntity(
            text="John", label="NAME", start=0, end=4, confidence=0.95
        )
        result = DeidentificationResult(
            original_text="John went home",
            deidentified_text="[NAME] went home",
            pii_entities=[entity],
            method="mask",
            timestamp=datetime.now(),
            mapping=None,
        )
        d = result.to_dict()
        assert "mapping" not in d

    def test_roundtrip_with_mapping(self):
        """Mapping survives serialization for reidentification workflows."""
        entity = PIIEntity(
            text="John", label="NAME", start=0, end=4, confidence=0.95
        )
        result = DeidentificationResult(
            original_text="John called",
            deidentified_text="[NAME] called",
            pii_entities=[entity],
            method="mask",
            timestamp=datetime.now(),
            mapping={"[NAME]": "John"},
        )
        d = result.to_dict()
        restored = reidentify(d["deidentified_text"], d["mapping"])
        assert restored == "John called"


# ---------------------------------------------------------------------------
# Fix 5 (MEDIUM): Thread-safe config
# ---------------------------------------------------------------------------


class TestThreadSafeConfig:
    def test_concurrent_reads_and_writes(self):
        """Config reads/writes from multiple threads don't race."""
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    set_config(OpenMedConfig(timeout=100 + n * 1000 + i))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    cfg = get_config()
                    assert isinstance(cfg, OpenMedConfig)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(i,)) for i in range(3)
        ] + [threading.Thread(target=reader) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# Fix 6 (MEDIUM): TOML parser uses tomllib
# ---------------------------------------------------------------------------


class TestTomllibParser:
    def test_loads_flat_toml(self, tmp_path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('timeout = 120\ndevice = "cpu"\n')
        data = _load_toml(toml_file)
        assert data["timeout"] == 120
        assert data["device"] == "cpu"

    def test_fallback_on_simple_file(self, tmp_path):
        toml_file = tmp_path / "simple.toml"
        toml_file.write_text('log_level = "DEBUG"\ntimeout = 60\n')
        data = _load_toml_fallback(toml_file)
        assert data["log_level"] == "DEBUG"
        assert data["timeout"] == 60

    def test_handles_quoted_strings(self, tmp_path):
        toml_file = tmp_path / "quoted.toml"
        toml_file.write_text('cache_dir = "/tmp/openmed"\n')
        data = _load_toml(toml_file)
        assert data["cache_dir"] == "/tmp/openmed"

    def test_handles_boolean(self, tmp_path):
        toml_file = tmp_path / "bool.toml"
        toml_file.write_text('use_medical_tokenizer = true\n')
        data = _load_toml(toml_file)
        assert data["use_medical_tokenizer"] is True


# ---------------------------------------------------------------------------
# Fix 3 (HIGH): analyze_text() decomposition — smoke test
# ---------------------------------------------------------------------------


class TestAnalyzeTextDecomposition:
    """Verify the extracted helper functions work correctly."""

    def test_build_segments_returns_fallback_without_pysbd(self):
        from openmed import _build_segments

        segments, sd = _build_segments("Hello world", False, "en", False, None)
        assert len(segments) == 1
        assert segments[0]["text"] == "Hello world"
        assert sd is False

    def test_build_chunks_without_sentence_detection(self):
        from openmed import _build_chunks

        segments = [{"index": 0, "text": "test", "start": 0, "end": 4}]
        chunks = _build_chunks("test", segments, False, None)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "test"

    def test_normalize_raw_predictions_single_dict(self):
        from openmed import _normalize_raw_predictions

        result = _normalize_raw_predictions([{"entity": "DISEASE", "start": 0}], 1)
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_normalize_raw_predictions_nested_list(self):
        from openmed import _normalize_raw_predictions

        preds = [[{"entity": "A"}], [{"entity": "B"}]]
        result = _normalize_raw_predictions(preds, 2)
        assert len(result) == 2

    def test_flatten_predictions_empty(self):
        from openmed import _flatten_predictions

        result = _flatten_predictions(
            [], [], [], "no text", False
        )
        assert result == []

    def test_remap_medical_tokens_disabled(self):
        from openmed import _remap_medical_tokens

        class FakeConfig:
            use_medical_tokenizer = False

        preds = [{"entity": "A", "start": 0, "end": 3}]
        result, remapped = _remap_medical_tokens(preds, "abc", FakeConfig())
        assert remapped is False
        assert result == preds
