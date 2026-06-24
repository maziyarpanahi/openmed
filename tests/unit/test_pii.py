"""Unit tests for PII extraction and de-identification module."""

from __future__ import annotations

import builtins
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openmed.core.pii import (
    DeidentificationResult,
    PIIEntity,
    _generate_fake_pii,
    _random_nonzero_shift,
    _redact_entity,
    _shift_date,
    _strip_accents,
    deidentify,
    extract_pii,
    reidentify,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pii_entities():
    """Mock PII entities for testing."""
    return [
        EntityPrediction(
            text="John Doe", label="NAME", start=8, end=16, confidence=0.95
        ),
        EntityPrediction(
            text="555-1234", label="PHONE", start=20, end=28, confidence=0.90
        ),
        EntityPrediction(
            text="john@email.com", label="EMAIL", start=32, end=46, confidence=0.92
        ),
    ]


@pytest.fixture
def mock_analyze_result(mock_pii_entities):
    """Mock PredictionResult for testing."""
    return PredictionResult(
        text="Patient John Doe at 555-1234 or john@email.com",
        entities=mock_pii_entities,
        model_name="test_pii_model",
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# PIIEntity Tests
# ---------------------------------------------------------------------------


class TestPIIEntity:
    """Tests for PIIEntity dataclass."""

    def test_basic_creation(self):
        """Test creating PIIEntity with basic attributes."""
        entity = PIIEntity(
            text="John Doe",
            label="NAME",
            start=0,
            end=8,
            confidence=0.95,
            entity_type="NAME",
        )
        assert entity.text == "John Doe"
        assert entity.label == "NAME"
        assert entity.entity_type == "NAME"
        assert entity.start == 0
        assert entity.end == 8
        assert entity.confidence == 0.95

    def test_entity_type_defaults_to_label(self):
        """Test entity_type is set from label if not provided."""
        entity = PIIEntity(
            text="test@example.com",
            label="EMAIL",
            start=0,
            end=16,
            confidence=0.88,
        )
        assert entity.entity_type == "EMAIL"

    def test_pii_specific_attributes(self):
        """Test PII-specific attributes."""
        entity = PIIEntity(
            text="555-1234",
            label="PHONE",
            start=0,
            end=8,
            confidence=0.90,
            redacted_text="[PHONE]",
            original_text="555-1234",
            hash_value="abc123",
        )
        assert entity.redacted_text == "[PHONE]"
        assert entity.original_text == "555-1234"
        assert entity.hash_value == "abc123"


# ---------------------------------------------------------------------------
# DeidentificationResult Tests
# ---------------------------------------------------------------------------


class TestDeidentificationResult:
    """Tests for DeidentificationResult dataclass."""

    def test_basic_creation(self):
        """Test creating DeidentificationResult."""
        entities = [
            PIIEntity(text="John", label="NAME", start=0, end=4, confidence=0.95)
        ]
        result = DeidentificationResult(
            original_text="Patient John",
            deidentified_text="Patient [NAME]",
            pii_entities=entities,
            method="mask",
            timestamp=datetime.now(),
        )
        assert result.original_text == "Patient John"
        assert result.deidentified_text == "Patient [NAME]"
        assert len(result.pii_entities) == 1
        assert result.method == "mask"
        assert result.mapping is None

    def test_to_dict(self):
        """Test converting result to dictionary."""
        entities = [
            PIIEntity(
                text="John Doe",
                label="NAME",
                start=0,
                end=8,
                confidence=0.95,
                redacted_text="[NAME]",
            )
        ]
        result = DeidentificationResult(
            original_text="John Doe",
            deidentified_text="[NAME]",
            pii_entities=entities,
            method="mask",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
        )
        result_dict = result.to_dict()

        assert result_dict["original_text"] == "John Doe"
        assert result_dict["deidentified_text"] == "[NAME]"
        assert result_dict["method"] == "mask"
        assert result_dict["num_entities_redacted"] == 1
        assert "timestamp" in result_dict
        assert len(result_dict["pii_entities"]) == 1
        assert result_dict["pii_entities"][0]["text"] == "John Doe"
        assert result_dict["pii_entities"][0]["label"] == "NAME"

    def test_to_dict_with_multiple_entities(self):
        """Test to_dict with multiple entities."""
        entities = [
            PIIEntity(
                text="John",
                label="NAME",
                start=0,
                end=4,
                confidence=0.95,
                redacted_text="[NAME]",
            ),
            PIIEntity(
                text="555-1234",
                label="PHONE",
                start=8,
                end=16,
                confidence=0.90,
                redacted_text="[PHONE]",
            ),
        ]
        result = DeidentificationResult(
            original_text="John at 555-1234",
            deidentified_text="[NAME] at [PHONE]",
            pii_entities=entities,
            method="mask",
            timestamp=datetime.now(),
        )
        result_dict = result.to_dict()
        assert result_dict["num_entities_redacted"] == 2
        assert len(result_dict["pii_entities"]) == 2


# ---------------------------------------------------------------------------
# extract_pii Tests
# ---------------------------------------------------------------------------


class TestExtractPII:
    """Tests for extract_pii function."""

    @patch("openmed.analyze_text")
    def test_extract_pii_calls_analyze_text(self, mock_analyze):
        """Test extract_pii calls analyze_text with correct parameters."""
        mock_analyze.return_value = PredictionResult(
            text="Test text",
            entities=[],
            model_name="test_model",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii(
            "Test text",
            model_name="test_pii_model",
            confidence_threshold=0.6,
        )

        mock_analyze.assert_called_once_with(
            "Test text",
            model_name="test_pii_model",
            confidence_threshold=0.6,
            config=None,
            loader=None,
            group_entities=True,
        )

    @patch("openmed.analyze_text")
    def test_extract_pii_returns_analysis_result(
        self, mock_analyze, mock_analyze_result
    ):
        """Test extract_pii returns PredictionResult."""
        mock_analyze.return_value = mock_analyze_result

        result = extract_pii("Test text")

        assert isinstance(result, PredictionResult)
        assert len(result.entities) == 3
        assert result.entities[0].label == "NAME"
        assert result.entities[1].label == "PHONE"
        assert result.entities[2].label == "EMAIL"

    @patch("openmed.analyze_text")
    def test_extract_pii_default_model(self, mock_analyze):
        """Test extract_pii uses default model."""
        mock_analyze.return_value = PredictionResult(
            text="Test",
            entities=[],
            model_name="default",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("Test")

        call_args = mock_analyze.call_args
        assert (
            call_args[1]["model_name"]
            == "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
        )

    @patch("openmed.analyze_text")
    def test_extract_pii_forwards_loader(self, mock_analyze):
        """Test extract_pii forwards an explicit loader."""
        mock_analyze.return_value = PredictionResult(
            text="Test",
            entities=[],
            model_name="default",
            timestamp=datetime.now().isoformat(),
        )
        loader = MagicMock()

        extract_pii("Test", loader=loader)

        assert mock_analyze.call_args.kwargs["loader"] is loader

    @patch("openmed.analyze_text")
    def test_extract_pii_privacy_filter_routes_through_dispatcher(self, mock_analyze):
        """Privacy-filter models route through ``create_privacy_filter_pipeline``
        rather than ``analyze_text``, and skip the regex smart-merging layer
        entirely (the model already does Viterbi-constrained span construction).
        """
        with (
            patch("openmed.core.backends.create_privacy_filter_pipeline") as mock_be,
            patch(
                "openmed.core.pii_entity_merger.merge_entities_with_semantic_units",
                return_value=[],
            ) as mock_merge,
        ):
            mock_be.return_value = lambda text: [
                {
                    "entity_group": "SSN",
                    "score": 0.95,
                    "word": "123-45-6789",
                    "start": 13,
                    "end": 24,
                }
            ]
            result = extract_pii(
                "Patient SSN: 123-45-6789",
                model_name="OpenMed/privacy-filter-mlx",
                use_smart_merging=True,
            )

        # analyze_text bypassed entirely
        mock_analyze.assert_not_called()
        # Smart merging skipped
        mock_merge.assert_not_called()
        # Backend dispatcher invoked with the requested model name
        mock_be.assert_called_once_with("OpenMed/privacy-filter-mlx")
        # Pipeline output preserved
        assert len(result.entities) == 1
        assert result.entities[0].label == "SSN"

    @patch("openmed.analyze_text")
    def test_extract_pii_local_privacy_filter_artifact_routes_through_dispatcher(
        self, mock_analyze, tmp_path
    ):
        """Local MLX artifacts identified by manifest also bypass smart merging."""
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / "openmed-mlx.json").write_text(
            '{"task":"token-classification","family":"openai-privacy-filter"}'
        )

        with (
            patch("openmed.core.backends.create_privacy_filter_pipeline") as mock_be,
            patch(
                "openmed.core.pii_entity_merger.merge_entities_with_semantic_units",
                return_value=[],
            ) as mock_merge,
        ):
            mock_be.return_value = lambda text: []
            extract_pii(
                "Patient MRN: ABC-123",
                model_name=str(artifact_dir),
                use_smart_merging=True,
            )

        mock_analyze.assert_not_called()
        mock_merge.assert_not_called()
        mock_be.assert_called_once_with(str(artifact_dir))


# ---------------------------------------------------------------------------
# deidentify Tests
# ---------------------------------------------------------------------------


class TestDeidentify:
    """Tests for deidentify function."""

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_mask_method(self, mock_extract):
        """Test deidentify with mask method."""
        mock_extract.return_value = PredictionResult(
            text="Patient John Doe",
            entities=[
                EntityPrediction(
                    text="John Doe", label="NAME", start=8, end=16, confidence=0.95
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify("Patient John Doe", method="mask")

        assert result.deidentified_text == "Patient [NAME]"
        assert len(result.pii_entities) == 1
        assert result.pii_entities[0].redacted_text == "[NAME]"
        assert result.method == "mask"

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_remove_method(self, mock_extract):
        """Test deidentify with remove method."""
        mock_extract.return_value = PredictionResult(
            text="Call 555-1234",
            entities=[
                EntityPrediction(
                    text="555-1234", label="PHONE", start=5, end=13, confidence=0.90
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify("Call 555-1234", method="remove")

        assert result.deidentified_text == "Call "
        assert result.pii_entities[0].redacted_text == ""

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_replace_method(self, mock_extract):
        """Test deidentify with replace method."""
        mock_extract.return_value = PredictionResult(
            text="Email: test@example.com",
            entities=[
                EntityPrediction(
                    text="test@example.com",
                    label="EMAIL",
                    start=7,
                    end=23,
                    confidence=0.92,
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify("Email: test@example.com", method="replace")

        assert "test@example.com" not in result.deidentified_text
        assert "@" in result.deidentified_text  # Should have a fake email

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_hash_method(self, mock_extract):
        """Test deidentify with hash method."""
        mock_extract.return_value = PredictionResult(
            text="Patient John Doe",
            entities=[
                EntityPrediction(
                    text="John Doe", label="NAME", start=8, end=16, confidence=0.95
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify("Patient John Doe", method="hash")

        assert result.deidentified_text.startswith("Patient NAME_")
        assert len(result.pii_entities[0].redacted_text or "") > 5  # NAME_<hash>
        assert result.pii_entities[0].hash_value is not None

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_with_multiple_entities(self, mock_extract):
        """Test deidentify handles multiple entities correctly."""
        mock_extract.return_value = PredictionResult(
            text="John Doe at 555-1234",
            entities=[
                EntityPrediction(
                    text="John Doe", label="NAME", start=0, end=8, confidence=0.95
                ),
                EntityPrediction(
                    text="555-1234", label="PHONE", start=12, end=20, confidence=0.90
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify("John Doe at 555-1234", method="mask")

        assert result.deidentified_text == "[NAME] at [PHONE]"
        assert len(result.pii_entities) == 2

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_with_keep_mapping(self, mock_extract):
        """Test deidentify stores mapping when keep_mapping=True."""
        mock_extract.return_value = PredictionResult(
            text="Patient John Doe",
            entities=[
                EntityPrediction(
                    text="John Doe", label="NAME", start=8, end=16, confidence=0.95
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify("Patient John Doe", method="mask", keep_mapping=True)

        assert result.mapping is not None
        assert "[NAME]" in result.mapping
        assert result.mapping["[NAME]"] == "John Doe"

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_without_keep_mapping(self, mock_extract):
        """Test deidentify doesn't store mapping when keep_mapping=False."""
        mock_extract.return_value = PredictionResult(
            text="Patient John Doe",
            entities=[
                EntityPrediction(
                    text="John Doe", label="NAME", start=8, end=16, confidence=0.95
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify("Patient John Doe", method="mask", keep_mapping=False)

        assert result.mapping is None

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_empty_text(self, mock_extract):
        """Test deidentify handles empty text."""
        mock_extract.return_value = PredictionResult(
            text="",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify("", method="mask")

        assert result.deidentified_text == ""
        assert len(result.pii_entities) == 0

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_no_entities(self, mock_extract):
        """Test deidentify handles text with no PII."""
        mock_extract.return_value = PredictionResult(
            text="No PII here",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify("No PII here", method="mask")

        assert result.deidentified_text == "No PII here"
        assert len(result.pii_entities) == 0

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_confidence_threshold(self, mock_extract):
        """Test deidentify uses custom confidence threshold."""
        mock_extract.return_value = PredictionResult(
            text="Test",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        deidentify("Test", confidence_threshold=0.8)

        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args[0][2] == 0.8  # confidence_threshold parameter

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_forwards_loader(self, mock_extract):
        """Test deidentify forwards an explicit loader to extract_pii."""
        mock_extract.return_value = PredictionResult(
            text="Test",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )
        loader = MagicMock()

        deidentify("Test", loader=loader)

        assert mock_extract.call_args.kwargs["loader"] is loader

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_shift_dates_method_does_not_require_alias(self, mock_extract):
        """Test method='shift_dates' shifts dates without the legacy boolean flag."""
        mock_extract.return_value = PredictionResult(
            text="DOB 01/15/2020",
            entities=[
                EntityPrediction(
                    text="01/15/2020", label="DATE", start=4, end=14, confidence=0.95
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify(
            "DOB 01/15/2020",
            method="shift_dates",
            date_shift_days=30,
        )

        assert result.method == "shift_dates"
        assert result.deidentified_text == "DOB 02/14/2020"

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_shift_dates_uses_lowercase_default_model_label(
        self, mock_extract
    ):
        """Default English model lowercase ``date`` labels must shift."""
        mock_extract.return_value = PredictionResult(
            text="DOB 01/15/2020",
            entities=[
                EntityPrediction(
                    text="01/15/2020", label="date", start=4, end=14, confidence=0.95
                )
            ],
            model_name="OpenMed-PII-SuperClinical-Small-44M-v1",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify(
            "DOB 01/15/2020",
            method="shift_dates",
            date_shift_days=30,
        )

        assert result.deidentified_text == "DOB 02/14/2020"
        assert "[DATE]" not in result.deidentified_text
        assert "[date]" not in result.deidentified_text

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_shift_dates_recognizes_date_of_birth_label(self, mock_extract):
        """Raw ``date_of_birth`` labels normalize to a date label before shifting."""
        mock_extract.return_value = PredictionResult(
            text="DOB 01/15/2020",
            entities=[
                EntityPrediction(
                    text="01/15/2020",
                    label="date_of_birth",
                    start=4,
                    end=14,
                    confidence=0.95,
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify(
            "DOB 01/15/2020",
            method="shift_dates",
            date_shift_days=30,
        )

        assert result.deidentified_text == "DOB 02/14/2020"

    @patch("openmed.core.pii.extract_pii")
    def test_shift_dates_keep_mapping_does_not_suffix_shifted_dates(self, mock_extract):
        """Shifted dates should not be counted as mask placeholders."""
        text = "DOB 01/15/2020 and 01/16/2020"
        mock_extract.return_value = PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text="01/15/2020", label="date", start=4, end=14, confidence=0.95
                ),
                EntityPrediction(
                    text="01/16/2020", label="date", start=19, end=29, confidence=0.95
                ),
            ],
            model_name="OpenMed-PII-SuperClinical-Small-44M-v1",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify(
            text,
            method="shift_dates",
            date_shift_days=30,
            keep_mapping=True,
        )

        assert result.deidentified_text == "DOB 02/14/2020 and 02/15/2020"
        assert result.mapping == {
            "02/14/2020": "01/15/2020",
            "02/15/2020": "01/16/2020",
        }

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_shift_dates_alias_promotes_method(self, mock_extract):
        """Test the legacy shift_dates boolean is treated as an alias."""
        mock_extract.return_value = PredictionResult(
            text="DOB 01/15/2020",
            entities=[
                EntityPrediction(
                    text="01/15/2020", label="DATE", start=4, end=14, confidence=0.95
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify(
            "DOB 01/15/2020",
            method="mask",
            shift_dates=True,
            date_shift_days=30,
        )

        assert result.method == "shift_dates"
        assert result.deidentified_text == "DOB 02/14/2020"

    def test_deidentify_rejects_conflicting_shift_dates_flag(self):
        """Test contradictory shift_dates combinations raise an error."""
        with pytest.raises(ValueError, match="shift_dates=false conflicts"):
            deidentify(
                "DOB 01/15/2020",
                method="shift_dates",
                shift_dates=False,
            )

    def test_deidentify_rejects_date_shift_days_without_shift_method(self):
        """Test date_shift_days requires shift_dates mode."""
        with pytest.raises(ValueError, match="date_shift_days requires"):
            deidentify(
                "DOB 01/15/2020",
                method="mask",
                date_shift_days=30,
            )

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_auto_shift_dates_retries_zero_offset(
        self, mock_extract, monkeypatch
    ):
        """Auto-selected shift_dates offsets must never be a no-op.

        ``date_shift_days=None`` used to fall back to ``random.randint(-365,
        365)``, which is inclusive of 0 and silently left every date in the
        document unchanged. Force the first draw to 0 so the regression is
        deterministic.
        """
        original = "01/15/2020"
        mock_extract.return_value = PredictionResult(
            text=f"DOB {original}",
            entities=[
                EntityPrediction(
                    text=original, label="DATE", start=4, end=14, confidence=0.95
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )
        draws = iter([0, 30])

        def fake_randint(low, high):
            assert (low, high) == (-365, 365)
            return next(draws)

        monkeypatch.setattr("openmed.core.pii.random.randint", fake_randint)

        result = deidentify(
            f"DOB {original}",
            method="shift_dates",
            date_shift_days=None,
            keep_year=False,
        )

        assert result.deidentified_text == "DOB 02/14/2020"

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_explicit_zero_shift_remains_allowed(self, mock_extract):
        """An explicit caller-supplied zero shift is a deliberate no-op."""
        original = "01/15/2020"
        mock_extract.return_value = PredictionResult(
            text=f"DOB {original}",
            entities=[
                EntityPrediction(
                    text=original, label="DATE", start=4, end=14, confidence=0.95
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify(
            f"DOB {original}",
            method="shift_dates",
            date_shift_days=0,
            keep_year=False,
        )

        assert result.deidentified_text == f"DOB {original}"


# ---------------------------------------------------------------------------
# _redact_entity Tests
# ---------------------------------------------------------------------------


class TestRedactEntity:
    """Tests for _redact_entity helper function."""

    def test_redact_mask(self):
        """Test mask redaction method."""
        entity = PIIEntity(
            text="John Doe", label="NAME", start=0, end=8, confidence=0.95
        )
        result = _redact_entity(entity, "mask")
        assert result == "[NAME]"

    def test_redact_remove(self):
        """Test remove redaction method."""
        entity = PIIEntity(
            text="555-1234", label="PHONE", start=0, end=8, confidence=0.90
        )
        result = _redact_entity(entity, "remove")
        assert result == ""

    def test_redact_replace(self):
        """Test replace redaction method."""
        entity = PIIEntity(
            text="John Doe", label="NAME", start=0, end=8, confidence=0.95
        )
        result = _redact_entity(entity, "replace")
        # Should return one of the fake names
        assert result in ["Jane Smith", "John Doe", "Alex Johnson", "Sam Taylor"]

    def test_redact_hash(self):
        """Test hash redaction method."""
        entity = PIIEntity(
            text="John Doe", label="NAME", start=0, end=8, confidence=0.95
        )
        result = _redact_entity(entity, "hash")
        assert result.startswith("NAME_")
        assert len(result) > 5  # NAME_<8-char hash>
        assert entity.hash_value is not None

    def test_redact_hash_consistent(self):
        """Test hash redaction is consistent for same input."""
        entity1 = PIIEntity(
            text="John Doe", label="NAME", start=0, end=8, confidence=0.95
        )
        entity2 = PIIEntity(
            text="John Doe", label="NAME", start=0, end=8, confidence=0.95
        )
        result1 = _redact_entity(entity1, "hash")
        result2 = _redact_entity(entity2, "hash")
        assert result1 == result2

    def test_redact_shift_dates_for_date_entity(self):
        """Test shift_dates method for DATE entities."""
        entity = PIIEntity(
            text="01/15/2020", label="DATE", start=0, end=10, confidence=0.90
        )
        result = _redact_entity(entity, "shift_dates", date_shift_days=30)
        # Date shifting now properly implemented - shifts by 30 days
        assert result == "02/14/2020"

    def test_redact_shift_dates_for_non_date(self):
        """Test shift_dates method masks non-DATE entities."""
        entity = PIIEntity(
            text="John Doe", label="NAME", start=0, end=8, confidence=0.95
        )
        result = _redact_entity(entity, "shift_dates", date_shift_days=30)
        assert result == "[NAME]"

    def test_redact_shift_dates_uses_canonical_label(self):
        """shift_dates must shift dates whose raw label is not literally 'DATE'.

        The default English model emits a lowercase ``date`` label; comparing
        the raw ``entity_type`` to ``"DATE"`` made such dates silently fall
        through to masking. Regression test for the canonical-label fix.
        """
        entity = PIIEntity(
            text="01/15/2020",
            label="date",
            start=0,
            end=10,
            confidence=0.90,
            entity_type="date",
            canonical_label="DATE",
        )
        result = _redact_entity(entity, "shift_dates", date_shift_days=30)
        assert result == "02/14/2020"


# ---------------------------------------------------------------------------
# _generate_fake_pii Tests
# ---------------------------------------------------------------------------


class TestGenerateFakePII:
    """Tests for _generate_fake_pii helper function."""

    def test_generate_fake_name(self):
        """Test generating fake names."""
        result = _generate_fake_pii("NAME")
        assert result in ["Jane Smith", "John Doe", "Alex Johnson", "Sam Taylor"]

    def test_generate_fake_email(self):
        """Test generating fake emails."""
        result = _generate_fake_pii("EMAIL")
        assert result in ["patient@example.com", "contact@example.org"]

    def test_generate_fake_phone(self):
        """Test generating fake phone numbers."""
        result = _generate_fake_pii("PHONE")
        assert result in ["555-0123", "555-0456", "555-0789"]

    def test_generate_fake_unknown_type(self):
        """Test generating placeholder for unknown types."""
        result = _generate_fake_pii("UNKNOWN_TYPE")
        assert result == "[UNKNOWN_TYPE]"

    def test_generate_fake_consistent_types(self):
        """Test fake data is from predefined list (not random strings)."""
        # Call multiple times, should always be from the list
        for _ in range(10):
            result = _generate_fake_pii("NAME")
            assert result in ["Jane Smith", "John Doe", "Alex Johnson", "Sam Taylor"]


# ---------------------------------------------------------------------------
# _shift_date Tests
# ---------------------------------------------------------------------------


class TestShiftDate:
    """Tests for _shift_date helper function."""

    def test_shift_date_us_format(self):
        """Test date shifting with US format MM/DD/YYYY."""
        result = _shift_date("01/15/2020", 30)
        assert result == "02/14/2020"

    def test_shift_date_with_keep_year(self):
        """Test shift_date with keep_year parameter keeps the year."""
        result = _shift_date("12/15/2020", 30, keep_year=True)
        # Shifts by 30 days (would be 01/14/2021) but keeps year as 2020
        assert result == "01/14/2020"

    def test_shift_date_iso_format(self):
        """Test date shifting with ISO format YYYY-MM-DD."""
        result = _shift_date("2020-01-15", 30)
        assert result == "2020-02-14"

    def test_shift_date_negative_shift(self):
        """Test date shifting backwards."""
        result = _shift_date("01/15/2020", -30)
        assert result == "12/16/2020"  # With keep_year=True (default)

    def test_shift_date_invalid_format(self):
        """Test shift_date with unparseable format returns placeholder."""
        result = _shift_date("not-a-date", 30)
        assert result == "[DATE_SHIFTED]"

    def test_shift_date_keep_year_handles_leap_day(self):
        """keep_year shifting onto Feb 29 must not fall through to a placeholder.

        02/28/2019 + 366 days lands on 2020-02-29 (a leap day). Restoring the
        original year 2019 (not a leap year) used to raise ValueError and
        degrade the whole result to ``[DATE_SHIFTED]``; it should clamp to
        Feb 28 instead.
        """
        result = _shift_date("02/28/2019", 366, keep_year=True)
        assert result == "02/28/2019"


# ---------------------------------------------------------------------------
# _random_nonzero_shift Tests
# ---------------------------------------------------------------------------


class TestRandomNonzeroShift:
    """Tests for the _random_nonzero_shift helper function."""

    def test_random_nonzero_shift_retries_zero(self, monkeypatch):
        """A zero draw must be retried until a non-zero offset is selected."""
        calls = []
        draws = iter([0, -7])

        def fake_randint(low, high):
            calls.append((low, high))
            return next(draws)

        monkeypatch.setattr("openmed.core.pii.random.randint", fake_randint)

        assert _random_nonzero_shift(low=-10, high=10) == -7
        assert calls == [(-10, 10), (-10, 10)]

    def test_random_nonzero_shift_respects_one_sided_ranges(self, monkeypatch):
        """Ranges that do not contain zero should be passed through unchanged."""
        calls = []
        draws = iter([-3, 4])

        def fake_randint(low, high):
            calls.append((low, high))
            return next(draws)

        monkeypatch.setattr("openmed.core.pii.random.randint", fake_randint)

        assert _random_nonzero_shift(low=-10, high=-1) == -3
        assert _random_nonzero_shift(low=1, high=10) == 4
        assert calls == [(-10, -1), (1, 10)]

    def test_random_nonzero_shift_rejects_invalid_ranges(self):
        """The configured range must contain at least one non-zero value."""
        with pytest.raises(ValueError, match="low must be less"):
            _random_nonzero_shift(low=10, high=-10)
        with pytest.raises(ValueError, match="non-zero shift"):
            _random_nonzero_shift(low=0, high=0)


# ---------------------------------------------------------------------------
# reidentify Tests
# ---------------------------------------------------------------------------


class TestReidentify:
    """Tests for reidentify function."""

    def test_reidentify_basic(self):
        """Test re-identification with simple mapping."""
        mapping = {"[NAME]": "John Doe", "[PHONE]": "555-1234"}
        deidentified = "Patient [NAME] at [PHONE]"

        result = reidentify(deidentified, mapping)

        assert result == "Patient John Doe at 555-1234"

    def test_reidentify_single_entity(self):
        """Test re-identification with single entity."""
        mapping = {"[EMAIL]": "john@example.com"}
        deidentified = "Contact: [EMAIL]"

        result = reidentify(deidentified, mapping)

        assert result == "Contact: john@example.com"

    def test_reidentify_empty_mapping(self):
        """Test re-identification with empty mapping."""
        mapping = {}
        deidentified = "No changes [NAME]"

        result = reidentify(deidentified, mapping)

        assert result == "No changes [NAME]"

    def test_reidentify_no_placeholders(self):
        """Test re-identification when text has no placeholders."""
        mapping = {"[NAME]": "John Doe"}
        deidentified = "Already clean text"

        result = reidentify(deidentified, mapping)

        assert result == "Already clean text"

    def test_reidentify_multiple_occurrences(self):
        """Test re-identification replaces all occurrences."""
        mapping = {"[NAME]": "John Doe"}
        deidentified = "[NAME] called [NAME] twice"

        result = reidentify(deidentified, mapping)

        assert result == "John Doe called John Doe twice"


# ---------------------------------------------------------------------------
# Multilingual PII Tests
# ---------------------------------------------------------------------------


class TestMultilingualPII:
    """Tests for multilingual PII detection and de-identification."""

    def test_extract_pii_unsupported_language_raises(self):
        """Test that unsupported language raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported language"):
            extract_pii("test", lang="ko")

    @patch("openmed.analyze_text")
    def test_extract_pii_french_uses_french_model(self, mock_analyze):
        """Test that lang='fr' auto-resolves to French default model."""
        mock_analyze.return_value = PredictionResult(
            text="Né le 15/01/1970",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("Né le 15/01/1970", lang="fr")

        call_args = mock_analyze.call_args
        assert "French" in call_args[1]["model_name"]

    @patch("openmed.analyze_text")
    def test_extract_pii_german_uses_german_model(self, mock_analyze):
        """Test that lang='de' auto-resolves to German default model."""
        mock_analyze.return_value = PredictionResult(
            text="Geboren am 15.01.1970",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("Geboren am 15.01.1970", lang="de")

        call_args = mock_analyze.call_args
        assert "German" in call_args[1]["model_name"]

    @patch("openmed.analyze_text")
    def test_extract_pii_italian_uses_italian_model(self, mock_analyze):
        """Test that lang='it' auto-resolves to Italian default model."""
        mock_analyze.return_value = PredictionResult(
            text="Nato il 15/01/1970",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("Nato il 15/01/1970", lang="it")

        call_args = mock_analyze.call_args
        assert "Italian" in call_args[1]["model_name"]

    @patch("openmed.analyze_text")
    def test_extract_pii_dutch_uses_dutch_model(self, mock_analyze):
        """Test that lang='nl' auto-resolves to Dutch default model."""
        mock_analyze.return_value = PredictionResult(
            text="Geboren op 15/01/1970",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("Geboren op 15/01/1970", lang="nl")

        call_args = mock_analyze.call_args
        assert "Dutch" in call_args[1]["model_name"]

    @patch("openmed.analyze_text")
    def test_extract_pii_hindi_uses_hindi_model(self, mock_analyze):
        """Test that lang='hi' auto-resolves to Hindi default model."""
        mock_analyze.return_value = PredictionResult(
            text="जन्म तिथि 15/01/1970",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("जन्म तिथि 15/01/1970", lang="hi")

        call_args = mock_analyze.call_args
        assert "Hindi" in call_args[1]["model_name"]

    @patch("openmed.analyze_text")
    def test_extract_pii_telugu_uses_telugu_model(self, mock_analyze):
        """Test that lang='te' auto-resolves to Telugu default model."""
        mock_analyze.return_value = PredictionResult(
            text="జన్మ తేదీ 15/01/1970",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("జన్మ తేదీ 15/01/1970", lang="te")

        call_args = mock_analyze.call_args
        assert "Telugu" in call_args[1]["model_name"]

    @patch("openmed.analyze_text")
    def test_extract_pii_portuguese_uses_portuguese_model(self, mock_analyze):
        """Test that lang='pt' auto-resolves to Portuguese default model."""
        mock_analyze.return_value = PredictionResult(
            text="Nascimento 15/03/1985",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("Nascimento 15/03/1985", lang="pt")

        call_args = mock_analyze.call_args
        assert "Portuguese" in call_args[1]["model_name"]
        assert "SnowflakeMed-Large-568M" in call_args[1]["model_name"]

    @patch("openmed.analyze_text")
    def test_extract_pii_arabic_uses_arabic_model(self, mock_analyze):
        """Test that lang='ar' auto-resolves to Arabic default model."""
        mock_analyze.return_value = PredictionResult(
            text="\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0645\u064a\u0644\u0627\u062f 15/03/1985",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii(
            "\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0645\u064a\u0644\u0627\u062f 15/03/1985",
            lang="ar",
        )

        call_args = mock_analyze.call_args
        assert "Arabic" in call_args[1]["model_name"]
        assert "SnowflakeMed-Large-568M" in call_args[1]["model_name"]

    @patch("openmed.analyze_text")
    def test_extract_pii_japanese_uses_japanese_model(self, mock_analyze):
        """Test that lang='ja' auto-resolves to Japanese default model."""
        mock_analyze.return_value = PredictionResult(
            text="\u751f\u5e74\u6708\u65e5 1985\u5e743\u670815\u65e5",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("\u751f\u5e74\u6708\u65e5 1985\u5e743\u670815\u65e5", lang="ja")

        call_args = mock_analyze.call_args
        assert "Japanese" in call_args[1]["model_name"]
        assert "BigMed-Large-560M" in call_args[1]["model_name"]

    @patch("openmed.analyze_text")
    def test_extract_pii_turkish_uses_turkish_model(self, mock_analyze):
        """Test that lang='tr' auto-resolves to Turkish default model."""
        mock_analyze.return_value = PredictionResult(
            text="Do\u011fum tarihi 15.03.1985",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("Do\u011fum tarihi 15.03.1985", lang="tr")

        call_args = mock_analyze.call_args
        assert "Turkish" in call_args[1]["model_name"]
        assert "SuperClinical-Small-44M" in call_args[1]["model_name"]

    @patch("openmed.analyze_text")
    def test_extract_pii_english_backward_compat(self, mock_analyze):
        """Test that lang='en' (default) uses English model."""
        mock_analyze.return_value = PredictionResult(
            text="Dr. Smith",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("Dr. Smith")

        call_args = mock_analyze.call_args
        model_name = call_args[1]["model_name"]
        assert "French" not in model_name
        assert "German" not in model_name
        assert "Italian" not in model_name
        assert "Arabic" not in model_name
        assert "Japanese" not in model_name
        assert "Turkish" not in model_name

    @patch("openmed.analyze_text")
    def test_extract_pii_custom_model_overrides_lang(self, mock_analyze):
        """Test that explicit model_name is used even with lang parameter."""
        mock_analyze.return_value = PredictionResult(
            text="test",
            entities=[],
            model_name="custom",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("test", model_name="custom-model", lang="fr")

        call_args = mock_analyze.call_args
        assert call_args[1]["model_name"] == "custom-model"

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_passes_lang(self, mock_extract):
        """Test that deidentify passes lang parameter to extract_pii."""
        mock_extract.return_value = PredictionResult(
            text="Patient Marie Dupont",
            entities=[
                EntityPrediction(
                    text="Marie Dupont",
                    label="NAME",
                    start=8,
                    end=20,
                    confidence=0.95,
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        deidentify("Patient Marie Dupont", method="mask", lang="fr")

        call_args = mock_extract.call_args
        assert call_args[1]["lang"] == "fr"

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_replace_uses_locale_aware_surrogates(self, mock_extract):
        """``method='replace'`` produces Faker-backed locale-appropriate surrogates.

        With ``consistent=True, seed=...`` we can assert exact equality across
        runs. The surrogate must be (a) non-empty, (b) different from the
        original, and (c) repeatable for the same seed.
        """
        mock_extract.return_value = PredictionResult(
            text="Patient Marie Dupont",
            entities=[
                EntityPrediction(
                    text="Marie Dupont",
                    label="NAME",
                    start=8,
                    end=20,
                    confidence=0.95,
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        r1 = deidentify(
            "Patient Marie Dupont",
            method="replace",
            lang="fr",
            consistent=True,
            seed=42,
        )
        r2 = deidentify(
            "Patient Marie Dupont",
            method="replace",
            lang="fr",
            consistent=True,
            seed=42,
        )

        surrogate = r1.pii_entities[0].redacted_text
        assert surrogate
        assert surrogate != "Marie Dupont"
        # Determinism: same seed -> identical surrogate
        assert r1.pii_entities[0].redacted_text == r2.pii_entities[0].redacted_text

    def test_generate_fake_pii_french(self):
        """Test fake PII generation with French locale."""
        result = _generate_fake_pii("NAME", lang="fr")
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

        assert result in LANGUAGE_FAKE_DATA["fr"]["NAME"]

    def test_generate_fake_pii_german(self):
        """Test fake PII generation with German locale."""
        result = _generate_fake_pii("NAME", lang="de")
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

        assert result in LANGUAGE_FAKE_DATA["de"]["NAME"]

    def test_generate_fake_pii_dutch(self):
        """Test fake PII generation with Dutch locale."""
        result = _generate_fake_pii("NAME", lang="nl")
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

        assert result in LANGUAGE_FAKE_DATA["nl"]["NAME"]

    def test_generate_fake_pii_hindi(self):
        """Test fake PII generation with Hindi locale."""
        result = _generate_fake_pii("NAME", lang="hi")
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

        assert result in LANGUAGE_FAKE_DATA["hi"]["NAME"]

    def test_generate_fake_pii_telugu(self):
        """Test fake PII generation with Telugu locale."""
        result = _generate_fake_pii("NAME", lang="te")
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

        assert result in LANGUAGE_FAKE_DATA["te"]["NAME"]

    def test_generate_fake_pii_portuguese(self):
        """Test fake PII generation with Portuguese locale."""
        result = _generate_fake_pii("NAME", lang="pt")
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

        assert result in LANGUAGE_FAKE_DATA["pt"]["NAME"]

    def test_generate_fake_pii_arabic(self):
        """Test fake PII generation with Arabic locale."""
        result = _generate_fake_pii("NAME", lang="ar")
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

        assert result in LANGUAGE_FAKE_DATA["ar"]["NAME"]

    def test_generate_fake_pii_japanese(self):
        """Test fake PII generation with Japanese locale."""
        result = _generate_fake_pii("NAME", lang="ja")
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

        assert result in LANGUAGE_FAKE_DATA["ja"]["NAME"]

    def test_generate_fake_pii_turkish(self):
        """Test fake PII generation with Turkish locale."""
        result = _generate_fake_pii("NAME", lang="tr")
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

        assert result in LANGUAGE_FAKE_DATA["tr"]["NAME"]

    def test_generate_fake_pii_portuguese_cpf(self):
        """Test CPF labels use Portuguese fake ID data."""
        result = _generate_fake_pii("cpf", lang="pt")
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

        assert result in LANGUAGE_FAKE_DATA["pt"]["ID_NUM"]

    def test_generate_fake_pii_fallback_to_english(self):
        """Test fake PII falls back to English for missing types."""
        result = _generate_fake_pii("USERNAME", lang="fr")
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA

        assert result in LANGUAGE_FAKE_DATA["fr"]["USERNAME"]

    def test_generate_fake_pii_unknown_type_returns_placeholder(self):
        """Test fake PII for unknown type returns placeholder."""
        result = _generate_fake_pii("UNKNOWN_ENTITY", lang="de")
        assert result == "[UNKNOWN_ENTITY]"

    def test_shift_date_german_format(self):
        """Test date shifting with German DD.MM.YYYY format."""
        result = _shift_date("15.01.2020", 30, lang="de")
        assert result == "14.02.2020"

    def test_shift_date_french_format(self):
        """Test date shifting with French DD/MM/YYYY format."""
        result = _shift_date("15/01/2020", 30, lang="fr")
        # French: day-first, so 15/01/2020 is Jan 15
        assert result == "14/02/2020"

    def test_shift_date_italian_format(self):
        """Test date shifting with Italian DD/MM/YYYY format."""
        result = _shift_date("15/01/2020", 30, lang="it")
        assert result == "14/02/2020"

    def test_shift_date_spanish_format(self):
        """Test date shifting with Spanish DD/MM/YYYY format."""
        result = _shift_date("15/01/2020", 30, lang="es")
        assert result == "14/02/2020"

    def test_shift_date_dutch_format(self):
        """Test date shifting with Dutch DD/MM/YYYY format."""
        result = _shift_date("15/01/2020", 30, lang="nl")
        assert result == "14/02/2020"

    def test_shift_date_hindi_format(self):
        """Test date shifting with Hindi DD/MM/YYYY format."""
        result = _shift_date("15/01/2020", 30, lang="hi")
        assert result == "14/02/2020"

    def test_shift_date_telugu_format(self):
        """Test date shifting with Telugu DD/MM/YYYY format."""
        result = _shift_date("15/01/2020", 30, lang="te")
        assert result == "14/02/2020"

    def test_shift_date_portuguese_format(self):
        """Test date shifting with Portuguese DD/MM/YYYY format."""
        result = _shift_date("15/01/2020", 30, lang="pt")
        assert result == "14/02/2020"

    def test_shift_date_dutch_month_name_format(self):
        """Test Dutch month-name date shifting."""
        result = _shift_date("15 januari 2020", 30, lang="nl")
        assert result == "14 februari 2020"

    def test_shift_date_hindi_month_name_format(self):
        """Test Hindi month-name date shifting."""
        result = _shift_date("15 जनवरी 2020", 30, lang="hi")
        assert result == "14 फ़रवरी 2020"

    def test_shift_date_telugu_month_name_format(self):
        """Test Telugu month-name date shifting."""
        result = _shift_date("15 జనవరి 2020", 30, lang="te")
        assert result == "14 ఫిబ్రవరి 2020"

    def test_shift_date_portuguese_month_name_format(self):
        """Test Portuguese month-name date shifting."""
        result = _shift_date("15 de janeiro de 2020", 30, lang="pt")
        assert result == "14 de fevereiro de 2020"

    def test_shift_date_localized_month_name_without_dateutil(self, monkeypatch):
        """Localized month-name shifting should not depend on python-dateutil."""

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "dateutil" or name.startswith("dateutil."):
                raise ImportError("dateutil unavailable")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        assert _shift_date("15 januari 2020", 30, lang="nl") == "14 februari 2020"
        assert _shift_date("15 जनवरी 2020", 30, lang="hi") == "14 फ़रवरी 2020"
        assert _shift_date("15 జనవరి 2020", 30, lang="te") == "14 ఫిబ్రవరి 2020"


# ---------------------------------------------------------------------------
# Accent Normalization Tests
# ---------------------------------------------------------------------------


class TestStripAccents:
    """Tests for _strip_accents helper."""

    def test_strip_spanish_accents(self):
        assert _strip_accents("María López") == "Maria Lopez"

    def test_strip_preserves_length(self):
        text = "María López García"
        stripped = _strip_accents(text)
        assert len(stripped) == len(text)

    def test_strip_no_accents_unchanged(self):
        assert _strip_accents("John Doe") == "John Doe"

    def test_strip_empty_string(self):
        assert _strip_accents("") == ""

    def test_strip_n_tilde(self):
        assert _strip_accents("niño") == "nino"

    def test_strip_u_diaeresis(self):
        assert _strip_accents("pingüino") == "pinguino"

    def test_strip_preserves_digits_and_punctuation(self):
        assert (
            _strip_accents("DNI: 12345678Z, teléfono: +34 612")
            == "DNI: 12345678Z, telefono: +34 612"
        )


class TestAccentNormalization:
    """Tests for accent normalization in extract_pii."""

    @patch("openmed.analyze_text")
    def test_spanish_auto_normalizes_accents(self, mock_analyze):
        """Test that lang='es' auto-strips accents for model inference."""
        mock_analyze.return_value = PredictionResult(
            text="Maria Lopez",
            entities=[
                EntityPrediction(
                    text="Maria Lopez",
                    label="first_name",
                    start=0,
                    end=11,
                    confidence=0.95,
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = extract_pii("María López", lang="es")

        # Model should receive accent-stripped text
        call_args = mock_analyze.call_args
        assert call_args[0][0] == "Maria Lopez"

        # But result entity text should reference original accented text
        assert result.entities[0].text == "María López"

    @patch("openmed.analyze_text")
    def test_spanish_accent_remapping_with_off_by_one(self, mock_analyze):
        """Off-by-one spans from model are fixed before accent remapping."""
        # Simulate a model returning end=4 instead of 5 for "Maria" (off-by-one)
        mock_analyze.return_value = PredictionResult(
            text="Maria Lopez",
            entities=[
                EntityPrediction(
                    text="Mari", label="first_name", start=0, end=4, confidence=0.95
                ),
                EntityPrediction(
                    text="Lope", label="last_name", start=6, end=10, confidence=0.93
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        # _fix_entity_spans runs inside format_predictions (within analyze_text),
        # NOT inside extract_pii. Since we mock analyze_text, the fix is bypassed.
        # So we pre-apply it manually to simulate the real pipeline.
        from openmed.processing.outputs import OutputFormatter

        fixed_entities = OutputFormatter._fix_entity_spans(
            mock_analyze.return_value.entities, "Maria Lopez"
        )
        mock_analyze.return_value.entities = fixed_entities

        result = extract_pii("María López", lang="es")

        assert result.entities[0].text == "María"
        assert result.entities[0].start == 0
        assert result.entities[0].end == 5
        assert result.entities[1].text == "López"
        assert result.entities[1].start == 6
        assert result.entities[1].end == 11

    @patch("openmed.analyze_text")
    def test_normalize_accents_false_skips(self, mock_analyze):
        """Test that normalize_accents=False sends original text."""
        mock_analyze.return_value = PredictionResult(
            text="María López",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("María López", lang="es", normalize_accents=False)

        call_args = mock_analyze.call_args
        assert call_args[0][0] == "María López"

    @patch("openmed.analyze_text")
    def test_english_no_normalization_by_default(self, mock_analyze):
        """Test that lang='en' does not normalize by default."""
        mock_analyze.return_value = PredictionResult(
            text="José García",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("José García", lang="en")

        call_args = mock_analyze.call_args
        assert call_args[0][0] == "José García"

    @patch("openmed.analyze_text")
    def test_normalize_accents_explicit_true_for_any_lang(self, mock_analyze):
        """Test that normalize_accents=True works for any language."""
        mock_analyze.return_value = PredictionResult(
            text="Jose Garcia",
            entities=[],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        extract_pii("José García", lang="en", normalize_accents=True)

        call_args = mock_analyze.call_args
        assert call_args[0][0] == "Jose Garcia"


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for full PII workflows."""

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_and_reidentify_roundtrip(self, mock_extract):
        """Test de-identify and re-identify round trip."""
        original_text = "Patient John Doe at 555-1234"
        mock_extract.return_value = PredictionResult(
            text=original_text,
            entities=[
                EntityPrediction(
                    text="John Doe", label="NAME", start=8, end=16, confidence=0.95
                ),
                EntityPrediction(
                    text="555-1234", label="PHONE", start=20, end=28, confidence=0.90
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        # De-identify
        deid_result = deidentify(original_text, method="mask", keep_mapping=True)
        assert deid_result.deidentified_text == "Patient [NAME] at [PHONE]"
        assert deid_result.mapping is not None

        # Re-identify
        reidentified = reidentify(deid_result.deidentified_text, deid_result.mapping)
        assert reidentified == original_text

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_result_to_dict(self, mock_extract):
        """Test converting deidentification result to dict."""
        mock_extract.return_value = PredictionResult(
            text="John Doe",
            entities=[
                EntityPrediction(
                    text="John Doe", label="NAME", start=0, end=8, confidence=0.95
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify("John Doe", method="mask")
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "original_text" in result_dict
        assert "deidentified_text" in result_dict
        assert "pii_entities" in result_dict
        assert "method" in result_dict
        assert "timestamp" in result_dict
        assert "num_entities_redacted" in result_dict

    @patch("openmed.core.pii.extract_pii")
    def test_roundtrip_two_persons_mask(self, mock_extract):
        """Test round-trip with two distinct PERSON entities using mask (#204, #222)."""
        original_text = "Dr. Alice Smith met Bob Jones today"
        mock_extract.return_value = PredictionResult(
            text=original_text,
            entities=[
                EntityPrediction(
                    text="Alice Smith", label="NAME", start=4, end=15, confidence=0.95
                ),
                EntityPrediction(
                    text="Bob Jones", label="NAME", start=20, end=29, confidence=0.93
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        deid_result = deidentify(original_text, method="mask", keep_mapping=True)
        # First NAME -> [NAME], second NAME -> [NAME_2]
        assert deid_result.deidentified_text == "Dr. [NAME] met [NAME_2] today"
        assert deid_result.mapping is not None

        reidentified = reidentify(deid_result.deidentified_text, deid_result.mapping)
        assert reidentified == original_text

    @patch("openmed.core.pii.extract_pii")
    def test_repeated_mask_without_mapping_keeps_default_placeholders(
        self, mock_extract
    ):
        """Repeated mask output is unchanged when no reversible mapping is requested."""
        original_text = "Dr. Alice Smith met Bob Jones today"
        mock_extract.return_value = PredictionResult(
            text=original_text,
            entities=[
                EntityPrediction(
                    text="Alice Smith",
                    label="NAME",
                    start=4,
                    end=15,
                    confidence=0.95,
                ),
                EntityPrediction(
                    text="Bob Jones",
                    label="NAME",
                    start=20,
                    end=29,
                    confidence=0.93,
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        deid_result = deidentify(original_text, method="mask", keep_mapping=False)

        assert deid_result.deidentified_text == "Dr. [NAME] met [NAME] today"
        assert [entity.redacted_text for entity in deid_result.pii_entities] == [
            "[NAME]",
            "[NAME]",
        ]
        assert deid_result.mapping is None

    @patch("openmed.core.pii.extract_pii")
    def test_roundtrip_two_dates_mask(self, mock_extract):
        """Test round-trip with two distinct DATE entities using mask."""
        original_text = "Born 1990-01-15 seen 2024-06-20"
        mock_extract.return_value = PredictionResult(
            text=original_text,
            entities=[
                EntityPrediction(
                    text="1990-01-15", label="DATE", start=5, end=15, confidence=0.95
                ),
                EntityPrediction(
                    text="2024-06-20", label="DATE", start=21, end=31, confidence=0.92
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        deid_result = deidentify(original_text, method="mask", keep_mapping=True)
        assert "[DATE]" in deid_result.deidentified_text
        assert "[DATE_2]" in deid_result.deidentified_text

        reidentified = reidentify(deid_result.deidentified_text, deid_result.mapping)
        assert reidentified == original_text

    @patch("openmed.core.pii.extract_pii")
    def test_roundtrip_mixed_repeated_and_unique(self, mock_extract):
        """Test round-trip with mixed repeated and unique entity types."""
        original_text = "Alice and Bob called 555-1234 on 2024-01-01"
        mock_extract.return_value = PredictionResult(
            text=original_text,
            entities=[
                EntityPrediction(
                    text="Alice", label="NAME", start=0, end=5, confidence=0.95
                ),
                EntityPrediction(
                    text="Bob", label="NAME", start=10, end=13, confidence=0.93
                ),
                EntityPrediction(
                    text="555-1234", label="PHONE", start=21, end=29, confidence=0.90
                ),
                EntityPrediction(
                    text="2024-01-01", label="DATE", start=33, end=43, confidence=0.91
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        deid_result = deidentify(original_text, method="mask", keep_mapping=True)
        assert "[NAME]" in deid_result.deidentified_text
        assert "[NAME_2]" in deid_result.deidentified_text
        assert "[PHONE]" in deid_result.deidentified_text
        assert "[DATE]" in deid_result.deidentified_text

        reidentified = reidentify(deid_result.deidentified_text, deid_result.mapping)
        assert reidentified == original_text

    @patch("openmed.core.pii.extract_pii")
    def test_roundtrip_two_persons_hash(self, mock_extract):
        """Test round-trip with two PERSON entities using hash method."""
        original_text = "Alice Smith and Bob Jones"
        mock_extract.return_value = PredictionResult(
            text=original_text,
            entities=[
                EntityPrediction(
                    text="Alice Smith", label="NAME", start=0, end=11, confidence=0.95
                ),
                EntityPrediction(
                    text="Bob Jones", label="NAME", start=16, end=25, confidence=0.93
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        deid_result = deidentify(original_text, method="hash", keep_mapping=True)
        # Hash produces unique values per text, so no counter needed
        reidentified = reidentify(deid_result.deidentified_text, deid_result.mapping)
        assert reidentified == original_text

    @patch("openmed.core.pii.extract_pii")
    def test_roundtrip_two_persons_replace(self, mock_extract):
        """Test round-trip with two PERSON entities using replace method."""
        original_text = "Alice Smith and Bob Jones"
        mock_extract.return_value = PredictionResult(
            text=original_text,
            entities=[
                EntityPrediction(
                    text="Alice Smith",
                    label="NAME",
                    start=0,
                    end=11,
                    confidence=0.95,
                ),
                EntityPrediction(
                    text="Bob Jones",
                    label="NAME",
                    start=16,
                    end=25,
                    confidence=0.93,
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        deid_result = deidentify(
            original_text,
            method="replace",
            keep_mapping=True,
            consistent=True,
            seed=123,
        )

        reidentified = reidentify(deid_result.deidentified_text, deid_result.mapping)
        assert reidentified == original_text

    @patch("openmed.core.pii.extract_pii")
    def test_roundtrip_two_persons_remove(self, mock_extract):
        """Test round-trip with two PERSON entities using remove method."""
        original_text = "Alice Smith and Bob Jones"
        mock_extract.return_value = PredictionResult(
            text=original_text,
            entities=[
                EntityPrediction(
                    text="Alice Smith",
                    label="NAME",
                    start=0,
                    end=11,
                    confidence=0.95,
                ),
                EntityPrediction(
                    text="Bob Jones",
                    label="NAME",
                    start=16,
                    end=25,
                    confidence=0.93,
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        deid_result = deidentify(original_text, method="remove", keep_mapping=True)

        assert deid_result.deidentified_text == "[NAME_REMOVED] and [NAME_REMOVED_2]"
        assert deid_result.mapping == {
            "[NAME_REMOVED]": "Alice Smith",
            "[NAME_REMOVED_2]": "Bob Jones",
        }
        reidentified = reidentify(deid_result.deidentified_text, deid_result.mapping)
        assert reidentified == original_text

    @patch("openmed.core.pii.extract_pii")
    def test_roundtrip_single_entity_unchanged(self, mock_extract):
        """Test that single entities still produce simple placeholders (no counter)."""
        original_text = "Patient John Doe"
        mock_extract.return_value = PredictionResult(
            text=original_text,
            entities=[
                EntityPrediction(
                    text="John Doe", label="NAME", start=8, end=16, confidence=0.95
                ),
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        deid_result = deidentify(original_text, method="mask", keep_mapping=True)
        # Single entity should NOT have a counter suffix
        assert deid_result.deidentified_text == "Patient [NAME]"
        assert "[NAME_2]" not in deid_result.deidentified_text

        reidentified = reidentify(deid_result.deidentified_text, deid_result.mapping)
        assert reidentified == original_text
