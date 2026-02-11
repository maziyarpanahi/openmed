"""Unit tests for PII extraction and de-identification module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from openmed.core.pii import (
    extract_pii,
    deidentify,
    reidentify,
    PIIEntity,
    DeidentificationResult,
    _redact_entity,
    _generate_fake_pii,
    _shift_date,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pii_entities():
    """Mock PII entities for testing."""
    return [
        EntityPrediction(text="John Doe", label="NAME", start=8, end=16, confidence=0.95),
        EntityPrediction(text="555-1234", label="PHONE", start=20, end=28, confidence=0.90),
        EntityPrediction(text="john@email.com", label="EMAIL", start=32, end=46, confidence=0.92),
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
            text="Test text", entities=[], model_name="test_model", timestamp=datetime.now().isoformat()
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
            group_entities=True,
        )

    @patch("openmed.analyze_text")
    def test_extract_pii_returns_analysis_result(self, mock_analyze, mock_analyze_result):
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
            text="Test", entities=[], model_name="default", timestamp=datetime.now().isoformat()
        )

        extract_pii("Test")

        call_args = mock_analyze.call_args
        assert call_args[1]["model_name"] == "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"


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
                EntityPrediction(text="John Doe", label="NAME", start=8, end=16, confidence=0.95)
            ],
            model_name="test", timestamp=datetime.now().isoformat(),
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
                EntityPrediction(text="555-1234", label="PHONE", start=5, end=13, confidence=0.90)
            ],
            model_name="test", timestamp=datetime.now().isoformat(),
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
            model_name="test", timestamp=datetime.now().isoformat(),
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
                EntityPrediction(text="John Doe", label="NAME", start=8, end=16, confidence=0.95)
            ],
            model_name="test", timestamp=datetime.now().isoformat(),
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
                EntityPrediction(text="John Doe", label="NAME", start=0, end=8, confidence=0.95),
                EntityPrediction(text="555-1234", label="PHONE", start=12, end=20, confidence=0.90),
            ],
            model_name="test", timestamp=datetime.now().isoformat(),
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
                EntityPrediction(text="John Doe", label="NAME", start=8, end=16, confidence=0.95)
            ],
            model_name="test", timestamp=datetime.now().isoformat(),
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
                EntityPrediction(text="John Doe", label="NAME", start=8, end=16, confidence=0.95)
            ],
            model_name="test", timestamp=datetime.now().isoformat(),
        )

        result = deidentify("Patient John Doe", method="mask", keep_mapping=False)

        assert result.mapping is None

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_empty_text(self, mock_extract):
        """Test deidentify handles empty text."""
        mock_extract.return_value = PredictionResult(
            text="", entities=[], model_name="test", timestamp=datetime.now().isoformat()
        )

        result = deidentify("", method="mask")

        assert result.deidentified_text == ""
        assert len(result.pii_entities) == 0

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_no_entities(self, mock_extract):
        """Test deidentify handles text with no PII."""
        mock_extract.return_value = PredictionResult(
            text="No PII here", entities=[], model_name="test", timestamp=datetime.now().isoformat()
        )

        result = deidentify("No PII here", method="mask")

        assert result.deidentified_text == "No PII here"
        assert len(result.pii_entities) == 0

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_confidence_threshold(self, mock_extract):
        """Test deidentify uses custom confidence threshold."""
        mock_extract.return_value = PredictionResult(
            text="Test", entities=[], model_name="test", timestamp=datetime.now().isoformat()
        )

        deidentify("Test", confidence_threshold=0.8)

        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args[0][2] == 0.8  # confidence_threshold parameter


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
            extract_pii("test", lang="es")

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
                    text="Marie Dupont", label="NAME", start=8, end=20, confidence=0.95,
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        deidentify("Patient Marie Dupont", method="mask", lang="fr")

        call_args = mock_extract.call_args
        assert call_args[1]["lang"] == "fr"

    @patch("openmed.core.pii.extract_pii")
    def test_deidentify_replace_uses_lang_fake_data(self, mock_extract):
        """Test replace method uses language-appropriate fake data."""
        mock_extract.return_value = PredictionResult(
            text="Patient Marie Dupont",
            entities=[
                EntityPrediction(
                    text="Marie Dupont", label="NAME", start=8, end=20, confidence=0.95,
                )
            ],
            model_name="test",
            timestamp=datetime.now().isoformat(),
        )

        result = deidentify("Patient Marie Dupont", method="replace", lang="fr")

        # Should use French fake names
        from openmed.core.pii_i18n import LANGUAGE_FAKE_DATA
        french_names = LANGUAGE_FAKE_DATA["fr"]["NAME"]
        assert result.pii_entities[0].redacted_text in french_names

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
                EntityPrediction(text="John Doe", label="NAME", start=8, end=16, confidence=0.95),
                EntityPrediction(text="555-1234", label="PHONE", start=20, end=28, confidence=0.90),
            ],
            model_name="test", timestamp=datetime.now().isoformat(),
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
                EntityPrediction(text="John Doe", label="NAME", start=0, end=8, confidence=0.95)
            ],
            model_name="test", timestamp=datetime.now().isoformat(),
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
