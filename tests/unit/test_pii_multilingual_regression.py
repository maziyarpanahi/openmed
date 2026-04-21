"""Multilingual PII regression tests with golden inputs.

Each language gets deterministic mock model output to verify:
- Expected entity types are detected
- Span text matches expected substrings
- Confidence is above per-language thresholds
- Smart merging produces correct span boundaries
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from openmed.core.pii import extract_pii
from openmed.processing.outputs import EntityPrediction, PredictionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(text, entities):
    """Build a PredictionResult from EntityPrediction list."""
    return PredictionResult(
        text=text,
        entities=entities,
        model_name="mock",
        timestamp="2026-01-01T00:00:00",
    )


def _ent(text, label, start, end, confidence=0.92):
    return EntityPrediction(
        text=text, label=label, start=start, end=end, confidence=confidence,
    )


def _ent_in(full_text, text, label, confidence=0.92):
    start = full_text.index(text)
    return _ent(text, label, start, start + len(text), confidence)


# ---------------------------------------------------------------------------
# English (en)
# ---------------------------------------------------------------------------


class TestEnglishRegression:

    CLINICAL_NOTE = "Patient John Doe, DOB 1990-05-15, phone 555-123-4567, SSN 123-45-6789."

    @patch("openmed.analyze_text")
    def test_en_name_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("John Doe", "first_name", 8, 16, 0.95),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="en")
        names = [e for e in result.entities if "name" in e.label.lower()]
        assert len(names) >= 1
        assert "John" in names[0].text

    @patch("openmed.analyze_text")
    def test_en_date_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("1990-05-15", "date_of_birth", 22, 32, 0.93),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="en")
        dates = [e for e in result.entities if "date" in e.label.lower()]
        assert len(dates) >= 1
        assert "1990" in dates[0].text

    @patch("openmed.analyze_text")
    def test_en_phone_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("555-123-4567", "phone_number", 40, 52, 0.91),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="en")
        phones = [e for e in result.entities if "phone" in e.label.lower()]
        assert len(phones) >= 1
        assert "555" in phones[0].text

    @patch("openmed.analyze_text")
    def test_en_ssn_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("123-45-6789", "ssn", 58, 69, 0.94),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="en")
        ssns = [e for e in result.entities if "ssn" in e.label.lower()]
        assert len(ssns) >= 1

    @patch("openmed.analyze_text")
    def test_en_smart_merging_spans(self, mock_analyze):
        text = "Patient John Doe visited the clinic"
        mock_analyze.return_value = _make_result(text, [
            _ent("John", "first_name", 8, 12, 0.93),
            _ent("Doe", "last_name", 13, 16, 0.91),
        ])
        result = extract_pii(text, use_smart_merging=True, lang="en")
        # After merging, fragments should be combined or kept with valid spans
        for e in result.entities:
            if e.start is not None and e.end is not None:
                assert e.start < e.end
                assert e.start >= 0
                assert e.end <= len(text)


# ---------------------------------------------------------------------------
# French (fr)
# ---------------------------------------------------------------------------


class TestFrenchRegression:

    CLINICAL_NOTE = "Patient Jean Dupont, né le 15/05/1990, téléphone 01 23 45 67 89, NIR 1 90 05 75 108 042 36."

    @patch("openmed.analyze_text")
    def test_fr_name_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("Jean Dupont", "first_name", 8, 19, 0.90),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="fr")
        names = [e for e in result.entities if "name" in e.label.lower()]
        assert len(names) >= 1
        assert "Jean" in names[0].text

    @patch("openmed.analyze_text")
    def test_fr_date_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("15/05/1990", "date_of_birth", 27, 37, 0.88),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="fr")
        dates = [e for e in result.entities if "date" in e.label.lower()]
        assert len(dates) >= 1
        assert "1990" in dates[0].text

    @patch("openmed.analyze_text")
    def test_fr_phone_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("01 23 45 67 89", "phone_number", 50, 64, 0.87),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="fr")
        phones = [e for e in result.entities if "phone" in e.label.lower()]
        assert len(phones) >= 1

    @patch("openmed.analyze_text")
    def test_fr_national_id_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("1 90 05 75 108 042 36", "national_id", 70, 91, 0.85),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="fr")
        ids = [e for e in result.entities if "national_id" in e.label.lower() or "nir" in e.label.lower()]
        assert len(ids) >= 1


# ---------------------------------------------------------------------------
# German (de)
# ---------------------------------------------------------------------------


class TestGermanRegression:

    CLINICAL_NOTE = "Patient Hans Müller, geb. 15.05.1990, Tel. 030 12345678, Steuer-ID 12 345 678 901."

    @patch("openmed.analyze_text")
    def test_de_name_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("Hans Müller", "first_name", 8, 19, 0.91),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="de")
        names = [e for e in result.entities if "name" in e.label.lower()]
        assert len(names) >= 1
        assert "Hans" in names[0].text

    @patch("openmed.analyze_text")
    def test_de_date_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("15.05.1990", "date_of_birth", 26, 36, 0.89),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="de")
        dates = [e for e in result.entities if "date" in e.label.lower()]
        assert len(dates) >= 1

    @patch("openmed.analyze_text")
    def test_de_phone_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("030 12345678", "phone_number", 43, 55, 0.88),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="de")
        phones = [e for e in result.entities if "phone" in e.label.lower()]
        assert len(phones) >= 1

    @patch("openmed.analyze_text")
    def test_de_smart_merging_boundaries(self, mock_analyze):
        text = "Patient Hans Müller besuchte die Klinik"
        mock_analyze.return_value = _make_result(text, [
            _ent("Hans", "first_name", 8, 12, 0.90),
            _ent("Müller", "last_name", 13, 19, 0.88),
        ])
        result = extract_pii(text, use_smart_merging=True, lang="de")
        for e in result.entities:
            if e.start is not None and e.end is not None:
                assert e.start < e.end
                assert e.end <= len(text)


# ---------------------------------------------------------------------------
# Italian (it)
# ---------------------------------------------------------------------------


class TestItalianRegression:

    CLINICAL_NOTE = "Paziente Marco Rossi, nato il 15/05/1990, tel. 06 1234567, CF RSSMRC90E15H501Z."

    @patch("openmed.analyze_text")
    def test_it_name_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("Marco Rossi", "first_name", 10, 21, 0.90),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="it")
        names = [e for e in result.entities if "name" in e.label.lower()]
        assert len(names) >= 1
        assert "Marco" in names[0].text

    @patch("openmed.analyze_text")
    def test_it_date_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("15/05/1990", "date_of_birth", 30, 40, 0.88),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="it")
        dates = [e for e in result.entities if "date" in e.label.lower()]
        assert len(dates) >= 1

    @patch("openmed.analyze_text")
    def test_it_phone_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("06 1234567", "phone_number", 48, 58, 0.87),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="it")
        phones = [e for e in result.entities if "phone" in e.label.lower()]
        assert len(phones) >= 1


# ---------------------------------------------------------------------------
# Spanish (es)
# ---------------------------------------------------------------------------


class TestSpanishRegression:

    CLINICAL_NOTE = "Paciente María García, nacida el 15/05/1990, teléfono 612 345 678, DNI 12345678Z."

    @patch("openmed.analyze_text")
    def test_es_name_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("María García", "first_name", 10, 22, 0.90),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="es")
        names = [e for e in result.entities if "name" in e.label.lower()]
        assert len(names) >= 1

    @patch("openmed.analyze_text")
    def test_es_date_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("15/05/1990", "date_of_birth", 33, 43, 0.88),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="es")
        dates = [e for e in result.entities if "date" in e.label.lower()]
        assert len(dates) >= 1

    @patch("openmed.analyze_text")
    def test_es_phone_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("612 345 678", "phone_number", 54, 65, 0.87),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="es")
        phones = [e for e in result.entities if "phone" in e.label.lower()]
        assert len(phones) >= 1

    @patch("openmed.analyze_text")
    def test_es_confidence_threshold(self, mock_analyze):
        text = "Paciente Carlos López"
        mock_analyze.return_value = _make_result(text, [
            _ent("Carlos López", "first_name", 10, 22, 0.85),
        ])
        result = extract_pii(text, use_smart_merging=False, lang="es", confidence_threshold=0.5)
        assert len(result.entities) >= 1
        assert all(e.confidence >= 0.5 for e in result.entities)


# ---------------------------------------------------------------------------
# Portuguese (pt)
# ---------------------------------------------------------------------------


class TestPortugueseRegression:

    CLINICAL_NOTE = (
        "Paciente Pedro Almeida, nascido em 15/03/1985, CPF 123.456.789-09, "
        "email pedro@hospital.pt, telefone +351 912 345 678, "
        "endere\u00e7o Rua das Flores 25, 1200-195 Lisboa."
    )

    @patch("openmed.analyze_text")
    def test_pt_name_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent_in(self.CLINICAL_NOTE, "Pedro Almeida", "FIRSTNAME", 0.90),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="pt")
        names = [e for e in result.entities if "name" in e.label.lower()]
        assert len(names) >= 1
        assert "Pedro" in names[0].text

    @patch("openmed.analyze_text")
    def test_pt_date_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent_in(self.CLINICAL_NOTE, "15/03/1985", "DATEOFBIRTH", 0.88),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="pt")
        dates = [e for e in result.entities if "date" in e.label.lower()]
        assert len(dates) >= 1

    @patch("openmed.analyze_text")
    def test_pt_cpf_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent_in(self.CLINICAL_NOTE, "123.456.789-09", "cpf", 0.89),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="pt")
        ids = [e for e in result.entities if e.label.lower() in {"cpf", "national_id", "ssn"}]
        assert len(ids) >= 1

    @patch("openmed.analyze_text")
    def test_pt_email_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent_in(self.CLINICAL_NOTE, "pedro@hospital.pt", "EMAIL", 0.91),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="pt")
        emails = [e for e in result.entities if "email" in e.label.lower()]
        assert len(emails) >= 1

    @patch("openmed.analyze_text")
    def test_pt_phone_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent_in(self.CLINICAL_NOTE, "+351 912 345 678", "PHONE", 0.87),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="pt")
        phones = [e for e in result.entities if "phone" in e.label.lower()]
        assert len(phones) >= 1

    @patch("openmed.analyze_text")
    def test_pt_address_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent_in(self.CLINICAL_NOTE, "Rua das Flores 25", "STREET", 0.86),
            _ent_in(self.CLINICAL_NOTE, "1200-195", "ZIPCODE", 0.84),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="pt")
        labels = {e.label.lower() for e in result.entities}
        assert "street" in labels
        assert "zipcode" in labels


# ---------------------------------------------------------------------------
# Dutch (nl)
# ---------------------------------------------------------------------------


class TestDutchRegression:

    CLINICAL_NOTE = "Patiënt Jan de Vries, geb. 15-05-1990, tel. 020 1234567, BSN 123456789."

    @patch("openmed.analyze_text")
    def test_nl_name_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("Jan de Vries", "first_name", 9, 21, 0.89),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="nl")
        names = [e for e in result.entities if "name" in e.label.lower()]
        assert len(names) >= 1
        assert "Jan" in names[0].text

    @patch("openmed.analyze_text")
    def test_nl_date_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("15-05-1990", "date_of_birth", 27, 37, 0.87),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="nl")
        dates = [e for e in result.entities if "date" in e.label.lower()]
        assert len(dates) >= 1

    @patch("openmed.analyze_text")
    def test_nl_phone_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("020 1234567", "phone_number", 44, 55, 0.86),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="nl")
        phones = [e for e in result.entities if "phone" in e.label.lower()]
        assert len(phones) >= 1


# ---------------------------------------------------------------------------
# Hindi (hi)
# ---------------------------------------------------------------------------


class TestHindiRegression:

    CLINICAL_NOTE = "रोगी राज कुमार, जन्म 15/05/1990, फोन 9876543210, आधार 1234 5678 9012."

    @patch("openmed.analyze_text")
    def test_hi_name_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("राज कुमार", "first_name", 5, 14, 0.88),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="hi")
        names = [e for e in result.entities if "name" in e.label.lower()]
        assert len(names) >= 1

    @patch("openmed.analyze_text")
    def test_hi_date_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("15/05/1990", "date_of_birth", 22, 32, 0.86),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="hi")
        dates = [e for e in result.entities if "date" in e.label.lower()]
        assert len(dates) >= 1

    @patch("openmed.analyze_text")
    def test_hi_phone_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("9876543210", "phone_number", 38, 48, 0.85),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="hi")
        phones = [e for e in result.entities if "phone" in e.label.lower()]
        assert len(phones) >= 1

    @patch("openmed.analyze_text")
    def test_hi_smart_merging_boundaries(self, mock_analyze):
        text = "रोगी राज कुमार ने दौरा किया"
        mock_analyze.return_value = _make_result(text, [
            _ent("राज", "first_name", 5, 8, 0.87),
            _ent("कुमार", "last_name", 9, 14, 0.85),
        ])
        result = extract_pii(text, use_smart_merging=True, lang="hi")
        for e in result.entities:
            if e.start is not None and e.end is not None:
                assert e.start < e.end
                assert e.end <= len(text)

    @patch("openmed.analyze_text")
    def test_hi_aadhaar_detection(self, mock_analyze):
        text = "रोगी का आधार 2345 6789 0123 है"
        mock_analyze.return_value = _make_result(text, [
            _ent("2345 6789 0123", "national_id", 14, 28, 0.85),
        ])
        result = extract_pii(text, use_smart_merging=False, lang="hi")
        ids = [e for e in result.entities if "national_id" in e.label.lower()]
        assert len(ids) >= 1


# ---------------------------------------------------------------------------
# Telugu (te)
# ---------------------------------------------------------------------------


class TestTeluguRegression:

    CLINICAL_NOTE = "రోగి రాజు కుమార్, పుట్టిన తేదీ 15/05/1990, ఫోన్ 9876543210."

    @patch("openmed.analyze_text")
    def test_te_name_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("రాజు కుమార్", "first_name", 5, 16, 0.86),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="te")
        names = [e for e in result.entities if "name" in e.label.lower()]
        assert len(names) >= 1

    @patch("openmed.analyze_text")
    def test_te_date_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("15/05/1990", "date_of_birth", 31, 41, 0.84),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="te")
        dates = [e for e in result.entities if "date" in e.label.lower()]
        assert len(dates) >= 1

    @patch("openmed.analyze_text")
    def test_te_phone_detection(self, mock_analyze):
        mock_analyze.return_value = _make_result(self.CLINICAL_NOTE, [
            _ent("9876543210", "phone_number", 47, 57, 0.83),
        ])
        result = extract_pii(self.CLINICAL_NOTE, use_smart_merging=False, lang="te")
        phones = [e for e in result.entities if "phone" in e.label.lower()]
        assert len(phones) >= 1

    @patch("openmed.analyze_text")
    def test_te_confidence_above_threshold(self, mock_analyze):
        text = "రోగి రాజు"
        mock_analyze.return_value = _make_result(text, [
            _ent("రాజు", "first_name", 5, 9, 0.80),
        ])
        result = extract_pii(text, use_smart_merging=False, lang="te", confidence_threshold=0.5)
        assert len(result.entities) >= 1
        assert all(e.confidence >= 0.5 for e in result.entities)

    @patch("openmed.analyze_text")
    def test_te_aadhaar_detection(self, mock_analyze):
        text = "రోగి ఆధార్ 2345 6789 0123"
        mock_analyze.return_value = _make_result(text, [
            _ent("2345 6789 0123", "national_id", 12, 26, 0.83),
        ])
        result = extract_pii(text, use_smart_merging=False, lang="te")
        ids = [e for e in result.entities if "national_id" in e.label.lower()]
        assert len(ids) >= 1
