"""Tests for the standalone FHIR R4 CodeableConcept builder (OM-361)."""

import pytest

from openmed.clinical.exporters.codeable_concept_simple import (
    codeable_concept,
    coding,
    system_uri,
)


class TestSystemUri:
    def test_rxnorm_returns_canonical_uri(self):
        assert system_uri("rxnorm") == "http://www.nlm.nih.gov/research/umls/rxnorm"

    def test_loinc_returns_canonical_uri(self):
        assert system_uri("loinc") == "http://loinc.org"

    def test_snomed_returns_canonical_uri(self):
        assert system_uri("snomed") == "http://snomed.info/sct"

    def test_icd10cm_returns_canonical_uri(self):
        assert system_uri("icd-10-cm") == "http://hl7.org/fhir/sid/icd-10-cm"

    def test_hpo_returns_canonical_uri(self):
        assert system_uri("hpo") == "http://purl.obolibrary.org/obo/hp.owl"

    def test_mesh_returns_canonical_uri(self):
        assert system_uri("mesh") == "https://meshb.nlm.nih.gov"

    def test_lookup_is_case_insensitive(self):
        assert system_uri("RxNorm") == system_uri("rxnorm")
        assert system_uri("LOINC") == system_uri("loinc")

    def test_already_canonical_http_uri_passes_through(self):
        uri = "http://loinc.org"
        assert system_uri(uri) == uri

    def test_already_canonical_https_uri_passes_through(self):
        uri = "https://meshb.nlm.nih.gov"
        assert system_uri(uri) == uri

    def test_arbitrary_canonical_uri_passes_through_unchanged(self):
        uri = "http://example.com/my-custom-system"
        assert system_uri(uri) == uri

    def test_unknown_vocabulary_id_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown vocabulary id"):
            system_uri("cpt")

    def test_unknown_vocabulary_id_error_names_the_bad_id(self):
        with pytest.raises(ValueError, match="'omop'"):
            system_uri("omop")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            system_uri("")


class TestCoding:
    def test_rxnorm_coding_has_canonical_system_and_code(self):
        result = coding("rxnorm", "1049502")
        assert result["system"] == "http://www.nlm.nih.gov/research/umls/rxnorm"
        assert result["code"] == "1049502"

    def test_coding_with_display_includes_display(self):
        result = coding("rxnorm", "1049502", "12 HR Oxycodone")
        assert result["display"] == "12 HR Oxycodone"

    def test_coding_without_display_omits_display_key(self):
        result = coding("loinc", "2160-0")
        assert "display" not in result

    def test_coding_shape_is_system_code_display(self):
        result = coding("snomed", "44054006", "Type 2 diabetes mellitus")
        assert set(result.keys()) == {"system", "code", "display"}

    def test_coding_without_display_shape_is_system_code(self):
        result = coding("snomed", "44054006")
        assert set(result.keys()) == {"system", "code"}

    def test_coding_accepts_already_canonical_uri(self):
        result = coding("http://loinc.org", "2160-0")
        assert result["system"] == "http://loinc.org"

    def test_coding_unknown_system_raises(self):
        with pytest.raises(ValueError):
            coding("cpt", "99213")


class TestCodeableConcept:
    def test_single_coding_produces_correct_shape(self):
        c = coding("loinc", "2160-0", "Creatinine")
        result = codeable_concept([c])
        assert result["coding"] == [c]
        assert "text" not in result

    def test_text_is_included_when_provided(self):
        c = coding("loinc", "2160-0", "Creatinine")
        result = codeable_concept([c], text="Creatinine [Mass/volume] in Serum")
        assert result["text"] == "Creatinine [Mass/volume] in Serum"

    def test_text_omitted_when_none(self):
        result = codeable_concept([coding("loinc", "2160-0")])
        assert "text" not in result

    def test_resource_type_key_is_not_present(self):
        # CodeableConcept is a data type, not a resource — no resourceType key.
        result = codeable_concept([coding("loinc", "2160-0")])
        assert "resourceType" not in result

    def test_empty_codings_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one coding"):
            codeable_concept([])

    def test_multiple_codings_ordered_by_default_priority(self):
        # Pass codings in reverse priority order; expect them sorted correctly.
        c_rxnorm = coding("rxnorm", "1049502")
        c_snomed = coding("snomed", "372687004")
        c_loinc = coding("loinc", "2160-0")

        result = codeable_concept([c_rxnorm, c_loinc, c_snomed])
        systems = [c["system"] for c in result["coding"]]

        # Default priority: snomed first, then loinc, then rxnorm.
        assert systems.index("http://snomed.info/sct") < systems.index(
            "http://loinc.org"
        )
        assert systems.index("http://loinc.org") < systems.index(
            "http://www.nlm.nih.gov/research/umls/rxnorm"
        )

    def test_ordering_is_deterministic_regardless_of_input_order(self):
        c_rxnorm = coding("rxnorm", "1049502")
        c_snomed = coding("snomed", "372687004")

        result_a = codeable_concept([c_rxnorm, c_snomed])
        result_b = codeable_concept([c_snomed, c_rxnorm])

        assert result_a["coding"] == result_b["coding"]

    def test_custom_system_priority_is_respected(self):
        c_snomed = coding("snomed", "372687004")
        c_loinc = coding("loinc", "2160-0")

        # Flip the priority so loinc sorts before snomed.
        result = codeable_concept(
            [c_snomed, c_loinc],
            system_priority=("http://loinc.org", "http://snomed.info/sct"),
        )
        systems = [c["system"] for c in result["coding"]]
        assert systems[0] == "http://loinc.org"
        assert systems[1] == "http://snomed.info/sct"

    def test_unknown_system_sorts_last_alphabetically(self):
        c_known = coding("loinc", "2160-0")
        c_unknown_a = {"system": "http://unknown-a.example.com", "code": "X1"}
        c_unknown_b = {"system": "http://unknown-b.example.com", "code": "X2"}

        result = codeable_concept([c_unknown_b, c_unknown_a, c_known])
        systems = [c["system"] for c in result["coding"]]

        # Known system first, then unknowns sorted alphabetically.
        assert systems[0] == "http://loinc.org"
        assert systems[1] == "http://unknown-a.example.com"
        assert systems[2] == "http://unknown-b.example.com"

    def test_input_list_is_not_mutated(self):
        c_rxnorm = coding("rxnorm", "1049502")
        c_snomed = coding("snomed", "372687004")
        original = [c_rxnorm, c_snomed]
        snapshot = list(original)

        codeable_concept(original)

        assert original == snapshot
