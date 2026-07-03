"""Tests for normalized concepts flowing into FHIR CodeableConcept."""

from __future__ import annotations

from openmed.clinical.exporters import (
    CONCEPT_NORMALIZATION_PROVENANCE_EXTENSION_URL,
    check_codeable_concept,
    codeable_concept_from_ranked_candidates,
)
from openmed.clinical.normalization import (
    ConceptNormalizer,
    SyntheticTerminologyBackend,
)


def test_ranked_candidates_export_to_codeable_concept_with_offsets():
    ranked = ConceptNormalizer(SyntheticTerminologyBackend()).normalize(
        "Aster pyrexia",
        start=4,
        end=17,
    )

    concept = codeable_concept_from_ranked_candidates(
        ranked,
        text="Aster fever",
        max_codings=1,
    )

    coding = concept["coding"][0]
    assert coding["system"] == ranked[0].concept.system_uri
    assert coding["code"] == "SYN-COND-001"
    assert coding["display"] == "Aster fever"
    assert coding["version"] == "2026.06-synthetic"

    provenance = coding["extension"][0]
    assert provenance["url"] == CONCEPT_NORMALIZATION_PROVENANCE_EXTENSION_URL
    extension_values = {
        extension["url"]: extension for extension in provenance["extension"]
    }
    assert extension_values["mentionStart"]["valueInteger"] == 4
    assert extension_values["mentionEnd"]["valueInteger"] == 17
    assert extension_values["confidence"]["valueDecimal"] >= 0.8
    assert extension_values["backendVersion"]["valueString"] == "2026.06"
    assert check_codeable_concept(concept) == []
