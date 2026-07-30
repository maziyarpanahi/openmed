"""Tests for code-system version pinning and provenance stamping."""

from __future__ import annotations

import json
from copy import deepcopy

from openmed.clinical.exporters import stamp_coding_provenance
from openmed.clinical.exporters.code_provenance import (
    CODE_SYSTEM_VERSION_SOURCE_EXTENSION_URL,
)
from openmed.clinical.exporters.codeable_concept_simple import coding


class TestStampCodingProvenance:
    def test_loinc_short_pin_sets_version_and_source_label(self):
        base = coding("loinc", "2160-0", "Creatinine")

        result = stamp_coding_provenance(
            base,
            {"loinc": "caller-loinc-release"},
            source_label="local terminology manifest",
        )

        assert result["version"] == "caller-loinc-release"
        assert result["extension"] == [
            {
                "url": CODE_SYSTEM_VERSION_SOURCE_EXTENSION_URL,
                "valueString": "local terminology manifest",
            }
        ]
        unstamped_fields = {
            key: value
            for key, value in result.items()
            if key not in {"version", "extension"}
        }
        assert unstamped_fields == base

    def test_canonical_uri_pin_sets_version(self):
        base = coding("loinc", "2160-0", "Creatinine")

        result = stamp_coding_provenance(
            base,
            {"http://loinc.org": "caller-uri-release"},
        )

        assert result["version"] == "caller-uri-release"
        assert "extension" not in result

    def test_empty_pin_map_does_not_bundle_known_system_version(self):
        base = coding("loinc", "2160-0", "Creatinine")

        result = stamp_coding_provenance(base, {})

        assert result == base
        assert "version" not in result

    def test_unpinned_system_does_not_invent_version_or_source(self):
        base = coding("rxnorm", "1049502", "12 HR Oxycodone")

        result = stamp_coding_provenance(
            base,
            {"loinc": "caller-loinc-release"},
            source_label="local terminology manifest",
        )

        assert result == base
        assert result is not base
        assert "version" not in result
        assert "extension" not in result

    def test_input_is_never_mutated(self):
        base = {
            "system": "http://loinc.org",
            "code": "2160-0",
            "display": "Creatinine",
            "extension": [
                {
                    "url": "http://example.org/fhir/StructureDefinition/source",
                    "valueString": "existing marker",
                }
            ],
        }
        snapshot = deepcopy(base)

        stamp_coding_provenance(
            base,
            {"loinc": "caller-loinc-release"},
            source_label="local terminology manifest",
        )

        assert base == snapshot

    def test_output_is_deterministic(self):
        base = {
            "system": "http://loinc.org",
            "code": "2160-0",
            "display": "Creatinine",
            "extension": [
                {
                    "url": "http://example.org/fhir/StructureDefinition/source",
                    "valueString": "existing marker",
                }
            ],
        }
        pins = {"loinc": "caller-loinc-release"}

        result_a = stamp_coding_provenance(
            base,
            pins,
            source_label="local terminology manifest",
        )
        result_b = stamp_coding_provenance(
            base,
            pins,
            source_label="local terminology manifest",
        )

        assert result_a == result_b
        assert json.dumps(result_a, sort_keys=True) == json.dumps(
            result_b,
            sort_keys=True,
        )
        assert result_a["extension"] == [
            {
                "url": "http://example.org/fhir/StructureDefinition/source",
                "valueString": "existing marker",
            },
            {
                "url": CODE_SYSTEM_VERSION_SOURCE_EXTENSION_URL,
                "valueString": "local terminology manifest",
            },
        ]
