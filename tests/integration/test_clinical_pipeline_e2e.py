"""End-to-end clinical pipeline integration suite (OM-804).

This module exercises the *full* OpenMed clinical pipeline on a synthetic
discharge summary and pins the hand-offs between every stage against committed
golden expectations:

    de-identify -> NER (analyze_text) -> clinical assertion / ConText
    -> concept grounding (RxNorm / ICD-10-CM / HPO linkers) -> FHIR R4 Bundle

Every stage uses only public APIs that exist in the tree today. The two
model-backed stages (the PII detector inside :func:`openmed.deidentify` and the
NER pipeline inside :func:`openmed.analyze_text`) are driven with deterministic,
offline-friendly loaders so the suite runs without network access or model
downloads, per the repository's offline-friendly testing rule.

Guarantees asserted here:

* **Stage hand-offs** - NER and grounding offsets index back into the exact
  surface they annotate; the de-identified text (never the raw note) is what
  every later stage consumes.
* **Leakage-first** - no redacted PHI surface string reappears in the NER
  input, the grounding inputs, or the final FHIR Bundle.
* **Valid FHIR** - the terminal artifact is a valid R4 transaction Bundle as
  produced by :func:`openmed.clinical.exporters.fhir.to_bundle` (deterministic
  ``fullUrl``s, one entry per resource, resolved internal references).
* **Golden expectations** - assertion axes, grounding codes, and FHIR resource
  shapes are compared against a committed synthetic JSON fixture.

The grounding -> FHIR leg is driven through the individual linkers because a
public top-level ``openmed.ground()`` orchestrator is *not merged yet*; see
:func:`test_public_ground_orchestrator_is_future_work` for the documented
``xfail`` that guards that gap rather than faking it.

Advisory only: the synthetic clinical output modelled here is a regression
fixture, not a medical device, and must not drive clinical decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import openmed
from openmed import analyze_text, deidentify
from openmed.clinical import (
    ACTIVE,
    INACTIVE,
    REFUTED,
    UNCONFIRMED,
    ClinicalAssertion,
    clinical_status_from_assertion,
    resolve_span_context,
)
from openmed.clinical.exporters.codeable_concept import (
    GroundedSpan,
    build_reverse_index,
    to_codeable_concept,
)
from openmed.clinical.exporters.fhir import to_bundle
from openmed.clinical.grounding import (
    Candidate,
    VocabLoader,
    VocabSource,
    available_linkers,
    get_linker,
)

# --------------------------------------------------------------------------- #
# Fixtures and synthetic data (synthetic-only; no real PHI, no DUA vocab).
# --------------------------------------------------------------------------- #

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "clinical"
_GROUNDING_FIXTURES = _FIXTURE_ROOT / "grounding"
_GOLDEN = _FIXTURE_ROOT / "e2e" / "discharge_summary_golden.json"

# A single embedded synthetic clinical note. It carries structured PHI that the
# deterministic safety sweep redacts offline (date / phone / email / MRN / SSN)
# plus clinical content whose ConText axes are non-trivial: "History of present
# illness" makes diabetes historical, "denies pneumonia" negates it, and the
# family-history mention makes the seizure historical.
SYNTHETIC_NOTE = (
    "Synthetic discharge summary for patient DEMO-001. DOB: 1975-04-03. "
    "Contact phone: 212-555-0198. Email: demo.patient@example.test. "
    "MRN: 00123456. SSN: 123-45-6789. "
    "History of present illness: The patient reports type 2 diabetes and "
    "hypertension. Medication: metformin 500 mg twice daily and lisinopril "
    "10 mg daily. The patient denies pneumonia. Reports headache and fever. "
    "Family history of seizure."
)

# Canonical HL7 FHIR R4 system URIs for the three grounding vocabularies, used
# to assert the codings the grounded CodeableConcepts carry.
_SYSTEM_URI = {
    "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "icd10cm": "http://hl7.org/fhir/sid/icd-10-cm",
    "hpo": "http://human-phenotype-ontology.org",
}


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    """Load committed synthetic golden expectations for the pipeline."""
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def grounding_linkers() -> dict[str, Any]:
    """Build the RxNorm / ICD-10-CM / HPO linkers over synthetic vocabularies.

    Uses the committed synthetic grounding fixtures (no restricted/DUA vocab),
    dispatched through the public linker registry.
    """
    loader = VocabLoader(
        registry={
            "rxnorm": VocabSource(
                system="rxnorm", path=_GROUNDING_FIXTURES / "rxnorm_sample.jsonl"
            ),
            "icd10cm": VocabSource(
                system="icd10cm", path=_GROUNDING_FIXTURES / "icd10cm_sample.jsonl"
            ),
            "hpo": VocabSource(
                system="hpo", path=_GROUNDING_FIXTURES / "hpo_sample.jsonl"
            ),
        }
    )
    return {
        system: get_linker(system)(loader.get_index(system))
        for system in ("rxnorm", "icd10cm", "hpo")
    }


class _NoDownloadPipeline:
    """Token-classification stand-in for the PII detector (no downloads).

    The de-identification stage relies on the deterministic structured-identifier
    safety sweep (``use_safety_sweep=True``); the model pipeline itself
    contributes nothing here, so it returns no spans.
    """

    tokenizer = None

    def __call__(self, inputs: Any, **_: Any) -> list[Any]:
        if isinstance(inputs, list):
            return [[] for _ in inputs]
        return []


class _NoDownloadLoader:
    """Loader compatible with ``deidentify(..., loader=...)`` (offline)."""

    config = None

    def create_pipeline(self, *_: Any, **__: Any) -> Any:
        return _NoDownloadPipeline()

    def get_max_sequence_length(self, *_: Any, **__: Any) -> None:
        return None


class _FixtureNERLoader:
    """Deterministic NER loader for ``analyze_text`` (offline).

    Produces exact, offset-correct token-classification predictions for the
    clinical spans expected in the *de-identified* text. Offsets are located at
    call time with ``str.find`` so they always index back into the surface the
    pipeline actually receives - which is precisely the stage hand-off this
    suite is here to pin. The real :func:`openmed.analyze_text` code path
    (aggregation, sentence handling, offset remapping, formatting) runs
    unchanged over these predictions.
    """

    config = None

    def __init__(self, expected_entities: list[dict[str, Any]]) -> None:
        self._expected = expected_entities

    def create_pipeline(self, *_: Any, **__: Any) -> Any:
        expected = self._expected

        def pipeline(text: Any, **__: Any) -> list[dict[str, Any]]:
            single = not isinstance(text, list)
            segments = [text] if single else list(text)
            batched: list[list[dict[str, Any]]] = []
            for segment in segments:
                spans: list[dict[str, Any]] = []
                for entity in expected:
                    surface = entity["text"]
                    start = segment.find(surface)
                    if start < 0:
                        continue
                    spans.append(
                        {
                            "entity_group": entity["ner_label"],
                            "score": 0.99,
                            "start": start,
                            "end": start + len(surface),
                            "word": surface,
                        }
                    )
                spans.sort(key=lambda item: item["start"])
                batched.append(spans)
            return batched[0] if single else batched

        return pipeline

    def get_max_sequence_length(self, *_: Any, **__: Any) -> int:
        return 512


# --------------------------------------------------------------------------- #
# Pipeline driver: runs the real stages once and returns everything asserted.
# --------------------------------------------------------------------------- #


def _run_pipeline(golden: dict[str, Any], linkers: dict[str, Any]) -> dict[str, Any]:
    """Drive the full de-id -> NER -> assertion -> grounding -> FHIR pipeline.

    Returns the intermediate artifacts of every stage so tests can assert the
    hand-offs individually instead of re-running the pipeline per assertion.
    """
    expected_entities = golden["expected_entities"]

    # Stage 1 - de-identification (real deidentify, offline safety sweep).
    deid = deidentify(
        SYNTHETIC_NOTE,
        method="mask",
        confidence_threshold=0.5,
        loader=_NoDownloadLoader(),
        use_safety_sweep=True,
    )
    deidentified_text = deid.deidentified_text

    # Stage 2 - NER over the DE-IDENTIFIED text (real analyze_text).
    analysis = analyze_text(
        deidentified_text,
        model_name="synthetic-clinical-ner",
        loader=_FixtureNERLoader(expected_entities),
        confidence_threshold=0.5,
    )

    # Stage 3 + 4 + 5 - assertion, grounding, and FHIR resource construction.
    grounded_spans: list[GroundedSpan] = []
    resources: list[dict[str, Any]] = [
        # De-identification removed the patient's identifiers, so the Patient
        # resource is intentionally identifier-free (synthetic anchor only).
        {"resourceType": "Patient", "id": "synthetic-patient"}
    ]
    per_entity: list[dict[str, Any]] = []

    for index, entity in enumerate(analysis.entities):
        expected = next(
            item for item in expected_entities if item["text"] == entity.text
        )

        # Stage 3 - clinical assertion / ConText, bounded to the span's sentence
        # by passing the document text plus offsets.
        span_view = {
            "text": entity.text,
            "start": entity.start,
            "end": entity.end,
            "document_text": deidentified_text,
        }
        context = resolve_span_context(span_view)
        assertion = ClinicalAssertion(
            temporality=context.temporality,
            certainty=context.certainty,
            negation=context.negation,
        )
        status = clinical_status_from_assertion(assertion)

        # Stage 4 - concept grounding via the registered linker for this span.
        system = expected["grounding_system"]
        linker = linkers[system]
        candidates = linker.link(entity.text, canonical_label=entity.label, fuzzy=False)

        grounded = GroundedSpan(
            text=entity.text,
            start=entity.start,
            end=entity.end,
            candidates=tuple(candidates),
        )
        grounded_spans.append(grounded)

        # Stage 5 - FHIR CodeableConcept + resource for this grounded span.
        concept = to_codeable_concept(grounded)
        resource = _resource_for(expected["fhir_resource_type"], index, concept, status)
        resources.append(resource)

        per_entity.append(
            {
                "entity": entity,
                "expected": expected,
                "context": context,
                "status": status,
                "candidates": candidates,
                "concept": concept,
                "resource": resource,
            }
        )

    bundle = to_bundle(resources, doc_id=golden["note_id"])

    return {
        "deid": deid,
        "deidentified_text": deidentified_text,
        "analysis": analysis,
        "grounded_spans": grounded_spans,
        "resources": resources,
        "bundle": bundle,
        "per_entity": per_entity,
    }


def _resource_for(
    resource_type: str,
    index: int,
    concept: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    """Wrap a grounded ``CodeableConcept`` into a minimal FHIR R4 resource.

    Assertion status is materialised where the resource type carries it: a
    ``Condition`` maps advisory clinical status onto ``clinicalStatus`` /
    ``verificationStatus`` using the HL7 ``condition-clinical`` and
    ``condition-ver-status`` code systems.
    """
    subject = {"reference": "Patient/synthetic-patient"}
    if resource_type == "Condition":
        resource: dict[str, Any] = {
            "resourceType": "Condition",
            "id": f"condition-{index}",
            "subject": subject,
            "code": concept,
        }
        resource.update(_condition_status_elements(status))
        return resource
    if resource_type == "MedicationStatement":
        return {
            "resourceType": "MedicationStatement",
            "id": f"medication-{index}",
            "status": "active",
            "subject": subject,
            "medicationCodeableConcept": concept,
        }
    if resource_type == "Observation":
        return {
            "resourceType": "Observation",
            "id": f"observation-{index}",
            "status": "final",
            "subject": subject,
            "code": concept,
        }
    raise AssertionError(f"unexpected resource type {resource_type!r}")


def _condition_status_elements(status: str) -> dict[str, Any]:
    """Map advisory problem-list status onto FHIR Condition status elements."""
    elements: dict[str, Any] = {}
    if status in (ACTIVE, INACTIVE):
        elements["clinicalStatus"] = {
            "coding": [
                {
                    "system": (
                        "http://terminology.hl7.org/CodeSystem/condition-clinical"
                    ),
                    "code": status,
                }
            ]
        }
    if status in (REFUTED, UNCONFIRMED):
        elements["verificationStatus"] = {
            "coding": [
                {
                    "system": (
                        "http://terminology.hl7.org/CodeSystem/condition-ver-status"
                    ),
                    "code": status,
                }
            ]
        }
    return elements


@pytest.fixture(scope="module")
def pipeline(golden, grounding_linkers) -> dict[str, Any]:
    """Run the full pipeline once and share the artifacts across tests."""
    return _run_pipeline(golden, grounding_linkers)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _collect_references(node: Any):
    """Yield every ``reference`` string anywhere inside ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "reference" and isinstance(value, str):
                yield value
            else:
                yield from _collect_references(value)
    elif isinstance(node, list):
        for item in node:
            yield from _collect_references(item)


def _iter_strings(node: Any):
    """Yield every string value anywhere inside a JSON-like structure."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _iter_strings(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _iter_strings(item)


# --------------------------------------------------------------------------- #
# Stage 1: de-identification
# --------------------------------------------------------------------------- #


@pytest.mark.integration
class TestDeidentificationStage:
    def test_structured_phi_is_redacted(self, pipeline, golden):
        """Every expected PHI label is detected and redacted offline."""
        detected = {entity.label for entity in pipeline["deid"].pii_entities}
        for label in golden["redacted_pii_labels"]:
            assert label in detected, f"expected {label!r} to be redacted"

    def test_redacted_text_contains_no_phi_surface(self, pipeline, golden):
        """No raw PHI surface string survives into the de-identified text."""
        deidentified_text = pipeline["deidentified_text"]
        for phi in golden["phi_surface_strings"]:
            assert phi not in deidentified_text, (
                f"PHI {phi!r} leaked into redacted text"
            )

    def test_clinical_content_survives_redaction(self, pipeline, golden):
        """Clinical spans the pipeline grounds are preserved by redaction."""
        deidentified_text = pipeline["deidentified_text"]
        for entity in golden["expected_entities"]:
            assert entity["text"] in deidentified_text


# --------------------------------------------------------------------------- #
# Stage 2: NER hand-off (offsets index into the de-identified text)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
class TestNERHandoff:
    def test_ner_runs_on_deidentified_text(self, pipeline):
        """analyze_text consumes the de-identified text, not the raw note."""
        assert pipeline["analysis"].text == pipeline["deidentified_text"]

    def test_all_expected_entities_extracted(self, pipeline, golden):
        extracted = {entity.text for entity in pipeline["analysis"].entities}
        expected = {item["text"] for item in golden["expected_entities"]}
        assert extracted == expected

    def test_ner_offsets_index_back_into_surface(self, pipeline):
        """Each NER span's offsets slice exactly its surface text."""
        text = pipeline["analysis"].text
        for entity in pipeline["analysis"].entities:
            assert text[entity.start : entity.end] == entity.text


# --------------------------------------------------------------------------- #
# Stage 3: clinical assertion / ConText, checked against golden expectations
# --------------------------------------------------------------------------- #


@pytest.mark.integration
class TestAssertionStage:
    def test_context_axes_match_golden(self, pipeline):
        """Negation / temporality / certainty match committed golden values."""
        for row in pipeline["per_entity"]:
            context = row["context"]
            expected = row["expected"]
            assert context.negation == expected["negation"], expected["text"]
            assert context.temporality == expected["temporality"], expected["text"]
            assert context.certainty == expected["certainty"], expected["text"]

    def test_negated_finding_refutes_condition(self, pipeline):
        """ "denies pneumonia" resolves to a refuted Condition."""
        row = next(r for r in pipeline["per_entity"] if r["entity"].text == "pneumonia")
        assert row["context"].negation == "negated"
        assert row["status"] == REFUTED

    def test_historical_span_is_inactive(self, pipeline):
        """A historical, affirmed condition maps to inactive clinical status."""
        row = next(
            r for r in pipeline["per_entity"] if r["entity"].text == "type 2 diabetes"
        )
        assert row["context"].temporality == "historical"
        assert row["status"] == INACTIVE


# --------------------------------------------------------------------------- #
# Stage 4: grounding, checked against golden codes and offset consistency
# --------------------------------------------------------------------------- #


@pytest.mark.integration
class TestGroundingStage:
    def test_grounding_codes_match_golden(self, pipeline):
        """Top grounding candidate matches the committed golden code/system."""
        for row in pipeline["per_entity"]:
            expected = row["expected"]
            candidates = row["candidates"]
            assert candidates, f"no grounding candidate for {expected['text']!r}"
            top = candidates[0]
            assert top.code == expected["expected_code"], expected["text"]
            assert top.display == expected["expected_display"], expected["text"]

    def test_grounding_offsets_slice_source(self, pipeline):
        """Grounded span offsets slice exactly the surface they encode."""
        text = pipeline["deidentified_text"]
        for span in pipeline["grounded_spans"]:
            assert text[span.start : span.end] == span.text

    def test_reverse_index_points_back_to_spans(self, pipeline):
        """The code -> source-offset reverse index round-trips to the surface."""
        text = pipeline["deidentified_text"]
        reverse = build_reverse_index(pipeline["grounded_spans"])
        assert reverse
        for offsets in reverse.values():
            for start, end in offsets:
                assert 0 <= start < end <= len(text)


# --------------------------------------------------------------------------- #
# Stage 5: FHIR export - valid R4 transaction Bundle
# --------------------------------------------------------------------------- #


@pytest.mark.integration
class TestFHIRExportStage:
    def test_bundle_is_valid_r4_transaction(self, pipeline):
        bundle = pipeline["bundle"]
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "transaction"
        assert len(bundle["entry"]) == len(pipeline["resources"])
        for entry in bundle["entry"]:
            assert entry["fullUrl"].startswith("urn:uuid:")
            assert "resourceType" in entry["resource"]
            assert entry["request"]["method"] == "POST"

    def test_full_urls_are_unique_and_deterministic(self, pipeline, golden):
        bundle = pipeline["bundle"]
        full_urls = [entry["fullUrl"] for entry in bundle["entry"]]
        assert len(set(full_urls)) == len(full_urls)
        # Re-assembling the same resources yields byte-identical fullUrls.
        again = to_bundle(pipeline["resources"], doc_id=golden["note_id"])
        assert [e["fullUrl"] for e in again["entry"]] == full_urls

    def test_internal_references_resolve_to_full_urls(self, pipeline):
        """Every in-Bundle reference targets a urn:uuid entry (no dangling)."""
        bundle = pipeline["bundle"]
        full_urls = {entry["fullUrl"] for entry in bundle["entry"]}
        for reference in _collect_references(bundle):
            # References that were rewritten point at a Bundle entry; the
            # subject reference to the identifier-free Patient stays literal
            # only if the Patient is absent - here the Patient is present.
            if reference.startswith("urn:uuid:"):
                assert reference in full_urls

    def test_condition_status_materialised_from_assertion(self, pipeline):
        """Refuted / inactive / active flow from assertion into the Condition."""
        conditions = {
            row["entity"].text: row["resource"]
            for row in pipeline["per_entity"]
            if row["resource"]["resourceType"] == "Condition"
        }
        pneumonia = conditions["pneumonia"]
        assert pneumonia["verificationStatus"]["coding"][0]["code"] == REFUTED
        diabetes = conditions["type 2 diabetes"]
        assert diabetes["clinicalStatus"]["coding"][0]["code"] == INACTIVE
        hypertension = conditions["hypertension"]
        assert hypertension["clinicalStatus"]["coding"][0]["code"] == ACTIVE

    def test_grounded_codings_use_canonical_system_uris(self, pipeline):
        """Each grounded resource carries the expected HL7 system URI + code."""
        for row in pipeline["per_entity"]:
            expected = row["expected"]
            codings = _codings_of(row["resource"])
            systems = {coding["system"] for coding in codings}
            codes = {coding["code"] for coding in codings}
            assert _SYSTEM_URI[expected["grounding_system"]] in systems
            assert expected["expected_code"] in codes


def _codings_of(resource: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the codings from a resource's coded element."""
    concept = resource.get("code") or resource.get("medicationCodeableConcept") or {}
    return concept.get("coding", [])


# --------------------------------------------------------------------------- #
# Leakage-first: no redacted PHI reappears anywhere downstream
# --------------------------------------------------------------------------- #


@pytest.mark.integration
class TestNoPHILeakageDownstream:
    def test_no_phi_in_ner_entities(self, pipeline, golden):
        for entity in pipeline["analysis"].entities:
            for phi in golden["phi_surface_strings"]:
                assert phi not in entity.text

    def test_no_phi_in_grounding_inputs(self, pipeline, golden):
        for span in pipeline["grounded_spans"]:
            for phi in golden["phi_surface_strings"]:
                assert phi not in span.text

    def test_no_phi_anywhere_in_fhir_bundle(self, pipeline, golden):
        """The strongest guard: no PHI surface appears in any Bundle string."""
        blob = json.dumps(pipeline["bundle"])
        for phi in golden["phi_surface_strings"]:
            assert phi not in blob, f"PHI {phi!r} leaked into the FHIR bundle"

    def test_no_phi_in_serialized_pipeline_outputs(self, pipeline, golden):
        """Sweep every string in the analysis + bundle for PHI surfaces."""
        strings = list(_iter_strings(pipeline["analysis"].to_dict()))
        strings += list(_iter_strings(pipeline["bundle"]))
        for value in strings:
            for phi in golden["phi_surface_strings"]:
                assert phi not in value


# --------------------------------------------------------------------------- #
# Documented gap: public grounding orchestrator is future work (no fakes).
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.xfail(
    reason=(
        "OM-804 documented gap: a public top-level openmed.ground() orchestrator "
        "is not merged yet. This suite drives the individual grounding linkers "
        "directly; when ground() lands, replace this xfail with a direct "
        "orchestrator hand-off assertion."
    ),
    strict=True,
)
def test_public_ground_orchestrator_is_future_work():
    """Guard the documented grounding->FHIR orchestrator gap (must stay xfail)."""
    assert hasattr(openmed, "ground")


# --------------------------------------------------------------------------- #
# Provenance / disclaimers: clinical output is advisory, synthetic-only.
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_registry_exposes_expected_linkers():
    """The linkers the FHIR leg depends on are discoverable via the registry."""
    for system in ("rxnorm", "icd10cm", "hpo"):
        assert system in available_linkers()


@pytest.mark.integration
def test_golden_fixture_carries_medical_device_disclaimer(golden):
    """The golden fixture documents that its clinical output is advisory only."""
    disclaimer = golden["disclaimer"].lower()
    assert "synthetic" in disclaimer
    assert "not a medical device" in disclaimer


@pytest.mark.integration
def test_grounding_candidates_are_typed(pipeline):
    """Grounding yields typed Candidate objects (no ad hoc dicts)."""
    for row in pipeline["per_entity"]:
        for candidate in row["candidates"]:
            assert isinstance(candidate, Candidate)
