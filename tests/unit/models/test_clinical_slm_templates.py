"""Offline acceptance tests for clinical SLM prompt-template provenance."""

from __future__ import annotations

import json
import socket
from dataclasses import replace

import pytest

import openmed.models.clinical_slm_templates as template_module
from openmed.models.clinical_slm_templates import (
    ClinicalSLMTemplateDigestMismatchError,
    ClinicalSLMTemplateError,
    ClinicalSLMTemplateProvenance,
    ClinicalSLMTemplateSet,
    ClinicalSLMTemplateSubstitutionError,
    UndeclaredTemplateSubstitutionError,
    build_template_provenance,
    build_template_set,
    canonicalize_template,
    compute_template_digest,
    render_clinical_slm_templates,
    verify_template_provenance,
)

SYSTEM_TEMPLATE = "You are a local clinical review assistant.\n"
TASK_TEMPLATE = "Summarize the de-identified note: {clinical_note}."
OUTPUT_FORMAT_TEMPLATE = 'Return only JSON with keys {{"summary", "review_required"}}.'


def _template_set() -> ClinicalSLMTemplateSet:
    return ClinicalSLMTemplateSet(
        system=SYSTEM_TEMPLATE,
        task=TASK_TEMPLATE,
        output_format=OUTPUT_FORMAT_TEMPLATE,
    )


def test_builder_rejects_conflicting_task_alias():
    with pytest.raises(ClinicalSLMTemplateError):
        build_template_set(
            system="s", task="t", task_template="different", output_format="o"
        )


def test_builder_rejects_conflicting_system_alias():
    with pytest.raises(ClinicalSLMTemplateError):
        build_template_set(
            system="s", system_template="different", task="t", output_format="o"
        )


def test_mutated_template_text_cannot_reuse_old_provenance():
    templates = _template_set()
    provenance = templates.provenance
    object.__setattr__(templates, "system", "Changed instruction")
    with pytest.raises(ClinicalSLMTemplateError):
        verify_template_provenance(templates, provenance)


def test_provenance_aggregate_must_bind_its_fields():
    with pytest.raises(ClinicalSLMTemplateError):
        replace(_template_set().provenance, template_set_digest="sha256:" + "0" * 64)


def test_unicode_error_discards_raw_input_context():
    with pytest.raises(ClinicalSLMTemplateError) as caught:
        canonicalize_template("SYNTHETIC_PRIVATE\ud800")
    assert caught.value.__context__ is None


def test_provenance_aliases_cannot_disagree():
    provenance = _template_set().provenance.to_dict()
    provenance["system_template_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ClinicalSLMTemplateError):
        ClinicalSLMTemplateProvenance.from_mapping(provenance)


def test_runtime_unicode_is_validated_before_render():
    with pytest.raises(ClinicalSLMTemplateError):
        _template_set().render({"clinical_note": "SYNTHETIC_PRIVATE\ud800"})


def test_expanding_render_is_bounded_before_concatenation(monkeypatch):
    templates = ClinicalSLMTemplateSet("s", "{value}" * 50, "o")
    monkeypatch.setattr(template_module, "MAX_RENDERED_BYTES", 100)
    with pytest.raises(ClinicalSLMTemplateSubstitutionError):
        templates.render({"value": "test"})


def test_declaration_iterator_stops_at_limit():
    read_count = 0

    def names():
        nonlocal read_count
        while True:
            read_count += 1
            assert read_count <= template_module.MAX_COLLECTION_ITEMS + 1
            yield "clinical_note"

    with pytest.raises(ClinicalSLMTemplateError):
        ClinicalSLMTemplateSet(
            SYSTEM_TEMPLATE,
            TASK_TEMPLATE,
            OUTPUT_FORMAT_TEMPLATE,
            declared_substitutions=names(),
        )
    assert read_count == template_module.MAX_COLLECTION_ITEMS + 1


def test_global_declarations_match_union_of_template_placeholders():
    templates = ClinicalSLMTemplateSet(
        SYSTEM_TEMPLATE,
        TASK_TEMPLATE,
        OUTPUT_FORMAT_TEMPLATE,
        declared_substitutions=["clinical_note"],
    )
    assert templates.provenance == _template_set().provenance


def test_direct_provenance_uses_current_template_text():
    templates = _template_set()
    old_digest = templates.provenance.system_digest
    object.__setattr__(templates, "system", "Changed instruction")
    assert templates.provenance.system_digest != old_digest


def test_conflicting_run_provenance_wrappers_are_rejected():
    templates = _template_set()
    with pytest.raises(ClinicalSLMTemplateError):
        verify_template_provenance(
            templates,
            {
                "prompt_templates": templates.provenance.to_dict(),
                "template_provenance": {},
            },
        )


def test_rendered_record_revalidates_provenance():
    rendered = _template_set().render(clinical_note="synthetic")
    object.__setattr__(rendered.provenance, "schema_version", "synthetic_private")
    with pytest.raises(ClinicalSLMTemplateError):
        replace(rendered)


def test_digest_is_deterministic_and_canonicalizes_nonsemantic_line_endings() -> None:
    first = _template_set()
    second = ClinicalSLMTemplateSet(
        system=SYSTEM_TEMPLATE.replace("\n", "\r\n"),
        task="Summarize the de-identified note: {clinical_note}.",
        output_format=OUTPUT_FORMAT_TEMPLATE,
    )

    assert canonicalize_template("Cafe\u0301\r\n") == "Café\n"
    assert compute_template_digest("Cafe\u0301\r\n") == compute_template_digest(
        "Café\n"
    )
    assert first.provenance.to_dict() == second.provenance.to_dict()
    assert set(first.digests) == {"system", "task", "output_format"}
    assert first.template_set_digest.startswith("sha256:")


def test_each_template_is_bound_and_provenance_is_value_free() -> None:
    original = _template_set()
    changed = ClinicalSLMTemplateSet(
        system=SYSTEM_TEMPLATE,
        task=TASK_TEMPLATE,
        output_format='Return only JSON with key {{"summary"}}.',
    )

    assert original.system_digest == changed.system_digest
    assert original.task_digest == changed.task_digest
    assert original.output_format_digest != changed.output_format_digest
    assert original.template_set_digest != changed.template_set_digest
    assert "de-identified note" not in original.provenance.to_json()
    assert "review assistant" not in original.provenance.to_markdown()
    assert "clinical_note" in original.provenance.to_json()


def test_render_requires_exactly_the_declared_runtime_substitutions() -> None:
    templates = _template_set()
    synthetic_value = "SYNTHETIC_CLINICAL_VALUE_01"

    rendered = templates.render({"clinical_note": synthetic_value})

    assert rendered.task.endswith(f"{synthetic_value}.")
    assert rendered["output_format"] == (
        'Return only JSON with keys {"summary", "review_required"}.'
    )
    assert synthetic_value not in repr(templates)
    assert synthetic_value not in repr(rendered)
    assert synthetic_value not in rendered.provenance.to_json()
    assert synthetic_value not in rendered.provenance.to_markdown()

    with pytest.raises(UndeclaredTemplateSubstitutionError) as undeclared:
        templates.render(
            {
                "clinical_note": synthetic_value,
                "unexpected_runtime_value": synthetic_value,
            }
        )
    assert undeclared.value.reason_code == "undeclared_substitution"
    assert synthetic_value not in str(undeclared.value)

    with pytest.raises(ClinicalSLMTemplateSubstitutionError) as missing:
        templates.render()
    assert missing.value.reason_code == "missing_substitution"


def test_explicit_declarations_must_match_placeholders() -> None:
    templates = ClinicalSLMTemplateSet(
        system=SYSTEM_TEMPLATE,
        task=TASK_TEMPLATE,
        output_format=OUTPUT_FORMAT_TEMPLATE,
        declared_substitutions={
            "output_format": (),
            "task": ("clinical_note",),
            "system": (),
        },
    )
    reordered = ClinicalSLMTemplateSet(
        system=SYSTEM_TEMPLATE,
        task=TASK_TEMPLATE,
        output_format=OUTPUT_FORMAT_TEMPLATE,
        declared_substitutions={
            "system": (),
            "task": ("clinical_note",),
            "output_format": (),
        },
    )

    assert templates.provenance.to_dict() == reordered.provenance.to_dict()
    with pytest.raises(ClinicalSLMTemplateError) as mismatch:
        ClinicalSLMTemplateSet(
            system=SYSTEM_TEMPLATE,
            task=TASK_TEMPLATE,
            output_format=OUTPUT_FORMAT_TEMPLATE,
            declared_substitutions={"task": (), "system": (), "output_format": ()},
        )
    assert mismatch.value.reason_code == "declaration_mismatch"


def test_provenance_round_trip_and_digest_mismatch_are_fail_closed() -> None:
    templates = _template_set()
    payload = templates.provenance.to_dict()
    parsed = ClinicalSLMTemplateProvenance.from_mapping(payload)

    assert verify_template_provenance(templates, parsed) == parsed
    assert verify_template_provenance(templates, payload).to_dict() == payload

    altered = json.loads(json.dumps(payload))
    altered["template_digests"]["task"] = "sha256:" + "0" * 64
    with pytest.raises(ClinicalSLMTemplateDigestMismatchError) as mismatch:
        verify_template_provenance(templates, altered)
    assert mismatch.value.to_dict() == {
        "code": "template_digest_mismatch",
        "message": "clinical SLM template digest does not match",
    }


def test_mapping_factory_and_report_formats_are_offline_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("template provenance attempted network access")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    templates = ClinicalSLMTemplateSet.from_mapping(
        {
            "system_template": SYSTEM_TEMPLATE,
            "task_template": TASK_TEMPLATE,
            "output_format_template": OUTPUT_FORMAT_TEMPLATE,
        }
    )
    direct = build_template_provenance(
        SYSTEM_TEMPLATE,
        TASK_TEMPLATE,
        OUTPUT_FORMAT_TEMPLATE,
    )

    assert templates.provenance.to_json() == direct.to_json()
    assert (
        render_clinical_slm_templates(
            templates,
            clinical_note="synthetic note",
        ).provenance_report
        == templates.provenance_report
    )
    assert json.loads(templates.to_json()) == templates.provenance_report
    assert templates.to_markdown().startswith("# Clinical SLM Template Provenance")
