"""Offline provider declaration compatibility, deterministic output and controls."""

import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.multimodal.capability_matrix import (
    CompatibilityEntry,
    ModalityCapability,
    ModalityProfile,
    ProviderCapabilityEnvelope,
    capability_matrix,
    render_capability_matrix_json,
    render_capability_matrix_markdown,
)


def provider(provider_id="local", capabilities=None):
    return ProviderCapabilityEnvelope(provider_id, "1.0.0", capabilities)


def profile(profile_id="mono", features=("mono",)):
    return ModalityProfile(profile_id, "audio", features)


def test_four_distinct_outcomes_and_reasons():
    rows = capability_matrix(
        [
            provider("exact", (ModalityCapability("audio", ("mono",)),)),
            provider("transform", (ModalityCapability("audio", (), ("mono",)),)),
            provider("unsupported", ()),
            provider("missing"),
        ],
        [profile()],
    )
    assert [(row.provider_id, row.outcome, row.reasons) for row in rows] == [
        ("exact", "exact", ()),
        ("missing", "missing_capability", ("capability_missing",)),
        ("transform", "transform_required", ("declared_transform_required",)),
        ("unsupported", "unsupported", ("modality_unsupported",)),
    ]


def test_unsupported_feature_takes_precedence_over_transformable_feature():
    (row,) = capability_matrix(
        [provider(capabilities=(ModalityCapability("audio", (), ("mono",)),))],
        [profile(features=("mono", "floating_point"))],
    )
    assert row.outcome == "unsupported"
    assert row.reasons == ("feature_unsupported",)


def test_reordered_envelopes_profiles_features_and_rows_render_identically():
    p1 = provider("first", (ModalityCapability("audio", ("mono", "pcm")),))
    p2 = provider("second", (ModalityCapability("audio", ("pcm", "mono")),))
    profiles = [profile("a", ("mono", "pcm")), profile("b", ())]
    a = capability_matrix([p1, p2], profiles)
    b = capability_matrix([p2, p1], reversed(profiles))
    assert a == b
    assert render_capability_matrix_json(a) == render_capability_matrix_json(
        reversed(b)
    )
    assert render_capability_matrix_markdown(a) == render_capability_matrix_markdown(
        reversed(b)
    )


def test_json_and_markdown_golden_only_expose_five_fields():
    rows = capability_matrix(
        [provider(capabilities=(ModalityCapability("audio", ("mono",)),))], [profile()]
    )
    assert render_capability_matrix_json(rows) == (
        '[{"capability_version":"1.0.0","outcome":"exact","profile_id":"mono",'
        '"provider_id":"local","reasons":[]}]\n'
    )
    assert render_capability_matrix_markdown(rows) == (
        "| Provider ID | Capability version | Profile ID | Outcome | Reasons |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| local | 1.0.0 | mono | exact | - |\n"
    )
    assert set(json.loads(render_capability_matrix_json(rows))[0]) == {
        "provider_id",
        "capability_version",
        "profile_id",
        "outcome",
        "reasons",
    }


def test_empty_matrix_has_stable_json_and_header():
    assert capability_matrix([], [profile()]) == ()
    assert render_capability_matrix_json(()) == "[]\n"
    assert render_capability_matrix_markdown(()).count("\n") == 2


def test_declarations_are_frozen_and_normalize_mutable_inputs():
    features = ["pcm", "mono"]
    declaration = ModalityCapability("audio", features)
    features.append("private_payload")
    assert declaration.exact_features == ("mono", "pcm")
    with pytest.raises(FrozenInstanceError):
        declaration.modality = "image"


@pytest.mark.parametrize(
    "build",
    [
        lambda: profile(features=("mono", "mono")),
        lambda: ModalityCapability("audio", ("mono",), ("mono",)),
        lambda: provider(
            capabilities=(ModalityCapability("audio"), ModalityCapability("audio"))
        ),
        lambda: capability_matrix([provider(), provider()], [profile()]),
        lambda: capability_matrix([provider()], [profile(), profile()]),
        lambda: ProviderCapabilityEnvelope("local", "latest"),
        lambda: ModalityProfile("mono", "unsupported"),
        lambda: ModalityProfile("mono", "audio", schema_version=True),
        lambda: provider(capabilities=(None,)),
        lambda: CompatibilityEntry(
            "local", "1.0.0", "mono", "exact", ("feature_unsupported",)
        ),
    ],
)
def test_invalid_or_ambiguous_declarations_reject_before_rendering(build):
    with pytest.raises(ValueError, match="capability_matrix_"):
        build()


@pytest.mark.parametrize("field", ["provider", "profile", "feature"])
def test_private_looking_values_never_reach_output_or_errors(field):
    value = "synthetic/private-model-or-endpoint"
    with pytest.raises(ValueError) as exc:
        if field == "provider":
            provider(value)
        elif field == "profile":
            profile(value)
        else:
            profile(features=(value,))
    assert value not in str(exc.value)


@pytest.mark.parametrize(
    "renderer", [render_capability_matrix_json, render_capability_matrix_markdown]
)
def test_renderers_validate_inputs_and_duplicate_cells(renderer):
    with pytest.raises(ValueError, match="invalid_entry"):
        renderer([{"provider_id": "synthetic/path"}])
    row = CompatibilityEntry("local", "1.0.0", "mono", "exact", ())
    with pytest.raises(ValueError, match="duplicate_entry"):
        renderer([row, row])


@pytest.mark.parametrize(
    "kind,limit", [("providers", 101), ("profiles", 101), ("features", 129)]
)
def test_infinite_inputs_have_a_hard_consumption_bound(kind, limit):
    consumed = []

    def stream():
        while True:
            consumed.append(1)
            yield (
                provider()
                if kind == "providers"
                else profile()
                if kind == "profiles"
                else "mono"
            )

    with pytest.raises(ValueError, match="collection_limit"):
        if kind == "providers":
            capability_matrix(stream(), [profile()])
        elif kind == "profiles":
            capability_matrix([provider()], stream())
        else:
            profile(features=stream())
    assert len(consumed) == limit
