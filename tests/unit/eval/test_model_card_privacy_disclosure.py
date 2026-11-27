"""Privacy-disclosure section rendering and validation for model cards."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

import pytest

from openmed.eval.model_card_builder import (
    ModelCardBuilderError,
    build_model_card,
    validate_model_card_consistency,
)
from openmed.eval.model_card_privacy import (
    NO_FORMAL_DP_GUARANTEE_STATEMENT,
    PRIVACY_ACCOUNTANT_STATEMENT_VERSION,
    PRIVACY_DISCLOSURE_SCHEMA_VERSION,
    PrivacyDisclosureError,
    build_privacy_disclosure,
    privacy_accountant_statement_digest,
    privacy_disclosure_json,
    render_privacy_disclosure_markdown,
)
from tests.fixtures.eval.model_card_privacy import (
    LINEAGE_DIGEST,
    PRIVACY_DISCLOSURE_SCHEMA,
    SITE_METRIC_SENTINEL,
    SITE_SENTINEL,
    dp_centralized_disclosure,
    dp_federated_disclosure,
    federated_without_dp_disclosure,
    gate_report,
    manifest_row,
    promotion_evidence,
    training_provenance,
    without,
)

# Digests of the card rendered on master with no privacy disclosure supplied.
_NO_DISCLOSURE_MARKDOWN_SHA256 = (
    "e43b33ed8bb4121606a610fe2893e1d13c158ef2b1fc38b84e09398215b1bbfd"
)
_NO_DISCLOSURE_DATASHEET_SHA256 = (
    "625a730c553a7f2f544ae0838f81063ba69906abb44c3dd365a10d42db31227c"
)
# Pinned digests of the complete synthetic federated DP disclosure.
_DP_STATEMENT_DIGEST = (
    "sha256:2ac0ea6716bcdc537af4ae84f0b8596adfbbe0d8e38853af4dc1cba065589241"
)
_DP_DISCLOSURE_DIGEST = (
    "sha256:8a63769bcb13ebaba3e9e1acb2ccf4106f49598aeaae926b148a91b84f3d0e0b"
)
_DP_SECTION_JSON_SHA256 = (
    "419e3521bb66320ce61a317be8111dc8c30aea557200dd7589f722dede52c367"
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CODE = re.compile(r"[a-z][a-z0-9_.]*\Z")


def _canonical_digest(payload: Any) -> str:
    text = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _expected_dp_statement_digest() -> str:
    return _canonical_digest(
        {
            "accountant": "rdp",
            "adjacency": "add_remove",
            "amplification_claimed": True,
            "clipping": "per_unit_l2",
            "conversion": "rdp_to_approx_dp",
            "delta": 1e-06,
            "epsilon": 3.0,
            "iteration_count": 200,
            "iteration_unit": "rounds",
            "mechanism": "dp_fedavg_gaussian",
            "privacy_unit": "patient",
            "sampling": "poisson",
            "statement_version": "openmed.privacy_accountant_statement.v1",
            "training_setting": "federated",
            "tuning_cost_included": True,
        }
    )


def _dp_section() -> dict[str, Any]:
    disclosure = dp_federated_disclosure()
    return build_privacy_disclosure(
        disclosure,
        promotion_evidence=promotion_evidence(
            privacy_accountant_statement_digest(disclosure)
        ),
    )


def _error_code(excinfo: pytest.ExceptionInfo[PrivacyDisclosureError]) -> str:
    return excinfo.value.code


# -- byte-stable rendering ---------------------------------------------------


def test_statement_digest_matches_independent_canonical_hash() -> None:
    assert PRIVACY_ACCOUNTANT_STATEMENT_VERSION == (
        "openmed.privacy_accountant_statement.v1"
    )
    assert privacy_accountant_statement_digest(dp_federated_disclosure()) == (
        _expected_dp_statement_digest()
    )
    assert _expected_dp_statement_digest() == _DP_STATEMENT_DIGEST


def test_complete_dp_disclosure_renders_golden_markdown() -> None:
    section = _dp_section()
    statement_digest = _DP_STATEMENT_DIGEST
    disclosure_digest = _DP_DISCLOSURE_DIGEST

    assert render_privacy_disclosure_markdown(section) == (
        "## Privacy Disclosure\n"
        "\n"
        "> Differential privacy: epsilon=3.0, delta=1e-06 per `patient` under "
        "`add_remove` adjacency, accounted with `rdp` (`rdp_to_approx_dp`).\n"
        "\n"
        "| Field | Value |\n"
        "|---|---|\n"
        "| Schema version | `openmed.model_card.privacy_disclosure.v1` |\n"
        "| Formal guarantee | `differential_privacy` |\n"
        "| Training setting | `federated` |\n"
        "| Mechanism | `dp_fedavg_gaussian` |\n"
        "| Privacy unit | `patient` |\n"
        "| Adjacency | `add_remove` |\n"
        "| Epsilon | 3.0 |\n"
        "| Delta | 1e-06 |\n"
        "| Accountant | `rdp` |\n"
        "| Conversion | `rdp_to_approx_dp` |\n"
        "| Iterations | 200 `rounds` |\n"
        "| Sampling | `poisson` |\n"
        "| Amplification by sampling claimed | Yes |\n"
        "| Clipping | `per_unit_l2` |\n"
        "| Secure aggregation | `cryptographic` |\n"
        "| Participant band | `fourfold_or_more` |\n"
        "| Tuning cost included | Yes |\n"
        f"| Accountant statement digest | `{statement_digest}` |\n"
        f"| Disclosure digest | `{disclosure_digest}` |\n"
        "\n"
        "### Risks Not Covered\n"
        "\n"
        "- `group_privacy_not_covered`: The guarantee does not extend to groups "
        "of related privacy units.\n"
        "- `public_pretraining_data_not_covered`: Data used to pretrain the base "
        "model is not covered.\n"
    )


def test_complete_dp_disclosure_json_is_byte_stable() -> None:
    section = _dp_section()
    reordered = dict(reversed(list(dp_federated_disclosure().items())))
    reordered["limitations"] = list(reversed(reordered["limitations"]))
    second = build_privacy_disclosure(
        reordered,
        promotion_evidence=dict(
            reversed(
                list(
                    promotion_evidence(
                        privacy_accountant_statement_digest(reordered)
                    ).items()
                )
            )
        ),
    )

    first_json = privacy_disclosure_json(section)
    assert first_json == privacy_disclosure_json(second)
    assert (
        hashlib.sha256(first_json.encode("utf-8")).hexdigest()
        == _DP_SECTION_JSON_SHA256
    )
    assert render_privacy_disclosure_markdown(section) == (
        render_privacy_disclosure_markdown(second)
    )
    assert first_json.endswith("}\n")
    assert first_json.isascii()

    payload = json.loads(first_json)
    without_digest = {
        key: value for key, value in payload.items() if key != "disclosure_digest"
    }
    assert payload["disclosure_digest"] == _canonical_digest(without_digest)
    assert payload == {
        "accountant": "rdp",
        "adjacency": "add_remove",
        "amplification_claimed": True,
        "clipping": "per_unit_l2",
        "conversion": "rdp_to_approx_dp",
        "delta": 1e-06,
        "disclosure_digest": _DP_DISCLOSURE_DIGEST,
        "epsilon": 3.0,
        "evidence": {"accountant_statement_digest": _expected_dp_statement_digest()},
        "formal_guarantee": "differential_privacy",
        "guarantee_statement": (
            "Differential privacy: epsilon=3.0, delta=1e-06 per `patient` under "
            "`add_remove` adjacency, accounted with `rdp` (`rdp_to_approx_dp`)."
        ),
        "iteration_count": 200,
        "iteration_unit": "rounds",
        "limitations": [
            {
                "code": "group_privacy_not_covered",
                "statement": (
                    "The guarantee does not extend to groups of related privacy units."
                ),
            },
            {
                "code": "public_pretraining_data_not_covered",
                "statement": "Data used to pretrain the base model is not covered.",
            },
        ],
        "mechanism": "dp_fedavg_gaussian",
        "participant_band": "fourfold_or_more",
        "privacy_unit": "patient",
        "sampling": "poisson",
        "schema_version": PRIVACY_DISCLOSURE_SCHEMA_VERSION,
        "secure_aggregation": "cryptographic",
        "training_setting": "federated",
        "tuning_cost_included": True,
    }


def test_model_card_with_disclosure_is_byte_stable() -> None:
    disclosure = dp_federated_disclosure()
    evidence = promotion_evidence(privacy_accountant_statement_digest(disclosure))

    def build():
        return build_model_card(
            manifest_row(),
            gate_report(),
            training_provenance=training_provenance(),
            privacy_disclosure=disclosure,
            privacy_evidence=evidence,
        )

    first = build()
    second = build()

    assert first.markdown == second.markdown
    assert first.datasheet_json() == second.datasheet_json()
    section = _dp_section()
    assert first.datasheet["privacy_disclosure"] == section
    assert first.markdown.endswith(render_privacy_disclosure_markdown(section))
    assert first.markdown.index("## Datasheet") < first.markdown.index(
        "## Privacy Disclosure"
    )


def test_centralized_dp_disclosure_reports_excluded_tuning_cost() -> None:
    disclosure = dp_centralized_disclosure()
    section = build_privacy_disclosure(
        disclosure,
        promotion_evidence=promotion_evidence(
            privacy_accountant_statement_digest(disclosure)
        ),
    )
    markdown = render_privacy_disclosure_markdown(section)

    assert section["epsilon"] == 8.0
    assert "| Epsilon | 8.0 |" in markdown
    assert "| Iterations | 1200 `steps` |" in markdown
    assert "| Secure aggregation | `not_applicable` |" in markdown
    assert "| Tuning cost included | No |" in markdown
    assert "| Amplification by sampling claimed | No |" in markdown


# -- missing or disagreeing evidence -----------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "privacy_unit",
        "delta",
        "adjacency",
        "accountant",
        "conversion",
        "sampling",
        "amplification_claimed",
        "tuning_cost_included",
    ],
)
def test_missing_dp_field_blocks_rendering(field: str) -> None:
    disclosure = without(dp_federated_disclosure(), field)
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_privacy_disclosure(
            disclosure,
            promotion_evidence=promotion_evidence(_expected_dp_statement_digest()),
        )
    assert _error_code(excinfo) == f"{field}_missing"


def test_null_privacy_unit_blocks_rendering() -> None:
    disclosure = dp_federated_disclosure()
    disclosure["privacy_unit"] = None
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_privacy_disclosure(
            disclosure,
            promotion_evidence=promotion_evidence(_expected_dp_statement_digest()),
        )
    assert _error_code(excinfo) == "privacy_unit_missing"


def test_epsilon_without_mechanism_is_rejected() -> None:
    disclosure = dp_federated_disclosure()
    del disclosure["mechanism"]
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_privacy_disclosure(disclosure)
    assert _error_code(excinfo) == "mechanism_missing"


@pytest.mark.parametrize(
    "evidence",
    [None, {}, {"lineage_digest": LINEAGE_DIGEST}],
    ids=["no-evidence", "empty-evidence", "no-accountant-digest"],
)
def test_dp_claim_without_accountant_evidence_blocks_rendering(
    evidence: dict[str, str] | None,
) -> None:
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_privacy_disclosure(
            dp_federated_disclosure(),
            promotion_evidence=evidence,
        )
    assert _error_code(excinfo) == "accountant_evidence_missing"


def test_model_card_without_accountant_evidence_blocks_rendering() -> None:
    with pytest.raises(ModelCardBuilderError) as excinfo:
        build_model_card(
            manifest_row(),
            gate_report(),
            training_provenance=training_provenance(),
            privacy_disclosure=dp_federated_disclosure(),
        )
    assert isinstance(excinfo.value, PrivacyDisclosureError)
    assert excinfo.value.code == "accountant_evidence_missing"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epsilon", 3.5),
        ("delta", 1e-05),
        ("privacy_unit", "note"),
        ("iteration_count", 201),
        ("tuning_cost_included", False),
    ],
)
def test_claim_disagreeing_with_evidence_digest_fails(
    field: str,
    value: Any,
) -> None:
    disclosure = dp_federated_disclosure()
    disclosure[field] = value
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_privacy_disclosure(
            disclosure,
            promotion_evidence=promotion_evidence(_expected_dp_statement_digest()),
        )
    assert _error_code(excinfo) == "accountant_evidence_mismatch"
    assert str(value) not in str(excinfo.value)


def test_matching_evidence_negative_control_renders() -> None:
    disclosure = dp_federated_disclosure()
    disclosure["epsilon"] = 3.5
    section = build_privacy_disclosure(
        disclosure,
        promotion_evidence=promotion_evidence(
            privacy_accountant_statement_digest(disclosure)
        ),
    )
    assert section["epsilon"] == 3.5


def test_malformed_accountant_digest_is_rejected() -> None:
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_privacy_disclosure(
            dp_federated_disclosure(),
            promotion_evidence={"accountant_statement_digest": "sha256:abc"},
        )
    assert _error_code(excinfo) == "promotion_evidence_not_digest"


# -- federation without differential privacy ---------------------------------


def test_federated_without_dp_renders_no_guarantee_statement() -> None:
    section = build_privacy_disclosure(federated_without_dp_disclosure())
    markdown = render_privacy_disclosure_markdown(section)

    assert NO_FORMAL_DP_GUARANTEE_STATEMENT == (
        "No formal differential-privacy guarantee applies to this model."
    )
    assert f"> {NO_FORMAL_DP_GUARANTEE_STATEMENT}\n" in markdown
    assert section["formal_guarantee"] == "none"
    assert section["guarantee_statement"] == NO_FORMAL_DP_GUARANTEE_STATEMENT
    assert section["epsilon"] is None
    assert section["delta"] is None
    assert section["evidence"] is None
    assert "| Epsilon | Not applicable |" in markdown
    assert "| Delta | Not applicable |" in markdown
    assert "| Privacy unit | Not applicable |" in markdown
    assert "| Iterations | 50 `rounds` |" in markdown
    assert "| Secure aggregation | `cryptographic` |" in markdown
    assert "| Participant band | `double_to_under_fourfold` |" in markdown
    assert "`secure_aggregation_not_dp`" in markdown


def test_federated_without_dp_model_card_includes_statement() -> None:
    result = build_model_card(
        manifest_row(),
        gate_report(),
        training_provenance=training_provenance(),
        privacy_disclosure=federated_without_dp_disclosure(),
    )
    assert NO_FORMAL_DP_GUARANTEE_STATEMENT in result.markdown
    datasheet = json.loads(result.datasheet_json())
    assert datasheet["privacy_disclosure"]["formal_guarantee"] == "none"


def test_dp_disclosure_does_not_render_no_guarantee_statement() -> None:
    markdown = render_privacy_disclosure_markdown(_dp_section())
    assert NO_FORMAL_DP_GUARANTEE_STATEMENT not in markdown


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epsilon", 3.0),
        ("delta", 1e-06),
        ("privacy_unit", "patient"),
        ("accountant", "rdp"),
        ("tuning_cost_included", True),
    ],
)
def test_federated_without_dp_cannot_carry_dp_claims(field: str, value: Any) -> None:
    disclosure = federated_without_dp_disclosure()
    disclosure[field] = value
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_privacy_disclosure(disclosure)
    assert _error_code(excinfo) == "dp_field_without_mechanism"


# -- structural validation ---------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        (
            {"schema_version": "openmed.model_card.privacy_disclosure.v0"},
            "schema_version_unsupported",
        ),
        ({"schema_version": None}, "schema_version_missing"),
        ({"training_setting": "hybrid"}, "training_setting_invalid"),
        ({"training_setting": "centralized"}, "mechanism_setting_mismatch"),
        ({"iteration_unit": "steps"}, "iteration_unit_mismatch"),
        ({"iteration_count": 0}, "iteration_count_invalid"),
        ({"iteration_count": True}, "iteration_count_invalid"),
        ({"conversion": "direct"}, "accountant_conversion_mismatch"),
        ({"sampling": "shuffled"}, "sampling_amplification_unsupported"),
        ({"clipping": "none"}, "clipping_required"),
        ({"epsilon": True}, "epsilon_invalid"),
        ({"epsilon": float("nan")}, "epsilon_invalid"),
        ({"epsilon": 0}, "epsilon_invalid"),
        ({"epsilon": "3.0"}, "epsilon_invalid"),
        ({"delta": 1.0}, "delta_invalid"),
        ({"delta": 0}, "delta_invalid"),
        ({"secure_aggregation": "not_applicable"}, "secure_aggregation_mismatch"),
        ({"participant_band": "not_applicable"}, "participant_band_mismatch"),
        ({"limitations": []}, "limitations_missing"),
        ({"limitations": ["not_a_limitation"]}, "limitation_invalid"),
        ({"amplification_claimed": "yes"}, "amplification_claimed_invalid"),
    ],
)
def test_invalid_disclosure_is_rejected(changes: dict[str, Any], code: str) -> None:
    disclosure = dp_federated_disclosure()
    disclosure.update(changes)
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_privacy_disclosure(
            disclosure,
            promotion_evidence=promotion_evidence(_expected_dp_statement_digest()),
        )
    assert _error_code(excinfo) == code


def test_centralized_setting_rejects_federation_fields() -> None:
    disclosure = dp_centralized_disclosure()
    disclosure["secure_aggregation"] = "cryptographic"
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_privacy_disclosure(disclosure)
    assert _error_code(excinfo) == "secure_aggregation_mismatch"


def test_errors_are_model_card_builder_errors() -> None:
    assert issubclass(PrivacyDisclosureError, ModelCardBuilderError)


def test_schema_version_constant_matches_fixture() -> None:
    assert PRIVACY_DISCLOSURE_SCHEMA_VERSION == PRIVACY_DISCLOSURE_SCHEMA


# -- privacy of rendered output ----------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update({"site_metrics": {SITE_SENTINEL: SITE_METRIC_SENTINEL}}),
        lambda d: d.update({SITE_SENTINEL: SITE_METRIC_SENTINEL}),
        lambda d: d.update({"participant_band": SITE_SENTINEL}),
        lambda d: d.update({"limitations": [SITE_SENTINEL]}),
        lambda d: d.update({"privacy_unit": SITE_SENTINEL}),
        lambda d: d.update({"iteration_count": SITE_SENTINEL}),
    ],
    ids=[
        "per-site-metrics-field",
        "site-named-field",
        "site-as-band",
        "site-as-limitation",
        "site-as-unit",
        "site-as-count",
    ],
)
def test_site_identities_never_reach_card_or_error(mutate) -> None:
    disclosure = dp_federated_disclosure()
    mutate(disclosure)
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_model_card(
            manifest_row(),
            gate_report(),
            training_provenance=training_provenance(),
            privacy_disclosure=disclosure,
            privacy_evidence=promotion_evidence(_expected_dp_statement_digest()),
        )
    message = str(excinfo.value)
    assert SITE_SENTINEL not in message
    assert SITE_METRIC_SENTINEL not in message
    assert excinfo.value.__cause__ is None
    assert re.fullmatch(r"privacy disclosure rejected: [a-z_]+ \([a-z_]+\)", message)


def test_promotion_evidence_raw_values_are_rejected_without_echo() -> None:
    evidence = promotion_evidence(_expected_dp_statement_digest())
    evidence["per_site_recall"] = SITE_METRIC_SENTINEL
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_privacy_disclosure(
            dp_federated_disclosure(),
            promotion_evidence=evidence,
        )
    assert _error_code(excinfo) == "promotion_evidence_not_digest"
    assert SITE_METRIC_SENTINEL not in str(excinfo.value)
    assert "per_site_recall" not in str(excinfo.value)


def test_card_contains_only_codes_bands_counts_and_digests() -> None:
    disclosure = dp_federated_disclosure()
    evidence = promotion_evidence(privacy_accountant_statement_digest(disclosure))
    # An unconsumed digest keyed by a site label must not leak its key.
    evidence[SITE_SENTINEL] = LINEAGE_DIGEST
    result = build_model_card(
        manifest_row(),
        gate_report(),
        training_provenance=training_provenance(),
        privacy_disclosure=disclosure,
        privacy_evidence=evidence,
    )

    assert SITE_SENTINEL not in result.markdown
    assert SITE_SENTINEL not in result.datasheet_json()
    assert LINEAGE_DIGEST not in result.markdown

    section = result.datasheet["privacy_disclosure"]
    statement_fields = {"guarantee_statement"}
    for key, value in section.items():
        if key in statement_fields or key in {"limitations", "evidence"}:
            continue
        if isinstance(value, str):
            assert _CODE.fullmatch(value) or _DIGEST.fullmatch(value), key
        else:
            assert value is None or isinstance(value, (bool, int, float)), key
    for item in section["limitations"]:
        assert _CODE.fullmatch(item["code"])
    assert set(section["evidence"]) == {"accountant_statement_digest"}
    assert _DIGEST.fullmatch(section["evidence"]["accountant_statement_digest"])


# -- consistency with the rest of the card -----------------------------------


def test_card_without_disclosure_is_byte_identical_to_previous_output() -> None:
    result = build_model_card(
        manifest_row(),
        gate_report(),
        training_provenance=training_provenance(),
    )
    assert (
        hashlib.sha256(result.markdown.encode("utf-8")).hexdigest()
        == _NO_DISCLOSURE_MARKDOWN_SHA256
    )
    assert (
        hashlib.sha256(result.datasheet_json().encode("utf-8")).hexdigest()
        == _NO_DISCLOSURE_DATASHEET_SHA256
    )
    assert "privacy_disclosure" not in result.datasheet
    assert "Privacy Disclosure" not in result.markdown


def test_privacy_evidence_without_disclosure_is_rejected() -> None:
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        build_model_card(
            manifest_row(),
            gate_report(),
            training_provenance=training_provenance(),
            privacy_evidence=promotion_evidence(_expected_dp_statement_digest()),
        )
    assert _error_code(excinfo) == "privacy_disclosure_missing"


def _dp_card():
    disclosure = dp_federated_disclosure()
    evidence = promotion_evidence(privacy_accountant_statement_digest(disclosure))
    return (
        build_model_card(
            manifest_row(),
            gate_report(),
            training_provenance=training_provenance(),
            privacy_disclosure=disclosure,
            privacy_evidence=evidence,
        ),
        evidence,
    )


def test_consistency_accepts_untampered_disclosure() -> None:
    result, evidence = _dp_card()
    validate_model_card_consistency(
        result.markdown,
        manifest_row(),
        gate_report(),
        datasheet=result.datasheet,
        privacy_evidence=evidence,
    )
    validate_model_card_consistency(
        result.markdown,
        manifest_row(),
        gate_report(),
        datasheet=result.datasheet,
    )


def test_consistency_rejects_tampered_datasheet_epsilon() -> None:
    result, evidence = _dp_card()
    tampered = copy.deepcopy(dict(result.datasheet))
    tampered["privacy_disclosure"]["epsilon"] = 0.5
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        validate_model_card_consistency(
            result.markdown,
            manifest_row(),
            gate_report(),
            datasheet=tampered,
            privacy_evidence=evidence,
        )
    assert excinfo.value.code == "accountant_evidence_mismatch"


def test_consistency_rejects_tampered_markdown_epsilon() -> None:
    result, evidence = _dp_card()
    tampered = result.markdown.replace("| Epsilon | 3.0 |", "| Epsilon | 0.5 |")
    assert tampered != result.markdown
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        validate_model_card_consistency(
            tampered,
            manifest_row(),
            gate_report(),
            datasheet=result.datasheet,
            privacy_evidence=evidence,
        )
    assert excinfo.value.code == "privacy_disclosure_markdown_mismatch"


def test_consistency_rejects_unbacked_markdown_section() -> None:
    plain = build_model_card(
        manifest_row(),
        gate_report(),
        training_provenance=training_provenance(),
    )
    dp_result, _ = _dp_card()
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        validate_model_card_consistency(
            dp_result.markdown,
            manifest_row(),
            gate_report(),
            datasheet=plain.datasheet,
        )
    assert excinfo.value.code == "privacy_disclosure_missing"


def test_consistency_rejects_different_injected_evidence() -> None:
    result, _ = _dp_card()
    other = promotion_evidence("sha256:" + "b" * 64)
    with pytest.raises(PrivacyDisclosureError) as excinfo:
        validate_model_card_consistency(
            result.markdown,
            manifest_row(),
            gate_report(),
            datasheet=result.datasheet,
            privacy_evidence=other,
        )
    assert excinfo.value.code == "accountant_evidence_mismatch"
