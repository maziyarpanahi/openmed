from __future__ import annotations

import pytest

from openmed.core.model_registry import resolve_pii_family_transfer_route
from openmed.core.pii_i18n import DEFAULT_PII_MODELS, SUPPORTED_LANGUAGES
from openmed.training.adapters.config import (
    CLINICAL_ADAPTER_DISCLAIMER,
    DEFAULT_BACKBONE_MODEL_ID,
    DEFAULT_FAMILY_TRANSFER_CONFIG,
    DEFAULT_LANGUAGE_FAMILIES,
    AdapterMetadata,
    FamilyTransferConfig,
    TransferEdge,
)
from openmed.training.adapters.family_transfer import (
    adapter_metadata_for,
    donor_languages_for,
    primary_donor_for,
    resolve_family_transfer,
)


def _edge(
    target_language: str = "te",
    donor_language: str = "hi",
    *,
    adapter_id: str = "test/adapter",
    license_name: str = "apache-2.0",
    offline_runnable: bool = True,
    priority: int = 1,
) -> TransferEdge:
    return TransferEdge(
        target_language=target_language,
        donor_language=donor_language,
        family_id="indic",
        adapter=AdapterMetadata(
            adapter_id=adapter_id,
            license=license_name,
            offline_runnable=offline_runnable,
            provenance="synthetic unit-test adapter metadata",
        ),
        priority=priority,
    )


def test_default_family_transfer_config_covers_supported_languages():
    assert set(DEFAULT_FAMILY_TRANSFER_CONFIG.languages) == SUPPORTED_LANGUAGES

    for language in SUPPORTED_LANGUAGES:
        family = DEFAULT_FAMILY_TRANSFER_CONFIG.family_for_language(language)
        assert family is not None
        assert language in family.languages
        assert family.scripts


def test_telugu_resolves_to_hindi_donor_with_adapter_metadata():
    resolution = resolve_family_transfer("te-IN")

    assert resolution is not None
    assert resolution.language == "te"
    assert resolution.family.family_id == "indic"
    assert resolution.primary_donor_language == "hi"
    assert primary_donor_for("te") == "hi"

    metadata = adapter_metadata_for("te")
    assert metadata is not None
    assert metadata.adapter_id == "family-transfer/indic-hi-to-te"
    assert metadata.backbone_model_id == DEFAULT_BACKBONE_MODEL_ID
    assert metadata.license == "apache-2.0"
    assert metadata.offline_runnable is True
    assert metadata.disclaimer == CLINICAL_ADAPTER_DISCLAIMER


def test_romance_transfer_donors_are_ordered_deterministically():
    assert donor_languages_for("pt") == ("es", "fr", "it")
    assert donor_languages_for("it") == ("es", "fr")
    assert primary_donor_for("fr") is None


def test_unsupported_language_has_no_family_transfer_route():
    assert resolve_family_transfer("ko") is None
    assert resolve_pii_family_transfer_route("ko") is None


def test_registry_route_exposes_target_donor_and_adapter_metadata():
    route = resolve_pii_family_transfer_route("te")

    assert route is not None
    assert route.language == "te"
    assert route.family_id == "indic"
    assert route.target_model_id == DEFAULT_PII_MODELS["te"]
    assert route.backbone_model_id == DEFAULT_BACKBONE_MODEL_ID
    assert route.donor_language == "hi"
    assert route.donor_model_id == DEFAULT_PII_MODELS["hi"]
    assert route.adapter_id == "family-transfer/indic-hi-to-te"
    assert route.adapter_license == "apache-2.0"
    assert route.adapter_provenance
    assert route.clinical_disclaimer == CLINICAL_ADAPTER_DISCLAIMER
    assert route.offline_runnable is True
    assert route.mode == "zero_shot_or_adapter_init"


def test_transfer_config_rejects_missing_donor_family():
    bad_edge = _edge(donor_language="xx")

    with pytest.raises(ValueError, match="donor 'xx' has no language family"):
        FamilyTransferConfig(
            families=DEFAULT_LANGUAGE_FAMILIES,
            transfer_graph={"te": (bad_edge,)},
        )


def test_transfer_edge_rejects_self_donor():
    with pytest.raises(ValueError, match="donor_language must differ"):
        _edge(target_language="te", donor_language="te")


def test_transfer_config_rejects_donor_cycles():
    hi_to_te = TransferEdge(
        target_language="hi",
        donor_language="te",
        family_id="indic",
        adapter=AdapterMetadata(
            adapter_id="test/hi-cycle",
            provenance="synthetic unit-test adapter metadata",
        ),
    )

    with pytest.raises(ValueError, match="contains a cycle"):
        FamilyTransferConfig(
            families=DEFAULT_LANGUAGE_FAMILIES,
            transfer_graph={
                "te": (_edge(adapter_id="test/te-cycle"),),
                "hi": (hi_to_te,),
            },
        )


def test_transfer_config_rejects_nonpermissive_adapter_license():
    bad_edge = _edge(license_name="cc-by-nc-4.0")

    with pytest.raises(ValueError, match="not permissive"):
        FamilyTransferConfig(
            families=DEFAULT_LANGUAGE_FAMILIES,
            transfer_graph={"te": (bad_edge,)},
        )


def test_transfer_config_rejects_non_offline_adapter_metadata():
    bad_edge = _edge(offline_runnable=False)

    with pytest.raises(ValueError, match="offline-runnable"):
        FamilyTransferConfig(
            families=DEFAULT_LANGUAGE_FAMILIES,
            transfer_graph={"te": (bad_edge,)},
        )
