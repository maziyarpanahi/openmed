from __future__ import annotations

import pytest

from openmed.core.model_registry import resolve_pii_family_transfer_route
from openmed.core.pii_i18n import DEFAULT_PII_MODELS, SUPPORTED_LANGUAGES
from openmed.training.adapters import (
    CLINICAL_ADAPTER_DISCLAIMER,
    DEFAULT_BACKBONE_MODEL_ID,
    DEFAULT_FAMILY_TRANSFER_CONFIG,
    DEFAULT_LANGUAGE_FAMILIES,
    PERMISSIVE_ADAPTER_LICENSES,
    AdapterMetadata,
    FamilyTransferAdapterUnavailableError,
    FamilyTransferConfig,
    FamilyTransferRouter,
    TransferEdge,
    UnsupportedFamilyTransferLanguageError,
    adapter_metadata_for,
    donor_languages_for,
    get_family_transfer_config,
    primary_donor_for,
    resolve_family_transfer,
    route_family_adapter,
)


def _edge(
    target_language: str = "te",
    donor_language: str = "hi",
    *,
    adapter_id: str = "synthetic-test/adapter",
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
            provenance="Synthetic offline unit-test adapter metadata",
        ),
        priority=priority,
    )


def _available_adapter(language: str) -> AdapterMetadata:
    return AdapterMetadata(
        adapter_id=f"synthetic-offline/{language}-adapter",
        provenance=f"Synthetic offline {language} adapter fixture",
    )


def test_public_config_covers_every_supported_pii_language() -> None:
    config = get_family_transfer_config()

    assert config is DEFAULT_FAMILY_TRANSFER_CONFIG
    assert set(config.languages) == SUPPORTED_LANGUAGES
    for language in SUPPORTED_LANGUAGES:
        family = config.family_for_language(language)
        assert family is not None
        assert language in family.languages
        assert family.family_id
        assert family.scripts


def test_default_graph_adapter_metadata_is_offline_and_permissive() -> None:
    for edges in DEFAULT_FAMILY_TRANSFER_CONFIG.transfer_graph.values():
        for edge in edges:
            metadata = edge.adapter
            assert metadata.adapter_id
            assert metadata.backbone_model_id == DEFAULT_BACKBONE_MODEL_ID
            assert metadata.license in PERMISSIVE_ADAPTER_LICENSES
            assert metadata.provenance
            assert metadata.disclaimer == CLINICAL_ADAPTER_DISCLAIMER
            assert metadata.offline_runnable is True


def test_telugu_resolves_to_hindi_donor_with_adapter_metadata() -> None:
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


def test_romance_transfer_donors_are_ordered_deterministically() -> None:
    assert donor_languages_for("pt") == ("es", "fr", "it")
    assert donor_languages_for("it") == ("es", "fr")
    assert donor_languages_for("ro-RO") == ("it", "es", "fr")
    assert primary_donor_for("fr") is None


def test_unsupported_language_has_no_family_transfer_metadata() -> None:
    assert resolve_family_transfer("xx") is None
    assert resolve_pii_family_transfer_route("xx") is None


def test_registry_route_exposes_target_donor_and_adapter_metadata() -> None:
    route = resolve_pii_family_transfer_route("te_IN")

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


def test_registry_route_without_donor_stays_native() -> None:
    route = resolve_pii_family_transfer_route("zh-Hant")

    assert route is not None
    assert route.language == "zh"
    assert route.family_id == "sinitic"
    assert route.target_model_id == DEFAULT_PII_MODELS["zh"]
    assert route.backbone_model_id == DEFAULT_BACKBONE_MODEL_ID
    assert route.donor_language is None
    assert route.adapter_id is None
    assert route.mode == "native"


def test_family_router_prefers_an_available_target_adapter() -> None:
    telugu_adapter = _available_adapter("te")
    hindi_adapter = _available_adapter("hi")

    route = route_family_adapter(
        "te-IN",
        {"te": telugu_adapter, "hi": hindi_adapter},
    )

    assert route.target_language == "te"
    assert route.adapter_language == "te"
    assert route.family_id == "indic"
    assert route.backbone_model_id == telugu_adapter.backbone_model_id
    assert route.adapter is telugu_adapter
    assert route.mode == "target_adapter"
    assert route.fallback is None


def test_family_router_falls_back_from_telugu_to_hindi_with_scored_metadata() -> None:
    hindi_adapter = _available_adapter("hi")

    route = FamilyTransferRouter({"hi-IN": hindi_adapter}).route("te")

    assert route.target_language == "te"
    assert route.adapter_language == "hi"
    assert route.backbone_model_id == hindi_adapter.backbone_model_id
    assert route.adapter is hindi_adapter
    assert route.mode == "zero_shot_fallback"
    assert route.fallback is not None
    assert route.fallback.donor == "hi"
    assert route.fallback.target == "te"
    assert route.fallback.score == pytest.approx(0.80)
    assert "hi-to-te" in route.fallback.provenance
    assert hindi_adapter.provenance in route.fallback.provenance
    assert "configured expected_f1_floor" in route.fallback.provenance


def test_family_router_keeps_donor_language_on_its_direct_adapter() -> None:
    hindi_adapter = _available_adapter("hi")

    route = FamilyTransferRouter({"hi": hindi_adapter}).route("hi-IN")

    assert route.target_language == "hi"
    assert route.adapter_language == "hi"
    assert route.adapter is hindi_adapter
    assert route.mode == "target_adapter"
    assert route.fallback is None


def test_family_router_rejects_unsupported_and_unavailable_targets() -> None:
    router = FamilyTransferRouter({})

    with pytest.raises(
        UnsupportedFamilyTransferLanguageError,
        match="unsupported family-transfer language 'xx'",
    ):
        router.route("xx")

    with pytest.raises(
        FamilyTransferAdapterUnavailableError,
        match="no target or compatible donor adapter.*'te'",
    ):
        router.route("te")


def test_transfer_config_rejects_missing_donor_family() -> None:
    bad_edge = _edge(donor_language="xx")

    with pytest.raises(ValueError, match="donor 'xx' has no language family"):
        FamilyTransferConfig(
            families=DEFAULT_LANGUAGE_FAMILIES,
            transfer_graph={"te": (bad_edge,)},
        )


def test_transfer_edge_rejects_self_donor() -> None:
    with pytest.raises(ValueError, match="donor_language must differ"):
        _edge(target_language="te", donor_language="te")


def test_transfer_config_rejects_donor_cycles() -> None:
    hi_to_te = TransferEdge(
        target_language="hi",
        donor_language="te",
        family_id="indic",
        adapter=AdapterMetadata(
            adapter_id="synthetic-test/hi-cycle",
            provenance="Synthetic offline unit-test adapter metadata",
        ),
    )

    with pytest.raises(ValueError, match="contains a cycle"):
        FamilyTransferConfig(
            families=DEFAULT_LANGUAGE_FAMILIES,
            transfer_graph={
                "te": (_edge(adapter_id="synthetic-test/te-cycle"),),
                "hi": (hi_to_te,),
            },
        )


def test_transfer_config_rejects_nonpermissive_adapter_license() -> None:
    bad_edge = _edge(license_name="cc-by-nc-4.0")

    with pytest.raises(ValueError, match="not permissive"):
        FamilyTransferConfig(
            families=DEFAULT_LANGUAGE_FAMILIES,
            transfer_graph={"te": (bad_edge,)},
        )


def test_transfer_config_rejects_non_offline_adapter_metadata() -> None:
    bad_edge = _edge(offline_runnable=False)

    with pytest.raises(ValueError, match="offline-runnable"):
        FamilyTransferConfig(
            families=DEFAULT_LANGUAGE_FAMILIES,
            transfer_graph={"te": (bad_edge,)},
        )
