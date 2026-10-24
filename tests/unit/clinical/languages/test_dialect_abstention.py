"""Synthetic gates for dialect-aware fallback, ensemble and abstention."""

from __future__ import annotations

import json
import logging
import math
import re
import socket
from dataclasses import FrozenInstanceError, replace

import pytest

from openmed.clinical.languages.dialect_abstention import (
    CHAR_CLASSES,
    DIALECT_ACTIONS,
    DIALECT_LIMITATIONS,
    MIN_REFERENCE_DOCUMENTS,
    NGRAM_BUCKETS,
    DialectDecision,
    DialectFeatureProfile,
    DialectPolicy,
    DialectRouteError,
    ValidatedVariety,
    apply_dialect_policy,
    assess_dialect,
    build_validated_variety,
    enforce_dialect_decision,
    estimate_variety_distance,
    extract_dialect_features,
)
from openmed.core.language_router import LanguagePrediction, LanguageRouter

# Every document below is synthetic. Names and numbers are invented markers
# that let the privacy tests prove nothing from the input reaches a report.
DIGEST = "0" * 63 + "1"
SYNTHETIC_NAME = "Mustermannova"
SYNTHETIC_ID = "4711-0815-2342"

GERMAN_DOCS = (
    "Der Patient klagt seit drei Tagen über starke Kopfschmerzen und Übelkeit. "
    "Die Vitalwerte sind stabil.",
    "Die Patientin wurde zur weiteren Abklärung der Beschwerden stationär "
    "aufgenommen. Es besteht kein Fieber.",
    "Bei der körperlichen Untersuchung zeigte sich ein unauffälliger Befund. "
    "Die Medikation wird fortgeführt.",
    "Der Hausarzt überwies den Patienten wegen anhaltender Schmerzen im "
    "Brustbereich. Das EKG war ohne Befund.",
    "Die Wunde ist reizlos und die Fäden werden am zehnten Tag entfernt.",
    "Der Patient erhält eine Schmerztherapie und wird morgen entlassen.",
)
AUSTRIAN_DOCS = (
    "Der Patient wurde im Jänner im Spital aufgenommen und klagt über Schmerzen "
    "im Bauch. Die Vitalwerte sind stabil.",
    "Die Patientin ist heuer schon zweimal im Spital gewesen. Es besteht kein "
    "Fieber und die Wunde ist reizlos.",
    "Bei der Untersuchung zeigte sich ein unauffälliger Befund. Die Medikation "
    "wird fortgeführt und die Kontrolle erfolgt im Feber.",
    "Der Hausarzt überwies den Patienten wegen anhaltender Beschwerden. Das EKG "
    "war ohne Befund und der Patient wird morgen entlassen.",
)
ENGLISH_DOCS = (
    "The patient was admitted for further evaluation of the symptoms. There "
    "is no fever.",
    "Physical examination revealed no abnormal findings. The medication will "
    "be continued.",
    "The general practitioner referred the patient because of persistent chest "
    "pain. The ECG was normal.",
    "The wound is clean and the sutures will be removed on the tenth day.",
)

GERMAN_NOTE = (
    f"Die Patientin {SYNTHETIC_NAME} berichtet über zunehmende Schmerzen im "
    f"Bauch seit gestern Abend. Fallnummer {SYNTHETIC_ID}. Der Blutdruck ist "
    "leicht erhöht und die Untersuchung wird morgen fortgesetzt."
)
SWISS_DIALECT_NOTE = (
    f"D Patientin {SYNTHETIC_NAME} verzellt, dass si sit geschter Znacht meh "
    f"Buchweh het. Fallnummer {SYNTHETIC_ID}. De Bluetdruck isch chli höcher "
    "und d Untersuechig gaht morn witer."
)
ENGLISH_NOTE = (
    f"The patient {SYNTHETIC_NAME} reports increasing abdominal pain since "
    f"yesterday evening. Case {SYNTHETIC_ID}. Blood pressure is slightly "
    "elevated and the examination continues tomorrow."
)
CHINESE_NOTE = (
    "患者自昨晚以来腹痛加重。血压略有升高，检查将于明天继续进行。"
    "患者无发热，生命体征平稳，继续观察。"
)


def variety(variety_id, docs, locale, **overrides):
    """Build one synthetic validated variety."""

    arguments = {
        "locale": locale,
        "tasks": ("deidentification",),
        "domains": ("clinical_note",),
        "evidence_digest": DIGEST,
        "language_pack": None,
    }
    arguments.update(overrides)
    return build_validated_variety(variety_id, docs, **arguments)


@pytest.fixture(scope="module")
def german():
    return variety("de-de.clinical", GERMAN_DOCS, "de-DE", language_pack="de")


@pytest.fixture(scope="module")
def austrian():
    return variety("de-at.clinical", AUSTRIAN_DOCS, "de-AT", language_pack="de")


@pytest.fixture(scope="module")
def english():
    return variety("en-us.clinical", ENGLISH_DOCS, "en-US", language_pack="en")


@pytest.fixture(scope="module")
def multilingual():
    return variety(
        "mul.fallback",
        GERMAN_DOCS[:3] + ENGLISH_DOCS[:3] + AUSTRIAN_DOCS[:2],
        "mul",
    )


@pytest.fixture(scope="module")
def varieties(german, austrian, english, multilingual):
    return (german, austrian, english, multilingual)


@pytest.fixture(scope="module")
def router():
    return LanguageRouter(use_optional_lid=False)


def assess(text, varieties, router, **kwargs):
    kwargs.setdefault("task", "deidentification")
    kwargs.setdefault("declared_domain", "clinical_note")
    return assess_dialect(text, varieties, router=router, **kwargs)


class FixedLanguageIdentifier:
    """Offline fake language-ID backend with one fixed prediction."""

    name = "fake-lid"

    def __init__(self, language, confidence=0.97):
        self.language = language
        self.confidence = confidence

    def identify(self, text, candidates):
        if self.language not in candidates:
            return None
        return LanguagePrediction(self.language, self.confidence)


# ---------------------------------------------------------------------------
# Distance estimation


def test_validated_variety_is_accepted(varieties, router):
    decision = assess(GERMAN_NOTE, varieties, router, declared_locale="de-DE")

    assert decision.action == "accept"
    assert decision.selected == ("de-de.clinical",)
    assert decision.nearest_variety == "de-de.clinical"
    assert decision.dialect_distance <= DialectPolicy().accept_dialect_distance
    assert "dialect_shift" not in decision.limitations
    assert not decision.abstained
    assert not decision.requires_review


def test_unvalidated_dialect_is_shifted_and_abstains_by_default(varieties, router):
    decision = assess(SWISS_DIALECT_NOTE, varieties, router, declared_locale="de-CH")
    policy = DialectPolicy()

    assert policy.accept_dialect_distance < decision.dialect_distance
    assert decision.dialect_distance <= policy.abstain_dialect_distance
    assert decision.action == "abstain"
    assert decision.selected == ()
    assert "dialect_shift" in decision.limitations
    assert decision.abstained and decision.requires_review


def test_unvalidated_region_of_standard_text_is_not_silently_accepted(
    varieties, router
):
    validated = assess(GERMAN_NOTE, varieties, router, declared_locale="de-DE")
    swiss = assess(GERMAN_NOTE, varieties, router, declared_locale="de-CH")

    assert swiss.dialect_distance > validated.dialect_distance
    assert swiss.action == "abstain"
    assert "dialect_shift" in swiss.limitations


def test_dialect_distance_orders_language_and_script_evidence(german, english, router):
    german_profile = extract_dialect_features(
        GERMAN_NOTE, declared_locale="de-DE", router=router
    )
    dialect_profile = extract_dialect_features(
        SWISS_DIALECT_NOTE, declared_locale="de-CH", router=router
    )
    english_profile = extract_dialect_features(
        ENGLISH_NOTE, declared_locale="en-GB", router=router
    )
    chinese_profile = extract_dialect_features(
        CHINESE_NOTE, declared_locale="zh-CN", router=router
    )

    distances = [
        estimate_variety_distance(profile, german).dialect_distance
        for profile in (german_profile, dialect_profile, english_profile)
    ]
    assert distances == sorted(distances)
    to_english = estimate_variety_distance(english_profile, english)
    assert to_english.dialect_distance < distances[2]
    assert to_english.locale_distance == 0.5  # en-GB against en-US

    chinese = estimate_variety_distance(chinese_profile, german)
    assert chinese.script_distance == 1.0
    assert chinese.locale_distance == 1.0
    assert chinese.dialect_distance > 0.8
    for item in (to_english, chinese):
        for value in item.to_dict().values():
            if isinstance(value, float):
                assert 0.0 <= value <= 1.0


def test_different_language_and_script_are_out_of_range(varieties, router):
    for text, locale in ((ENGLISH_NOTE, "en-GB"), (CHINESE_NOTE, "zh-CN")):
        decision = assess(
            text,
            (varieties[0], varieties[1]),
            router,
            declared_locale=locale,
            policy=DialectPolicy(intermediate_action="ensemble"),
        )
        assert decision.action == "abstain"
        assert "dialect_out_of_range" in decision.limitations


def test_domain_distance_uses_declared_domain_and_shape(german, router):
    lab_panel = (
        "Hb 12,4 g/dl; Leuk 7,1 /nl; Thrombo 240 /nl; CRP 3,2 mg/l; "
        "Krea 0,9 mg/dl; Natrium 139; Kalium 4,1; Glukose 102 mg/dl."
    )
    note = extract_dialect_features(
        GERMAN_NOTE, declared_domain="clinical_note", router=router
    )
    undeclared = extract_dialect_features(GERMAN_NOTE, router=router)
    radiology = extract_dialect_features(
        GERMAN_NOTE, declared_domain="radiology_report", router=router
    )
    lab = extract_dialect_features(
        lab_panel, declared_domain="clinical_note", router=router
    )

    note_distance = estimate_variety_distance(note, german)
    assert note_distance.declared_domain_distance == 0.0
    assert estimate_variety_distance(undeclared, german).domain_distance > (
        note_distance.domain_distance
    )
    assert estimate_variety_distance(radiology, german).declared_domain_distance == 1.0
    lab_distance = estimate_variety_distance(lab, german)
    assert lab_distance.shape_distance > note_distance.shape_distance + 0.2


def test_unvalidated_domain_surfaces_and_blocks_acceptance(varieties, router):
    decision = assess(
        GERMAN_NOTE,
        varieties,
        router,
        declared_locale="de-DE",
        declared_domain="radiology_report",
    )

    assert decision.action == "abstain"
    assert {"domain_shift", "domain_unvalidated"} <= set(decision.limitations)

    undeclared = assess(
        GERMAN_NOTE, varieties, router, declared_locale="de-DE", declared_domain=None
    )
    assert undeclared.action == "accept"
    assert "domain_undeclared" in undeclared.limitations


# ---------------------------------------------------------------------------
# Declared policy: fallback, ensemble and abstention


def test_ensemble_policy_requires_nearest_validated_varieties(varieties, router):
    policy = DialectPolicy(policy_id="swiss-ensemble", intermediate_action="ensemble")
    decision = assess(
        SWISS_DIALECT_NOTE,
        varieties,
        router,
        declared_locale="de-CH",
        policy=policy,
    )

    assert decision.action == "ensemble"
    assert len(decision.selected) == 2
    assert set(decision.selected) <= {
        "de-de.clinical",
        "de-at.clinical",
        "mul.fallback",
    }
    assert decision.selected[0] == decision.nearest_variety
    assert {"dialect_shift", "ensemble_route"} <= set(decision.limitations)
    assert decision.requires_review
    assert decision.policy_id == "swiss-ensemble"
    assert decision.policy_digest == policy.digest


def test_ensemble_without_two_close_members_abstains(german, english, router):
    decision = assess(
        SWISS_DIALECT_NOTE,
        (german, english),
        router,
        declared_locale="de-CH",
        policy=DialectPolicy(intermediate_action="ensemble"),
    )

    assert decision.action == "abstain"
    assert "ensemble_insufficient_members" in decision.limitations


def test_fallback_policy_routes_to_declared_variety(varieties, router):
    policy = DialectPolicy(
        intermediate_action="fallback", fallback_variety="mul.fallback"
    )
    decision = assess(
        SWISS_DIALECT_NOTE, varieties, router, declared_locale="de-CH", policy=policy
    )

    assert decision.action == "fallback"
    assert decision.selected == ("mul.fallback",)
    assert {"dialect_shift", "fallback_route"} <= set(decision.limitations)


def test_fallback_must_be_validated_and_in_range(german, austrian, english, router):
    missing = DialectPolicy(
        intermediate_action="fallback", fallback_variety="mul.fallback"
    )
    decision = assess(
        SWISS_DIALECT_NOTE,
        (german, austrian),
        router,
        declared_locale="de-CH",
        policy=missing,
    )
    assert decision.action == "abstain"
    assert "fallback_unavailable" in decision.limitations

    distant = DialectPolicy(
        intermediate_action="fallback", fallback_variety="en-us.clinical"
    )
    decision = assess(
        SWISS_DIALECT_NOTE,
        (german, english),
        router,
        declared_locale="de-CH",
        policy=distant,
    )
    assert decision.action == "abstain"
    assert "fallback_out_of_range" in decision.limitations


def test_short_input_is_insufficient_evidence(varieties, router):
    short = "Der Blutdruck ist leicht erhöht."
    decision = assess(short, varieties, router, declared_locale="de-DE")
    assert decision.action == "abstain"
    assert "insufficient_evidence" in decision.limitations
    assert decision.requires_review

    policy = DialectPolicy(
        insufficient_evidence_action="fallback", fallback_variety="mul.fallback"
    )
    decision = assess(short, varieties, router, declared_locale="de-DE", policy=policy)
    assert decision.action == "fallback"
    assert decision.selected == ("mul.fallback",)


def test_empty_input_abstains_without_routing_claims(varieties, router):
    decision = assess("", varieties, router, declared_locale="de-DE")

    assert decision.action == "abstain"
    assert decision.profile["letter_count"] == 0
    assert decision.profile["routed_language"] == "und"
    assert {"insufficient_evidence", "language_unconfirmed"} <= set(
        decision.limitations
    )


def test_task_without_validation_abstains(varieties, router):
    decision = assess(
        GERMAN_NOTE, varieties, router, declared_locale="de-DE", task="extraction"
    )

    assert decision.action == "abstain"
    assert decision.nearest_variety is None
    assert decision.candidates == ()
    assert "no_validated_variety" in decision.limitations


def test_unregistered_language_pack_is_not_eligible(german, router):
    unregistered = replace(german, variety_id="qq.clinical", language_pack="qq")
    decision = assess(GERMAN_NOTE, (unregistered,), router, declared_locale="de-DE")

    assert decision.action == "abstain"
    assert {"no_validated_variety", "variety_pack_unregistered"} <= set(
        decision.limitations
    )
    profile = extract_dialect_features(GERMAN_NOTE, declared_locale="de-DE")
    unchecked = apply_dialect_policy(profile, (unregistered,), task="deidentification")
    assert unchecked.action == "accept"


def test_policy_band_is_the_stricter_of_dialect_and_domain(varieties, router):
    strict_domain = DialectPolicy(
        accept_domain_distance=0.0, abstain_domain_distance=0.0
    )
    decision = assess(
        GERMAN_NOTE, varieties, router, declared_locale="de-DE", policy=strict_domain
    )

    assert decision.action == "abstain"
    assert "domain_out_of_range" in decision.limitations
    assert "dialect_shift" not in decision.limitations


# ---------------------------------------------------------------------------
# Language evidence from deterministic routing


def test_latin_route_without_language_id_is_unconfirmed(varieties, router):
    decision = assess(GERMAN_NOTE, varieties, router)

    assert {"language_unconfirmed", "locale_undeclared"} <= set(decision.limitations)
    assert decision.profile["routing_decisive"] is False
    assert decision.candidates[0].locale_distance == 0.5


def test_decisive_route_confirms_language_and_detects_conflict(varieties):
    router = LanguageRouter(language_identifier=FixedLanguageIdentifier("de"))
    confirmed = assess(GERMAN_NOTE, varieties, router)

    assert confirmed.profile["routed_language"] == "de"
    assert confirmed.profile["routing_decisive"] is True
    assert "language_unconfirmed" not in confirmed.limitations
    assert confirmed.candidates[0].locale_distance == 0.25  # region unknown
    assert confirmed.action == "accept"

    conflict = assess(GERMAN_NOTE, varieties, router, declared_locale="fr-FR")
    assert "declared_locale_conflict" in conflict.limitations
    assert conflict.action == "abstain"


def test_mixed_and_unsupported_scripts_are_surfaced(varieties, router):
    mixed = assess(GERMAN_NOTE + " " + CHINESE_NOTE, varieties, router)
    assert "mixed_language" in mixed.limitations
    assert mixed.profile["script_shares"]["Han"] > 0.0
    assert mixed.requires_review

    # Georgian has no script block in the router; Myanmar has one but no pack.
    for foreign in ("პაციენტი " * 8, "လူနာ " * 8):
        unsupported = assess(GERMAN_NOTE + " " + foreign, varieties, router)
        assert "unsupported_script" in unsupported.limitations
        assert unsupported.profile["routing_decisive"] is False
        assert unsupported.requires_review


# ---------------------------------------------------------------------------
# Enforcement and result surfacing


def test_enforcement_fails_closed(varieties, router):
    accepted = assess(GERMAN_NOTE, varieties, router, declared_locale="de-DE")
    enforce_dialect_decision(accepted, ["de-de.clinical"])
    for executed in ([], ["de-at.clinical"], ["de-de.clinical", "de-de.clinical"]):
        with pytest.raises(DialectRouteError) as error:
            enforce_dialect_decision(accepted, executed)
        assert error.value.code == "dialect_route_mismatch"

    ensemble = assess(
        SWISS_DIALECT_NOTE,
        varieties,
        router,
        declared_locale="de-CH",
        policy=DialectPolicy(intermediate_action="ensemble"),
    )
    enforce_dialect_decision(ensemble, reversed(ensemble.selected))
    with pytest.raises(DialectRouteError, match="dialect_route_mismatch"):
        enforce_dialect_decision(ensemble, ensemble.selected[:1])

    abstained = assess(SWISS_DIALECT_NOTE, varieties, router, declared_locale="de-CH")
    with pytest.raises(DialectRouteError) as error:
        enforce_dialect_decision(abstained, [])
    assert error.value.code == "dialect_abstained"
    with pytest.raises(TypeError):
        enforce_dialect_decision(accepted, "de-de.clinical")


def test_decision_metadata_surfaces_limitations(varieties, router):
    decision = assess(SWISS_DIALECT_NOTE, varieties, router, declared_locale="de-CH")
    metadata = decision.to_metadata()["dialect_assessment"]

    assert metadata["action"] == "abstain"
    assert metadata["abstained"] is True
    assert metadata["requires_review"] is True
    assert metadata["limitations"] == list(decision.limitations)
    assert set(metadata["limitations"]) <= set(DIALECT_LIMITATIONS)
    assert "not a measurement" in metadata["disclaimer"]
    assert metadata["policy_digest"] == DialectPolicy().digest
    metadata["profile"]["script_shares"]["Latin"] = 0.0
    assert decision.to_dict()["profile"]["script_shares"]["Latin"] == 1.0
    assert set(DIALECT_ACTIONS) == {"accept", "fallback", "ensemble", "abstain"}


# ---------------------------------------------------------------------------
# Determinism, offline execution and privacy


def test_assessment_is_deterministic_and_order_independent(varieties, router):
    policy = DialectPolicy(intermediate_action="ensemble")
    first = assess(
        SWISS_DIALECT_NOTE, varieties, router, declared_locale="de_ch", policy=policy
    )
    second = assess(
        SWISS_DIALECT_NOTE,
        tuple(reversed(varieties)),
        LanguageRouter(use_optional_lid=False),
        declared_locale="DE-ch",
        policy=DialectPolicy(intermediate_action="ensemble"),
    )

    assert first.to_json() == second.to_json()
    assert first.profile["declared_locale"] == "de-CH"
    assert json.loads(first.to_json())["schema_version"] == 1


PINNED_BUCKETS = [98, 132, 183, 360, 480]


def test_hashed_ngram_scheme_is_pinned():
    profile = extract_dialect_features("ab")

    # Bigrams " a", "ab", "b " and trigrams " ab", "ab " in keyed buckets. The
    # scheme is pinned so serialized references stay comparable across
    # releases and interpreter hash seeds.
    assert [bucket for bucket, _share in profile.ngram_histogram] == PINNED_BUCKETS
    assert all(0 <= bucket < NGRAM_BUCKETS for bucket in PINNED_BUCKETS)
    assert math.isclose(sum(share for _bucket, share in profile.ngram_histogram), 1.0)


def test_numbers_and_punctuation_do_not_contribute_ngrams():
    with_numbers = extract_dialect_features(f"Fall {SYNTHETIC_ID}: Befund!")
    without_numbers = extract_dialect_features("Fall Befund")

    assert with_numbers.ngram_histogram == without_numbers.ngram_histogram
    assert with_numbers.char_class_shares != without_numbers.char_class_shares


def test_assessment_makes_no_network_calls(varieties, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    decision = assess_dialect(
        GERMAN_NOTE,
        varieties,
        task="deidentification",
        declared_locale="de-DE",
        declared_domain="clinical_note",
    )
    assert decision.action == "accept"


def _sensitive_words(*texts):
    return {
        word
        for text in texts
        for word in re.findall(r"[^\W\d_]{5,}", text)
        if word.casefold() not in {"clinical"}
    }


def test_reports_and_logs_contain_no_input_text(varieties, router, caplog):
    caplog.set_level(logging.DEBUG)
    decision = assess(
        SWISS_DIALECT_NOTE,
        varieties,
        router,
        declared_locale="de-CH",
        policy=DialectPolicy(intermediate_action="ensemble"),
    )
    profile = extract_dialect_features(SWISS_DIALECT_NOTE, router=router)
    rendered = "\n".join(
        (
            decision.to_json(),
            repr(decision),
            str(decision.to_metadata()),
            json.dumps(profile.summary()),
            repr(profile),
        )
    )

    for value in (SYNTHETIC_NAME, SYNTHETIC_ID, *_sensitive_words(SWISS_DIALECT_NOTE)):
        assert value not in rendered
    assert "ngram" not in json.dumps(profile.summary())
    assert caplog.records == []


def test_references_contain_aggregates_only(german):
    payload = json.dumps(german.to_dict(), ensure_ascii=False)

    for word in _sensitive_words(*GERMAN_DOCS):
        assert word not in payload
    assert german.document_count == len(GERMAN_DOCS)
    assert ValidatedVariety.from_dict(json.loads(payload)) == german
    assert "ngram_centroid" not in repr(german)


SENSITIVE = f"/Users/{SYNTHETIC_NAME}/{SYNTHETIC_ID}"


@pytest.mark.parametrize(
    "call",
    [
        lambda: extract_dialect_features("text", declared_locale=SENSITIVE),
        lambda: extract_dialect_features("text", declared_domain=SENSITIVE),
        lambda: extract_dialect_features(SENSITIVE * 100_000),
        lambda: variety(SENSITIVE, GERMAN_DOCS, "de-DE"),
        lambda: variety("de.clinical", GERMAN_DOCS, SENSITIVE),
        lambda: variety("de.clinical", GERMAN_DOCS, "de-DE", tasks=(SENSITIVE,)),
        lambda: variety("de.clinical", GERMAN_DOCS, "de-DE", evidence_digest=SENSITIVE),
        lambda: variety("de.clinical", GERMAN_DOCS[:2], "de-DE"),
        lambda: variety("de.clinical", (SENSITIVE,) * 3, "de-DE"),
        lambda: DialectPolicy(policy_id=SENSITIVE),
        lambda: DialectPolicy(fallback_variety=SENSITIVE),
        lambda: assess_dialect(GERMAN_NOTE, (), task=SENSITIVE),
        lambda: ValidatedVariety.from_dict({"schema_version": 1, "locale": SENSITIVE}),
    ],
)
def test_errors_never_echo_input_values(call):
    with pytest.raises(ValueError) as error:
        call()

    assert SYNTHETIC_NAME not in str(error.value)
    assert SYNTHETIC_ID not in str(error.value)


# ---------------------------------------------------------------------------
# Declaration validation and bounds


@pytest.mark.parametrize(
    "overrides",
    [
        {"accept_dialect_distance": 0.5, "abstain_dialect_distance": 0.4},
        {"accept_domain_distance": 0.9, "abstain_domain_distance": 0.5},
        {"accept_dialect_distance": math.nan},
        {"abstain_dialect_distance": 1.5},
        {"accept_dialect_distance": True},
        {"intermediate_action": "accept"},
        {"intermediate_action": "guess"},
        {"insufficient_evidence_action": "ensemble"},
        {"intermediate_action": "fallback"},
        {"insufficient_evidence_action": "fallback"},
        {"ensemble_size": 1},
        {"ensemble_size": 9},
        {"min_letters": 0},
        {"min_letters": 2.5},
    ],
)
def test_invalid_policy_is_rejected(overrides):
    with pytest.raises(ValueError):
        DialectPolicy(**overrides)


def test_policy_digest_tracks_the_declaration():
    assert DialectPolicy().digest == DialectPolicy().digest
    assert DialectPolicy(min_letters=41).digest != DialectPolicy().digest
    assert re.fullmatch(r"[0-9a-f]{64}", DialectPolicy().digest)


def test_declarations_are_frozen_and_bounded(german):
    with pytest.raises(FrozenInstanceError):
        german.locale = "de-CH"
    with pytest.raises(FrozenInstanceError):
        DialectPolicy().intermediate_action = "ensemble"
    with pytest.raises(ValueError):
        replace(german, document_count=MIN_REFERENCE_DOCUMENTS - 1)
    with pytest.raises(ValueError):
        replace(german, char_class_shares=(1.0,) * len(CHAR_CLASSES))
    with pytest.raises(ValueError):
        replace(german, ngram_centroid=((NGRAM_BUCKETS, 1.0),))
    with pytest.raises(ValueError):
        apply_dialect_policy(
            extract_dialect_features("text"),
            (german, german),
            task="deidentification",
        )


def test_unbounded_document_streams_are_consumed_with_a_hard_bound():
    consumed = []

    def endless():
        while True:
            consumed.append(1)
            yield "Synthetische Notiz."

    with pytest.raises(ValueError):
        variety("de.clinical", endless(), "de-DE")
    assert len(consumed) == 10_001


def test_type_contracts():
    with pytest.raises(TypeError):
        extract_dialect_features(b"bytes")
    with pytest.raises(TypeError):
        extract_dialect_features("text", router=object())
    with pytest.raises(TypeError):
        estimate_variety_distance(object(), object())
    profile = extract_dialect_features("text")
    assert isinstance(profile, DialectFeatureProfile)
    with pytest.raises(TypeError):
        apply_dialect_policy(profile, (), task="deidentification", policy=object())
    with pytest.raises(TypeError):
        apply_dialect_policy(
            profile, (), task="deidentification", registered_packs="de"
        )
    decision = apply_dialect_policy(profile, (), task="deidentification")
    assert isinstance(decision, DialectDecision)
    assert decision.action == "abstain"
