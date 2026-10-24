"""Compare model language decisions with the deterministic language router.

The evaluation builds a synthetic multilingual set from the language-pack
golden fixtures and asks five fixed-option questions:

* ``language``: which language a note, or one annotated run of a note, is in,
  out of every evaluated language (open set);
* ``language_in_script``: the same question restricted to the languages
  natively written in the scripts the note contains, which is how a language
  identifier is consulted inside the router;
* ``script``: which writing script a single-script note or run uses;
* ``dialect``: which regional variety (locale) of its language a note carries;
* ``code_switch``: whether a note mixes languages.

Every sample is answered twice. The deterministic baseline is the stdlib
:class:`~openmed.core.language_router.LanguageRouter` (optional CLD2 disabled),
:func:`~openmed.core.script_detect.detect_script` for scripts, and the language
pack's default locale for dialects. The model side runs through
:func:`openmed.structured.decision.decide` with any local
:class:`~openmed.structured.decision.DecisionBackend`, such as
:class:`~openmed.ner.families.gliner2_decision.GLiNER2DecisionBackend`, so the
contract's calibrated abstention applies unchanged.

Gold labels come from fixture declarations only: the record ``language``, the
record ``metadata.locale``, annotated runs, declared code-mixing flags and the
script a language pack declares. A record is used for a script or code-switch
question only when it is unambiguous for that question.

Reports contain counts, rates, controlled codes, model identity and a dataset
digest. Sample text stays in memory and is never serialized. This module
evaluates routing signals only: it changes no routing default, activates no
language pack, and its results are assistive review evidence, not a clinical
qualification.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Optional

from openmed.core.language_pack_catalog import LANG_TO_LOCALE, LANGUAGE_PACK_ADAPTERS
from openmed.core.language_router import LanguageRouter
from openmed.core.script_detect import detect_script, segment_by_script
from openmed.structured.decision import (
    DEFAULT_CALIBRATION_PROFILE,
    MAX_TIMEOUT_MS,
    DecisionBackend,
    DecisionCalibrationProfile,
    DecisionMode,
    DecisionRequest,
    DecisionState,
    decide,
)

SCHEMA_VERSION: Final = "openmed.eval.language_decisions.v1"

GLINER2_LANGUAGE_DECISION_MODEL: Final = "fastino/GLiNER2.5-multi-Decide"
"""Checkpoint evaluated by the committed benchmark summary."""

GLINER2_LANGUAGE_DECISION_REVISION: Final = "e173bca1f0c4217d7a55b3b0c705d5ece8a0218d"
"""Pinned Hugging Face commit of :data:`GLINER2_LANGUAGE_DECISION_MODEL`."""

LANGUAGE: Final = "language"
LANGUAGE_IN_SCRIPT: Final = "language_in_script"
SCRIPT: Final = "script"
DIALECT: Final = "dialect"
CODE_SWITCH: Final = "code_switch"
DECISION_KINDS: Final = (LANGUAGE, LANGUAGE_IN_SCRIPT, SCRIPT, DIALECT, CODE_SWITCH)

DECISION_TASKS: Final[Mapping[str, str]] = {
    LANGUAGE: "language",
    LANGUAGE_IN_SCRIPT: "language",
    SCRIPT: "writing_script",
    DIALECT: "regional_variety",
    CODE_SWITCH: "mixes_languages",
}
"""Classification task names sent to GLiNER2, one per decision kind."""

POLICIES: Final = (
    "router_always",
    "router_evidence",
    "model_only",
    "model_agreement",
    "model_fallback",
)
"""Accept/abstain policies compared for every decision kind.

* ``router_always``: accept every deterministic answer (current behavior).
* ``router_evidence``: accept the deterministic answer only when script,
  marker, context or cue evidence decided it; abstain on pack-priority
  fallbacks and on default-locale dialect answers.
* ``model_only``: accept the model answer when the decision contract does not
  abstain.
* ``model_agreement``: accept the deterministic answer only when the model
  answers confidently and agrees.
* ``model_fallback``: accept evidence-backed deterministic answers, otherwise
  the model's confident answer, otherwise abstain.
"""

ROUTER_EVIDENCE_CONFIDENCE: Final = 0.99
"""Run confidence the router assigns to evidence-backed routing decisions."""

CONFIDENCE_SWEEP: Final = (0.5, 0.7, 0.9)
"""Model confidence thresholds reported as a coverage/accuracy sweep."""

MULTIPLE_LANGUAGES: Final = "mul"
"""ISO 639-2 code used to group samples that span several languages."""

RESOURCE_TIERS: Final = ("dedicated_model", "shared_model", "no_pack", "multiple")
"""Resource tiers derived from the language-pack registry.

``dedicated_model`` packs ship a model no other pack uses, ``shared_model``
packs route to a model shared with other packs, ``no_pack`` languages have no
registered pack (for example national-ID-only languages), and ``multiple``
groups samples spanning several languages. The low-resource slice is
``shared_model`` plus ``no_pack``.
"""

LOW_RESOURCE_TIERS: Final = frozenset({"shared_model", "no_pack"})

GROUP_ROW_COLUMNS: Final = (
    "samples",
    "router_correct",
    "router_decisive",
    "model_top1_correct",
    "model_covered",
    "model_covered_correct",
    "model_agreement_accepted",
    "model_agreement_correct",
    "model_fallback_accepted",
    "model_fallback_correct",
)
"""Columns of the per-tier and per-mix count tables."""

LANGUAGE_ROW_COLUMNS: Final = ("resource_tier", *GROUP_ROW_COLUMNS)
"""Columns of the per-language count table."""

_EXACT_COLUMNS: Final = frozenset(
    {"resource_tier", "samples", "router_correct", "router_decisive"}
)

MIX_DECLARED: Final = "declared"
MIX_UNDECLARED: Final = "undeclared"
MIX_NONE: Final = "none"
MIX_CATEGORIES: Final = (MIX_NONE, MIX_DECLARED, MIX_UNDECLARED)

ADVISORY: Final = (
    "Language decisions are routing evidence on synthetic fixtures, not a "
    "measurement of extraction or de-identification accuracy and not a "
    "clinical qualification."
)

LANGUAGE_NAMES: Final[Mapping[str, str]] = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "ig": "Igbo",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lv": "Latvian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ms": "Malay",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "or": "Odia",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "sk": "Slovak",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "xh": "Xhosa",
    "yo": "Yoruba",
    "zh": "Chinese",
    "zu": "Zulu",
}
"""English language names used as model-facing option labels."""

_LATIN: Final = ("Latin",)
LANGUAGE_SCRIPTS: Final[Mapping[str, tuple[str, ...]]] = {
    "af": _LATIN,
    "am": ("Ethiopic",),
    "ar": ("Arabic",),
    "as": ("Bengali",),
    "bg": ("Cyrillic",),
    "bn": ("Bengali",),
    "cs": _LATIN,
    "da": _LATIN,
    "de": _LATIN,
    "el": ("Greek",),
    "en": _LATIN,
    "es": _LATIN,
    "et": _LATIN,
    "fa": ("Arabic",),
    "fi": _LATIN,
    "fr": _LATIN,
    "gu": ("Gujarati",),
    "ha": _LATIN,
    "he": ("Hebrew",),
    "hi": ("Devanagari",),
    "hr": _LATIN,
    "hu": _LATIN,
    "id": _LATIN,
    "ig": _LATIN,
    "it": _LATIN,
    "ja": ("Han", "Hiragana/Katakana"),
    "kn": ("Kannada",),
    "ko": ("Hangul",),
    "lv": _LATIN,
    "ml": ("Malayalam",),
    "mr": ("Devanagari",),
    "ms": _LATIN,
    "ne": ("Devanagari",),
    "nl": _LATIN,
    "no": _LATIN,
    "or": ("Odia",),
    "pa": ("Gurmukhi",),
    "pl": _LATIN,
    "pt": _LATIN,
    "ro": _LATIN,
    "ru": ("Cyrillic",),
    "rw": _LATIN,
    "sk": _LATIN,
    "sr": ("Cyrillic", "Latin"),
    "sv": _LATIN,
    "sw": _LATIN,
    "ta": ("Tamil",),
    "te": ("Telugu",),
    "th": ("Thai",),
    "tl": _LATIN,
    "tr": _LATIN,
    "uk": ("Cyrillic",),
    "ur": ("Arabic",),
    "vi": _LATIN,
    "xh": _LATIN,
    "yo": _LATIN,
    "zh": ("Han",),
    "zu": _LATIN,
}
"""Scripts each language is natively written in.

Used only to pick ``language_in_script`` candidates. Pack script
declarations are not used here because some packs list fallback scripts (the
English pack also accepts Cyrillic and Greek runs).
"""

SCRIPT_LABELS: Final[Mapping[str, str]] = {
    "Arabic": "Arabic script",
    "Bengali": "Bengali script",
    "Cyrillic": "Cyrillic script",
    "Devanagari": "Devanagari script",
    "Ethiopic": "Ethiopic script",
    "Greek": "Greek script",
    "Gujarati": "Gujarati script",
    "Gurmukhi": "Gurmukhi script",
    "Han": "Han script",
    "Hangul": "Hangul script",
    "Hebrew": "Hebrew script",
    "Hiragana/Katakana": "Kana script",
    "Kannada": "Kannada script",
    "Latin": "Latin script",
    "Malayalam": "Malayalam script",
    "Odia": "Odia script",
    "Tamil": "Tamil script",
    "Telugu": "Telugu script",
    "Thai": "Thai script",
}
"""Model-facing labels for the scripts that registered packs declare."""

LOCALE_LABELS: Final[Mapping[str, str]] = {
    "de_AT": "Austrian German",
    "de_CH": "Swiss German",
    "de_DE": "German from Germany",
    "en_AU": "Australian English",
    "en_CA": "Canadian English",
    "en_GB": "British English",
    "en_IE": "Irish English",
    "en_US": "American English",
    "es_ES": "Spanish from Spain",
    "es_MX": "Mexican Spanish",
    "fr_BE": "Belgian French",
    "fr_CA": "Canadian French",
    "fr_CH": "Swiss French",
    "fr_FR": "French from France",
    "nl_BE": "Belgian Dutch",
    "nl_NL": "Dutch from the Netherlands",
    "pt_BR": "Brazilian Portuguese",
    "pt_PT": "European Portuguese",
}
"""Model-facing labels for regional varieties."""

BOOLEAN_OPTIONS: Final = ("yes", "no")

_GOLDEN_DIR: Final = Path(__file__).resolve().parent / "golden"
_I18N_DIR: Final = "fixtures/i18n"
_ALL_SCRIPT_FIXTURE: Final = "all_script_routing.jsonl"
_REGIONAL_FIXTURES: Final = ("be.jsonl", "ch.jsonl", "es_mx.jsonl", "ie.jsonl")
_CODE_MIXED_FIXTURE: Final = "fixtures/code_mixed_hinglish.jsonl"
_CODE_MIXED_LANGUAGE: Final = "hi"
_CODE_MIXED_TOKEN_LANGUAGES: Final = frozenset({"en", _CODE_MIXED_LANGUAGE})
_MIXED_KIND_MARKERS: Final = ("code-mixed", "code mixed", "code-switched")
_RATE_DIGITS: Final = 4

# First word of a letter's Unicode name, folded to one family per writing
# system. Families outside this table count as ``OTHER`` so unusual letters
# make a record ambiguous instead of silently single-script.
_LETTER_FAMILIES: Final[Mapping[str, str]] = {
    "ARABIC": "ARABIC",
    "BENGALI": "BENGALI",
    "CJK": "CJK",
    "CYRILLIC": "CYRILLIC",
    "DEVANAGARI": "DEVANAGARI",
    "ETHIOPIC": "ETHIOPIC",
    "GREEK": "GREEK",
    "GUJARATI": "GUJARATI",
    "GURMUKHI": "GURMUKHI",
    "HANGUL": "HANGUL",
    "HEBREW": "HEBREW",
    "HIRAGANA": "CJK",
    "IDEOGRAPHIC": "CJK",
    "KANNADA": "KANNADA",
    "KATAKANA": "CJK",
    "KATAKANA-HIRAGANA": "CJK",
    "LATIN": "LATIN",
    "MALAYALAM": "MALAYALAM",
    "ORIYA": "ORIYA",
    "TAMIL": "TAMIL",
    "TELUGU": "TELUGU",
    "THAI": "THAI",
}


class LanguageDecisionEvalError(ValueError):
    """Raised for invalid samples, fixtures or evaluation inputs.

    Messages carry controlled identifiers only, never sample text.
    """


@dataclass(frozen=True)
class LanguageDecisionSample:
    """One synthetic decision question with its gold answer.

    Attributes:
        sample_id: Unique identifier built from the fixture record id.
        kind: One of :data:`DECISION_KINDS`.
        language: Fixture language used for per-language grouping, or
            :data:`MULTIPLE_LANGUAGES`.
        gold: Gold option code.
        options: Option codes; language codes, script names, locales or
            ``yes``/``no``.
        source: Fixture path relative to the golden directory.
        start: Start offset of :attr:`text` in the fixture record text.
        end: End offset of :attr:`text` in the fixture record text.
        mix: Code-mixing category, one of :data:`MIX_CATEGORIES`.
        text: Synthetic input text. Never serialized.
    """

    sample_id: str
    kind: str
    language: str
    gold: str
    options: tuple[str, ...]
    source: str
    start: int
    end: int
    mix: str
    text: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.kind not in DECISION_KINDS:
            raise LanguageDecisionEvalError("unknown decision kind")
        if self.mix not in MIX_CATEGORIES:
            raise LanguageDecisionEvalError("unknown code-mixing category")
        if len(self.options) < 2 or len(set(self.options)) != len(self.options):
            raise LanguageDecisionEvalError(
                f"{self.sample_id} needs two or more distinct options"
            )
        if self.gold not in self.options:
            raise LanguageDecisionEvalError(
                f"{self.sample_id} gold answer is not an option"
            )
        if not isinstance(self.text, str) or not self.text.strip():
            raise LanguageDecisionEvalError(f"{self.sample_id} has no input text")
        if not 0 <= self.start < self.end or self.end - self.start != len(self.text):
            raise LanguageDecisionEvalError(f"{self.sample_id} has invalid offsets")

    @property
    def text_digest(self) -> str:
        """Return the SHA-256 digest of the sample text."""

        return _sha256(self.text)

    def manifest_entry(self) -> dict[str, Any]:
        """Return a value-free manifest entry used for the dataset digest."""

        return {
            "end": self.end,
            "gold": self.gold,
            "kind": self.kind,
            "language": self.language,
            "mix": self.mix,
            "options": list(self.options),
            "sample_id": self.sample_id,
            "source": self.source,
            "start": self.start,
            "text_sha256": self.text_digest,
        }


@dataclass(frozen=True)
class RouterOutcome:
    """Deterministic answer and whether evidence, not a fallback, decided it."""

    prediction: str
    decisive: bool


@dataclass(frozen=True)
class ModelOutcome:
    """Decision-contract outcome mapped back to option codes.

    ``top1`` is the highest-ranked option even when the contract abstained;
    ``choice`` is set only for a successful decision.
    """

    state: str
    choice: Optional[str]
    top1: Optional[str]
    confidence: Optional[float]
    margin: Optional[float]

    @property
    def covered(self) -> bool:
        """Return whether the decision contract answered without abstaining."""

        return self.state == DecisionState.SUCCESS.value


@dataclass(frozen=True)
class LanguageDecisionRecord:
    """Value-free per-sample outcome kept in memory for aggregation."""

    sample_id: str
    kind: str
    language: str
    resource_tier: str
    mix: str
    gold: str
    router: RouterOutcome
    model: ModelOutcome

    def policy(self, name: str) -> tuple[bool, Optional[str]]:
        """Return whether a policy accepts an answer for this sample.

        Args:
            name: One of :data:`POLICIES`.

        Returns:
            ``(accepted, answer)``; ``answer`` is the option code the policy
            would act on, or ``None`` when it has none.

        Raises:
            LanguageDecisionEvalError: If the policy is unknown.
        """

        router, model = self.router, self.model
        if name == "router_always":
            return True, router.prediction
        if name == "router_evidence":
            return router.decisive, router.prediction
        if name == "model_only":
            return model.covered, model.choice
        if name == "model_agreement":
            agrees = model.covered and model.choice == router.prediction
            return agrees, router.prediction
        if name == "model_fallback":
            if router.decisive:
                return True, router.prediction
            return model.covered, model.choice
        raise LanguageDecisionEvalError("unknown policy")


@dataclass(frozen=True)
class LanguageDecisionEvaluation:
    """Per-sample records plus the identities needed to report them."""

    dataset_digest: str
    sources: tuple[str, ...]
    records: tuple[LanguageDecisionRecord, ...]
    backends: Mapping[str, Mapping[str, Any]]
    tasks: Mapping[str, str]
    calibration: Mapping[str, Any]
    lid_backend: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the aggregate, value-free report."""

        return build_language_decision_report(self)


# ---------------------------------------------------------------------------
# Evaluation set
# ---------------------------------------------------------------------------


def resource_tier(language: str) -> str:
    """Return the :data:`RESOURCE_TIERS` entry for a language code.

    Args:
        language: Language code, or :data:`MULTIPLE_LANGUAGES`.

    Returns:
        ``dedicated_model``, ``shared_model``, ``no_pack`` or ``multiple``.
    """

    if language == MULTIPLE_LANGUAGES:
        return "multiple"
    packs = tuple(LANGUAGE_PACK_ADAPTERS.registry.iter_packs())
    pack = next((item for item in packs if item.code == language), None)
    if pack is None:
        return "no_pack"
    sharing = sum(1 for item in packs if item.default_model == pack.default_model)
    return "dedicated_model" if sharing == 1 else "shared_model"


def build_language_decision_samples(
    golden_dir: Optional[Path] = None,
) -> tuple[LanguageDecisionSample, ...]:
    """Build the synthetic evaluation set from the language-pack fixtures.

    Sources are every ``fixtures/i18n/*.jsonl`` file, the regional checksum
    fixtures (``be``, ``ch``, ``es_mx``, ``ie``) and the code-mixed Hinglish
    fixture. The build is deterministic: the same fixtures always produce the
    same samples in the same order.

    Args:
        golden_dir: Golden fixture root. Defaults to the packaged
            ``openmed/eval/golden`` directory.

    Returns:
        Samples ordered by kind, then source, then record order.

    Raises:
        LanguageDecisionEvalError: If fixtures are malformed or sample ids
            collide.
    """

    root = Path(golden_dir) if golden_dir is not None else _GOLDEN_DIR
    records = _language_records(root)
    packs = tuple(LANGUAGE_PACK_ADAPTERS.registry.iter_packs())
    language_options = tuple(
        sorted({pack.code for pack in packs} | {item.language for item in records})
    )
    script_options = tuple(
        sorted(
            {
                script
                for pack in packs
                for script in pack.scripts
                if script in SCRIPT_LABELS
            }
        )
    )

    samples: list[LanguageDecisionSample] = []
    language_samples = _language_samples(root, records, language_options)
    samples.extend(language_samples)
    samples.extend(_language_in_script_samples(language_samples))
    samples.extend(_script_samples(root, records, script_options))
    samples.extend(_dialect_samples(records))
    samples.extend(_code_switch_samples(root, records))

    seen: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            raise LanguageDecisionEvalError(f"duplicate sample id {sample.sample_id}")
        seen.add(sample.sample_id)
    return tuple(samples)


def dataset_digest(samples: Sequence[LanguageDecisionSample]) -> str:
    """Return a SHA-256 digest of the value-free sample manifest.

    Args:
        samples: Evaluation samples in order.

    Returns:
        ``sha256:<hex>`` over every sample's manifest entry, which pins ids,
        offsets, gold answers, options and a digest of each text.
    """

    manifest = [sample.manifest_entry() for sample in samples]
    return _sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")))


def option_labels(sample: LanguageDecisionSample) -> tuple[str, ...]:
    """Return the model-facing labels for a sample's option codes.

    Args:
        sample: Evaluation sample.

    Returns:
        One label per option, in option order: English language names, script
        labels, regional-variety names, or ``yes``/``no``.

    Raises:
        LanguageDecisionEvalError: If an option has no label.
    """

    if sample.kind == CODE_SWITCH:
        return sample.options
    table = {
        LANGUAGE: LANGUAGE_NAMES,
        LANGUAGE_IN_SCRIPT: LANGUAGE_NAMES,
        SCRIPT: SCRIPT_LABELS,
        DIALECT: LOCALE_LABELS,
    }[sample.kind]
    missing = [code for code in sample.options if code not in table]
    if missing:
        raise LanguageDecisionEvalError(
            f"{sample.kind} options have no label: {', '.join(sorted(missing))}"
        )
    return tuple(table[code] for code in sample.options)


@dataclass(frozen=True)
class _FixtureRecord:
    source: str
    record_id: str
    language: str
    locale: Optional[str]
    text: str = field(repr=False)
    declared_mix: bool = False
    declared_scripts: tuple[str, ...] = ()


def _language_records(root: Path) -> tuple[_FixtureRecord, ...]:
    paths = sorted((root / _I18N_DIR).glob("*.jsonl"))
    paths.extend(root / name for name in _REGIONAL_FIXTURES)
    records: list[_FixtureRecord] = []
    for path in paths:
        if path.name == _ALL_SCRIPT_FIXTURE:
            continue
        source = path.relative_to(root).as_posix()
        for payload in _read_jsonl(path, source):
            language = payload.get("language")
            text = payload.get("text")
            if not isinstance(language, str) or not isinstance(text, str):
                continue
            metadata = payload.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            records.append(
                _FixtureRecord(
                    source=source,
                    record_id=_record_id(payload, source),
                    language=language,
                    locale=_optional_text(metadata.get("locale")),
                    text=text,
                    declared_mix=_declares_code_mixing(payload, metadata),
                    declared_scripts=_declared_scripts(payload, metadata),
                )
            )
    return tuple(records)


def _language_samples(
    root: Path,
    records: Sequence[_FixtureRecord],
    options: tuple[str, ...],
) -> list[LanguageDecisionSample]:
    samples = [
        LanguageDecisionSample(
            sample_id=f"{LANGUAGE}:{record.record_id}",
            kind=LANGUAGE,
            language=record.language,
            gold=record.language,
            options=options,
            source=record.source,
            start=0,
            end=len(record.text),
            mix=_mix_category(record),
            text=record.text,
        )
        for record in records
    ]
    for record_id, source, text, index, run in _annotated_runs(root):
        language = run.get("language")
        if not isinstance(language, str) or language not in options:
            continue
        samples.append(
            _run_sample(
                LANGUAGE, record_id, source, text, index, run, language, options
            )
        )
    return samples


def _language_in_script_samples(
    language_samples: Sequence[LanguageDecisionSample],
) -> list[LanguageDecisionSample]:
    """Restrict language samples to candidates written in the text's scripts.

    A sample is kept only when the gold language is among its candidates and
    there are at least two. Romanized text in a language natively written in
    another script, such as Hindi in Latin letters, is therefore excluded:
    script-conditioned candidates cannot reach it.
    """

    samples: list[LanguageDecisionSample] = []
    for sample in language_samples:
        scripts = _letter_scripts(sample.text)
        options = tuple(
            code
            for code in sample.options
            if scripts.intersection(LANGUAGE_SCRIPTS.get(code, ()))
        )
        if sample.gold not in options or len(options) < 2:
            continue
        samples.append(
            LanguageDecisionSample(
                sample_id=LANGUAGE_IN_SCRIPT + sample.sample_id[len(LANGUAGE) :],
                kind=LANGUAGE_IN_SCRIPT,
                language=sample.language,
                gold=sample.gold,
                options=options,
                source=sample.source,
                start=sample.start,
                end=sample.end,
                mix=sample.mix,
                text=sample.text,
            )
        )
    return samples


def _letter_scripts(text: str) -> frozenset[str]:
    return frozenset(
        script
        for start, end, script in segment_by_script(text)
        if any(char.isalpha() for char in text[start:end])
    )


def _script_samples(
    root: Path,
    records: Sequence[_FixtureRecord],
    options: tuple[str, ...],
) -> list[LanguageDecisionSample]:
    pack_scripts = {
        pack.code: pack.scripts for pack in LANGUAGE_PACK_ADAPTERS.registry.iter_packs()
    }
    samples: list[LanguageDecisionSample] = []
    for record in records:
        if record.declared_mix or len(_letter_families(record.text)) != 1:
            continue
        declared = record.declared_scripts or pack_scripts.get(record.language, ())
        if len(declared) != 1 or declared[0] not in options:
            continue
        samples.append(
            LanguageDecisionSample(
                sample_id=f"{SCRIPT}:{record.record_id}",
                kind=SCRIPT,
                language=record.language,
                gold=declared[0],
                options=options,
                source=record.source,
                start=0,
                end=len(record.text),
                mix=MIX_NONE,
                text=record.text,
            )
        )
    for record_id, source, text, index, run in _annotated_runs(root):
        script = run.get("script")
        language = run.get("language")
        if script not in options or not isinstance(language, str):
            continue
        samples.append(
            _run_sample(SCRIPT, record_id, source, text, index, run, script, options)
        )
    return samples


def _dialect_samples(records: Sequence[_FixtureRecord]) -> list[LanguageDecisionSample]:
    observed: dict[str, set[str]] = {}
    for record in records:
        if _locale_language(record.locale) == record.language:
            observed.setdefault(record.language, set()).add(str(record.locale))
    samples: list[LanguageDecisionSample] = []
    for record in records:
        if _locale_language(record.locale) != record.language:
            continue
        default = LANG_TO_LOCALE.get(record.language)
        if default is None or _locale_language(default) != record.language:
            continue
        options = tuple(sorted(observed[record.language] | {default}))
        if len(options) < 2:
            continue
        samples.append(
            LanguageDecisionSample(
                sample_id=f"{DIALECT}:{record.record_id}",
                kind=DIALECT,
                language=record.language,
                gold=str(record.locale),
                options=options,
                source=record.source,
                start=0,
                end=len(record.text),
                mix=_mix_category(record),
                text=record.text,
            )
        )
    return samples


def _code_switch_samples(
    root: Path,
    records: Sequence[_FixtureRecord],
) -> list[LanguageDecisionSample]:
    samples: list[LanguageDecisionSample] = []
    for record in records:
        mix = _mix_category(record)
        if mix == MIX_UNDECLARED:
            continue
        samples.append(
            LanguageDecisionSample(
                sample_id=f"{CODE_SWITCH}:{record.record_id}",
                kind=CODE_SWITCH,
                language=record.language,
                gold="yes" if mix == MIX_DECLARED else "no",
                options=BOOLEAN_OPTIONS,
                source=record.source,
                start=0,
                end=len(record.text),
                mix=mix,
                text=record.text,
            )
        )

    source = _CODE_MIXED_FIXTURE
    for payload in _read_jsonl(root / source, source):
        text = payload.get("text")
        tokens = payload.get("gold_tokens")
        if not isinstance(text, str) or not isinstance(tokens, list):
            continue
        token_languages = {
            token.get("label") for token in tokens if isinstance(token, Mapping)
        } & _CODE_MIXED_TOKEN_LANGUAGES
        if len(token_languages) < 2:
            continue
        record_id = _record_id(payload, source)
        samples.append(
            LanguageDecisionSample(
                sample_id=f"{CODE_SWITCH}:{record_id}",
                kind=CODE_SWITCH,
                language=_CODE_MIXED_LANGUAGE,
                gold="yes",
                options=BOOLEAN_OPTIONS,
                source=source,
                start=0,
                end=len(text),
                mix=MIX_DECLARED,
                text=text,
            )
        )

    source = f"{_I18N_DIR}/{_ALL_SCRIPT_FIXTURE}"
    for payload in _read_jsonl(root / source, source):
        text = payload.get("text")
        runs = payload.get("runs")
        if not isinstance(text, str) or not isinstance(runs, list):
            continue
        run_languages = {
            run.get("language") for run in runs if isinstance(run, Mapping)
        }
        if len(run_languages) < 2:
            continue
        samples.append(
            LanguageDecisionSample(
                sample_id=f"{CODE_SWITCH}:{_record_id(payload, source)}",
                kind=CODE_SWITCH,
                language=MULTIPLE_LANGUAGES,
                gold="yes",
                options=BOOLEAN_OPTIONS,
                source=source,
                start=0,
                end=len(text),
                mix=MIX_DECLARED,
                text=text,
            )
        )
    return samples


def _annotated_runs(
    root: Path,
) -> Iterable[tuple[str, str, str, int, Mapping[str, Any]]]:
    source = f"{_I18N_DIR}/{_ALL_SCRIPT_FIXTURE}"
    for payload in _read_jsonl(root / source, source):
        text = payload.get("text")
        runs = payload.get("runs")
        if not isinstance(text, str) or not isinstance(runs, list):
            continue
        record_id = _record_id(payload, source)
        for index, run in enumerate(runs):
            if isinstance(run, Mapping):
                yield record_id, source, text, index, run


def _run_sample(
    kind: str,
    record_id: str,
    source: str,
    text: str,
    index: int,
    run: Mapping[str, Any],
    gold: str,
    options: tuple[str, ...],
) -> LanguageDecisionSample:
    start, end = run.get("start"), run.get("end")
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= start < end <= len(text)
    ):
        raise LanguageDecisionEvalError(f"{record_id} run {index} has invalid offsets")
    language = run.get("language")
    return LanguageDecisionSample(
        sample_id=f"{kind}:{record_id}#run{index}",
        kind=kind,
        language=str(language),
        gold=gold,
        options=options,
        source=source,
        start=start,
        end=end,
        mix=MIX_NONE,
        text=text[start:end],
    )


def _read_jsonl(path: Path, source: str) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                raise LanguageDecisionEvalError(
                    f"{source}:{line_number} is not valid JSON"
                ) from None
            if not isinstance(payload, Mapping):
                raise LanguageDecisionEvalError(
                    f"{source}:{line_number} must be a JSON object"
                )
            if payload.get("synthetic") is False or (
                isinstance(payload.get("metadata"), Mapping)
                and payload["metadata"].get("synthetic") is False
            ):
                raise LanguageDecisionEvalError(
                    f"{source}:{line_number} is not marked synthetic"
                )
            payloads.append(payload)
    return payloads


def _record_id(payload: Mapping[str, Any], source: str) -> str:
    record_id = payload.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise LanguageDecisionEvalError(f"{source} has a record without an id")
    return record_id


def _optional_text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _declares_code_mixing(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bool:
    """Return whether a fixture record declares that it mixes languages.

    Declarations are the ``code_mixed`` and ``india_clinical_codemixed``
    flags, a ``languages`` list with two or more entries, a ``register`` that
    names code-switching or mixing, and a ``fixture_kind`` that says
    code-mixed or pairs the record language with English.
    """

    if metadata.get("code_mixed") is True:
        return True
    if metadata.get("india_clinical_codemixed") is True:
        return True
    languages = payload.get("languages")
    if isinstance(languages, list) and len(set(map(str, languages))) > 1:
        return True
    register = str(metadata.get("register", "")).casefold()
    if "code_switched" in register or "mixed" in register:
        return True
    fixture_kind = str(metadata.get("fixture_kind", "")).casefold()
    if any(marker in fixture_kind for marker in _MIXED_KIND_MARKERS):
        return True
    words = fixture_kind.replace("-", " ").split()
    return "english" in words and payload.get("language") != "en"


def _declared_scripts(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    scripts = payload.get("scripts")
    if isinstance(scripts, list) and all(isinstance(item, str) for item in scripts):
        return tuple(dict.fromkeys(scripts))
    script = metadata.get("script")
    if isinstance(script, str) and script.strip():
        return (script.strip().capitalize(),)
    return ()


def _mix_category(record: _FixtureRecord) -> str:
    if record.declared_mix:
        return MIX_DECLARED
    if len(_letter_families(record.text)) > 1:
        return MIX_UNDECLARED
    return MIX_NONE


def _letter_families(text: str) -> frozenset[str]:
    families = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        families.add(_LETTER_FAMILIES.get(name.split(" ", 1)[0], "OTHER"))
    return frozenset(families)


def _locale_language(locale: Optional[str]) -> Optional[str]:
    if not locale or "_" not in locale:
        return None
    return locale.split("_", 1)[0]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_language_decisions(
    samples: Sequence[LanguageDecisionSample],
    backends: Mapping[str, DecisionBackend],
    *,
    router: Optional[LanguageRouter] = None,
    calibration: DecisionCalibrationProfile = DEFAULT_CALIBRATION_PROFILE,
    tasks: Optional[Mapping[str, str]] = None,
) -> LanguageDecisionEvaluation:
    """Answer every sample with the deterministic router and a decision backend.

    Args:
        samples: Samples from :func:`build_language_decision_samples`.
        backends: One local decision backend per decision kind present in
            ``samples``.
        router: Deterministic router. Defaults to the stdlib-only router with
            optional CLD2 disabled, so results do not depend on optional
            packages.
        calibration: Calibration profile applied by the decision contract.
        tasks: Optional task name per kind, recorded in the report.

    Returns:
        A :class:`LanguageDecisionEvaluation` with value-free records.

    Raises:
        LanguageDecisionEvalError: If samples are empty or a backend is
            missing.
    """

    if not samples:
        raise LanguageDecisionEvalError("evaluation requires samples")
    kinds = sorted({sample.kind for sample in samples})
    missing = [kind for kind in kinds if kind not in backends]
    if missing:
        raise LanguageDecisionEvalError(
            f"no decision backend for kinds: {', '.join(missing)}"
        )
    active_router = router or LanguageRouter(use_optional_lid=False)
    records = tuple(
        LanguageDecisionRecord(
            sample_id=sample.sample_id,
            kind=sample.kind,
            language=sample.language,
            resource_tier=resource_tier(sample.language),
            mix=sample.mix,
            gold=sample.gold,
            router=router_outcome(sample, active_router),
            model=model_outcome(sample, backends[sample.kind], calibration),
        )
        for sample in samples
    )
    lid = active_router.language_identifier
    return LanguageDecisionEvaluation(
        dataset_digest=dataset_digest(samples),
        sources=tuple(sorted({sample.source for sample in samples})),
        records=records,
        backends={kind: backends[kind].identity.to_dict() for kind in kinds},
        tasks={kind: tasks[kind] for kind in kinds if tasks and kind in tasks},
        calibration=calibration.to_dict(),
        lid_backend=str(lid.name) if lid is not None else None,
    )


def router_outcome(
    sample: LanguageDecisionSample,
    router: LanguageRouter,
) -> RouterOutcome:
    """Return the deterministic answer for one sample.

    Languages use the document decision; the answer is decisive when every run
    routed to the dominant language carries evidence-level confidence.
    Scripts use :func:`detect_script`, which is always decisive. Dialects use
    the language pack's default locale; the router has no dialect evidence, so
    that answer is never decisive. Code-switching reports whether
    letter-bearing runs route to more than one language and is decisive when
    every such run carries evidence-level confidence.

    Args:
        sample: Evaluation sample.
        router: Deterministic language router.

    Returns:
        The predicted option code and whether evidence decided it.
    """

    if sample.kind == SCRIPT:
        return RouterOutcome(prediction=detect_script(sample.text), decisive=True)
    if sample.kind == DIALECT:
        return RouterOutcome(
            prediction=LANG_TO_LOCALE.get(sample.language, ""),
            decisive=False,
        )
    decision = router.route(sample.text)
    if sample.kind in (LANGUAGE, LANGUAGE_IN_SCRIPT):
        dominant = [run for run in decision.runs if run.language == decision.language]
        return RouterOutcome(
            prediction=decision.language,
            decisive=bool(dominant)
            and all(run.confidence >= ROUTER_EVIDENCE_CONFIDENCE for run in dominant),
        )
    lexical = [
        run
        for run in decision.runs
        if any(char.isalpha() for char in sample.text[run.start : run.end])
    ]
    return RouterOutcome(
        prediction="yes" if len({run.language for run in lexical}) > 1 else "no",
        decisive=bool(lexical)
        and all(run.confidence >= ROUTER_EVIDENCE_CONFIDENCE for run in lexical),
    )


def model_outcome(
    sample: LanguageDecisionSample,
    backend: DecisionBackend,
    calibration: DecisionCalibrationProfile = DEFAULT_CALIBRATION_PROFILE,
) -> ModelOutcome:
    """Ask the decision contract one sample's question and map the answer.

    Code-switch questions use ``boolean_choice``; every other kind uses
    ``fixed_choice`` over :func:`option_labels`. Backend failures become a
    non-success state instead of an exception.

    Args:
        sample: Evaluation sample.
        backend: Local decision backend.
        calibration: Calibration profile applied by the contract.

    Returns:
        The contract state, the chosen and top-ranked option codes, and the
        confidence and margin.
    """

    labels = option_labels(sample)
    mode = (
        DecisionMode.BOOLEAN_CHOICE
        if sample.kind == CODE_SWITCH
        else DecisionMode.FIXED_CHOICE
    )
    result = decide(
        DecisionRequest(
            mode=mode,
            input_text=sample.text,
            options=labels,
            purpose="quality",
            calibration_id=calibration.calibration_id,
            timeout_ms=MAX_TIMEOUT_MS,
        ),
        backend=backend,
        calibration_profiles={calibration.calibration_id: calibration},
    )
    codes = dict(zip(labels, sample.options, strict=True))
    return ModelOutcome(
        state=result.state.value,
        choice=codes.get(result.choice) if result.choice is not None else None,
        top1=codes.get(result.ranking[0]) if result.ranking else None,
        confidence=result.confidence,
        margin=result.margin,
    )


def gliner2_language_decision_backends(handle: Any) -> dict[str, DecisionBackend]:
    """Wrap one loaded GLiNER2 handle as a decision backend per kind.

    Args:
        handle: A :class:`~openmed.ner.families.gliner2.GLiNER2Handle`.

    Returns:
        A backend per entry of :data:`DECISION_KINDS`, each sending its
        :data:`DECISION_TASKS` name as the classification task.
    """

    from openmed.ner.families.gliner2_decision import GLiNER2DecisionBackend

    return {
        kind: GLiNER2DecisionBackend(handle, task=DECISION_TASKS[kind])
        for kind in DECISION_KINDS
    }


def run_gliner2_language_decision_eval(
    *,
    model_id: str = GLINER2_LANGUAGE_DECISION_MODEL,
    revision: Optional[str] = GLINER2_LANGUAGE_DECISION_REVISION,
    device: Optional[str] = "cpu",
    cache_dir: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    loader: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Load a pinned GLiNER2.5 checkpoint and return the aggregate report.

    Args:
        model_id: Hugging Face repository id or local checkpoint directory.
        revision: Commit to pin. Defaults to
            :data:`GLINER2_LANGUAGE_DECISION_REVISION`.
        device: Torch device; CPU by default for reproducible scores.
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: Never download; ``None`` follows ``OPENMED_OFFLINE``.
        loader: Optional replacement for
            :func:`openmed.ner.families.gliner2.load_gliner2_handle`.

    Returns:
        The value-free report from :func:`build_language_decision_report`.
    """

    if loader is None:
        from openmed.ner.families.gliner2 import load_gliner2_handle

        loader = load_gliner2_handle
    handle = loader(
        model_id,
        revision=revision,
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    samples = build_language_decision_samples()
    evaluation = evaluate_language_decisions(
        samples,
        gliner2_language_decision_backends(handle),
        tasks=DECISION_TASKS,
    )
    return evaluation.to_dict()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_language_decision_report(
    evaluation: LanguageDecisionEvaluation,
) -> dict[str, Any]:
    """Aggregate records into per-kind, per-tier and per-language counts.

    Args:
        evaluation: Result of :func:`evaluate_language_decisions`.

    Returns:
        A JSON-ready report with counts, rates, controlled codes and digests
        only. ``contains_source_text`` is always false.
    """

    by_kind: dict[str, list[LanguageDecisionRecord]] = {}
    for record in evaluation.records:
        by_kind.setdefault(record.kind, []).append(record)
    kinds = [kind for kind in DECISION_KINDS if kind in by_kind]

    report_kinds: dict[str, Any] = {}
    for kind in kinds:
        records = by_kind[kind]
        report_kinds[kind] = {
            "summary": _summary(records),
            "low_resource": _summary(
                [item for item in records if item.resource_tier in LOW_RESOURCE_TIERS]
            ),
            "by_tier": _table(
                {
                    tier: _row([item for item in records if item.resource_tier == tier])
                    for tier in RESOURCE_TIERS
                    if any(item.resource_tier == tier for item in records)
                }
            ),
            "by_mix": _table(
                {
                    mix: _row([item for item in records if item.mix == mix])
                    for mix in MIX_CATEGORIES
                    if any(item.mix == mix for item in records)
                }
            ),
            "confidence_sweep": _confidence_sweep(records),
            "languages": _table(
                {
                    language: [
                        tier,
                        *_row([item for item in records if item.language == language]),
                    ]
                    for language, tier in sorted(
                        {(item.language, item.resource_tier) for item in records}
                    )
                },
                columns=LANGUAGE_ROW_COLUMNS,
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "advisory": ADVISORY,
        "contains_source_text": False,
        "dataset": {
            "digest": evaluation.dataset_digest,
            "samples": len(evaluation.records),
            "kinds": {kind: len(by_kind[kind]) for kind in kinds},
            "languages": {
                kind: len({item.language for item in by_kind[kind]}) for kind in kinds
            },
            "sources": list(evaluation.sources),
            "synthetic": True,
        },
        "router": {
            "lid_backend": evaluation.lid_backend,
            "evidence_confidence": ROUTER_EVIDENCE_CONFIDENCE,
            "dialect_baseline": "pack_default_locale",
        },
        "backends": {kind: dict(value) for kind, value in evaluation.backends.items()},
        "tasks": dict(evaluation.tasks),
        "calibration": dict(evaluation.calibration),
        "policies": list(POLICIES),
        "kinds": report_kinds,
    }


def _counts(records: Sequence[LanguageDecisionRecord]) -> dict[str, Any]:
    policies: dict[str, dict[str, int]] = {}
    for name in POLICIES:
        outcomes = [record.policy(name) for record in records]
        policies[name] = {
            "accepted": sum(1 for accepted, _answer in outcomes if accepted),
            "correct": sum(
                1
                for (accepted, answer), record in zip(outcomes, records, strict=True)
                if accepted and answer == record.gold
            ),
        }
    return {
        "samples": len(records),
        "router": {
            "correct": sum(1 for r in records if r.router.prediction == r.gold),
            "decisive": sum(1 for r in records if r.router.decisive),
            "decisive_correct": sum(
                1
                for r in records
                if r.router.decisive and r.router.prediction == r.gold
            ),
        },
        "model": {
            "top1_correct": sum(1 for r in records if r.model.top1 == r.gold),
            "covered": sum(1 for r in records if r.model.covered),
            "covered_correct": sum(
                1 for r in records if r.model.covered and r.model.choice == r.gold
            ),
            "abstained": sum(
                1 for r in records if r.model.state == DecisionState.ABSTAINED.value
            ),
            "failed": sum(
                1
                for r in records
                if r.model.state
                not in (DecisionState.SUCCESS.value, DecisionState.ABSTAINED.value)
            ),
            "agrees_with_router": sum(
                1 for r in records if r.model.top1 == r.router.prediction
            ),
        },
        "policies": policies,
    }


def _row(records: Sequence[LanguageDecisionRecord]) -> list[int]:
    counts = _counts(records)
    router, model = counts["router"], counts["model"]
    agreement = counts["policies"]["model_agreement"]
    fallback = counts["policies"]["model_fallback"]
    return [
        counts["samples"],
        router["correct"],
        router["decisive"],
        model["top1_correct"],
        model["covered"],
        model["covered_correct"],
        agreement["accepted"],
        agreement["correct"],
        fallback["accepted"],
        fallback["correct"],
    ]


def _table(
    rows: Mapping[str, Sequence[Any]],
    *,
    columns: Sequence[str] = GROUP_ROW_COLUMNS,
) -> dict[str, Any]:
    return {
        "columns": list(columns),
        "rows": {key: list(row) for key, row in rows.items()},
    }


def _summary(records: Sequence[LanguageDecisionRecord]) -> dict[str, Any]:
    counts = _counts(records)
    total = counts["samples"]
    router, model = counts["router"], counts["model"]
    rates: dict[str, Any] = {
        "router_accuracy": _rate(router["correct"], total),
        "router_decisive_accuracy": _rate(
            router["decisive_correct"], router["decisive"]
        ),
        "model_top1_accuracy": _rate(model["top1_correct"], total),
        "model_coverage": _rate(model["covered"], total),
        "model_selective_accuracy": _rate(model["covered_correct"], model["covered"]),
        "policies": {
            name: {
                "coverage": _rate(values["accepted"], total),
                "selective_accuracy": _rate(values["correct"], values["accepted"]),
                "silent_error_rate": _rate(
                    values["accepted"] - values["correct"], total
                ),
            }
            for name, values in counts["policies"].items()
        },
    }
    return {**counts, "rates": rates}


def _confidence_sweep(
    records: Sequence[LanguageDecisionRecord],
) -> list[dict[str, Any]]:
    sweep = []
    for threshold in CONFIDENCE_SWEEP:
        covered = [
            record
            for record in records
            if record.model.top1 is not None
            and record.model.confidence is not None
            and record.model.confidence >= threshold
        ]
        correct = sum(1 for record in covered if record.model.top1 == record.gold)
        sweep.append(
            {
                "min_confidence": threshold,
                "covered": len(covered),
                "correct": correct,
                "coverage": _rate(len(covered), len(records)),
                "selective_accuracy": _rate(correct, len(covered)),
            }
        )
    return sweep


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator, _RATE_DIGITS)


def format_language_decision_summary(report: Mapping[str, Any]) -> str:
    """Render the report's per-kind and low-resource rates as Markdown.

    Args:
        report: Output of :func:`build_language_decision_report`.

    Returns:
        Markdown tables with counts and rates only.
    """

    lines = [
        "| Kind | Slice | n | Router acc. | Router evidence cov./acc. | "
        "Model top-1 | Model cov./acc. | Silent errors: router / evidence / "
        "agreement / fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, sections in report["kinds"].items():
        for label, summary in (
            ("all", sections["summary"]),
            ("low-resource", sections["low_resource"]),
        ):
            if not summary["samples"]:
                continue
            rates = summary["rates"]
            policy = rates["policies"]
            lines.append(
                f"| {kind} | {label} | {summary['samples']} | "
                f"{_percent(rates['router_accuracy'])} | "
                f"{_percent(policy['router_evidence']['coverage'])} / "
                f"{_percent(policy['router_evidence']['selective_accuracy'])} | "
                f"{_percent(rates['model_top1_accuracy'])} | "
                f"{_percent(rates['model_coverage'])} / "
                f"{_percent(rates['model_selective_accuracy'])} | "
                + " / ".join(
                    _percent(policy[name]["silent_error_rate"])
                    for name in (
                        "router_always",
                        "router_evidence",
                        "model_agreement",
                        "model_fallback",
                    )
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def format_language_table(report: Mapping[str, Any], kind: str = LANGUAGE) -> str:
    """Render per-language counts for one decision kind as Markdown.

    Args:
        report: Output of :func:`build_language_decision_report`.
        kind: Decision kind to render.

    Returns:
        A Markdown table with one row per language: sample count, router
        correct and decisive counts, model top-1 correct, covered and
        covered-correct counts, and ``model_fallback`` accepted and correct
        counts.
    """

    shown = (
        "resource_tier",
        "samples",
        "router_correct",
        "router_decisive",
        "model_top1_correct",
        "model_covered",
        "model_covered_correct",
        "model_fallback_accepted",
        "model_fallback_correct",
    )
    lines = [
        "| Language | Tier | n | Router correct | Router decisive | "
        "Model top-1 correct | Model covered | Model covered correct | "
        "Fallback accepted | Fallback correct |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table = report["kinds"][kind]["languages"]
    positions = [table["columns"].index(column) for column in shown]
    for language, row in table["rows"].items():
        cells = " | ".join(str(row[position]) for position in positions)
        lines.append(f"| {language} | {cells} |")
    return "\n".join(lines) + "\n"


def format_report_json(report: Mapping[str, Any]) -> str:
    """Serialize a report as stable, review-friendly JSON.

    Keys are sorted and indented, but objects and arrays that hold only
    scalars stay on one line, which keeps per-language rows compact.

    Args:
        report: Output of :func:`build_language_decision_report`.

    Returns:
        JSON text ending with a newline.
    """

    return _format_json(report, 0) + "\n"


def report_differences(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    tolerance: int = 2,
) -> list[str]:
    """Compare a fresh report against a committed one.

    The dataset, model identity, calibration and every deterministic router
    count must match exactly. Model and policy counts may differ by at most
    ``tolerance`` samples per count, because CPU kernels can flip a score that
    sits on a calibration threshold.

    Args:
        expected: Committed report.
        actual: Freshly computed report.
        tolerance: Allowed absolute difference for model-dependent counts.

    Returns:
        Value-free descriptions of each difference; empty when they agree.
    """

    differences: list[str] = []
    for key in ("schema_version", "dataset", "calibration", "router", "tasks"):
        if expected.get(key) != actual.get(key):
            differences.append(f"{key} differs")
    for kind, backend in (expected.get("backends") or {}).items():
        other = (actual.get("backends") or {}).get(kind) or {}
        for field_name in ("model_id", "revision", "license_id", "runtime"):
            if backend.get(field_name) != other.get(field_name):
                differences.append(f"backends.{kind}.{field_name} differs")
    expected_kinds = expected.get("kinds") or {}
    actual_kinds = actual.get("kinds") or {}
    if set(expected_kinds) != set(actual_kinds):
        differences.append("decision kinds differ")
    for kind in sorted(set(expected_kinds) & set(actual_kinds)):
        for section in ("summary", "low_resource"):
            _compare_counts(
                expected_kinds[kind][section],
                actual_kinds[kind][section],
                f"kinds.{kind}.{section}",
                tolerance,
                differences,
            )
        for section in ("by_tier", "by_mix", "languages"):
            _compare_table(
                expected_kinds[kind][section],
                actual_kinds[kind][section],
                f"kinds.{kind}.{section}",
                tolerance,
                differences,
            )
    return differences


def _compare_table(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    path: str,
    tolerance: int,
    differences: list[str],
) -> None:
    columns = list(expected.get("columns") or ())
    if columns != list(actual.get("columns") or ()):
        differences.append(f"{path}.columns differ")
        return
    expected_rows = expected.get("rows") or {}
    actual_rows = actual.get("rows") or {}
    if set(expected_rows) != set(actual_rows):
        differences.append(f"{path} rows differ")
        return
    for key in sorted(expected_rows):
        if not len(columns) == len(expected_rows[key]) == len(actual_rows[key]):
            differences.append(f"{path}.{key} has a malformed row")
            continue
        for column, value, other in zip(
            columns, expected_rows[key], actual_rows[key], strict=True
        ):
            name = f"{path}.{key}.{column}"
            if column in _EXACT_COLUMNS or not isinstance(value, int):
                if value != other:
                    differences.append(f"{name} differs")
            elif not isinstance(other, int) or abs(value - other) > tolerance:
                differences.append(f"{name} differs")


def _compare_counts(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    path: str,
    tolerance: int,
    differences: list[str],
) -> None:
    for key, value in expected.items():
        if key == "rates":
            continue
        other = actual.get(key)
        name = f"{path}.{key}"
        if isinstance(value, Mapping):
            if not isinstance(other, Mapping):
                differences.append(f"{name} is missing")
                continue
            _compare_counts(value, other, name, tolerance, differences)
        elif isinstance(value, bool) or not isinstance(value, int):
            if value != other:
                differences.append(f"{name} differs")
        elif not isinstance(other, int):
            differences.append(f"{name} is missing")
        elif ".router" in name or key == "samples":
            if value != other:
                differences.append(f"{name} differs")
        elif abs(value - other) > tolerance:
            differences.append(f"{name} differs by {abs(value - other)}")


def _format_json(value: Any, depth: int) -> str:
    if isinstance(value, Mapping):
        items = sorted(value.items())
        if all(not isinstance(item, (Mapping, list)) for _key, item in items):
            return json.dumps(dict(items), ensure_ascii=False, separators=(", ", ": "))
        inner = "  " * (depth + 1)
        body = ",\n".join(
            f"{inner}{json.dumps(key)}: {_format_json(item, depth + 1)}"
            for key, item in items
        )
        return "{\n" + body + "\n" + "  " * depth + "}"
    if isinstance(value, list):
        if all(not isinstance(item, (Mapping, list)) for item in value):
            return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        inner = "  " * (depth + 1)
        body = ",\n".join(f"{inner}{_format_json(item, depth + 1)}" for item in value)
        return "[\n" + body + "\n" + "  " * depth + "]"
    return json.dumps(value, ensure_ascii=False)


def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "ADVISORY",
    "BOOLEAN_OPTIONS",
    "CODE_SWITCH",
    "CONFIDENCE_SWEEP",
    "DECISION_KINDS",
    "DECISION_TASKS",
    "DIALECT",
    "GLINER2_LANGUAGE_DECISION_MODEL",
    "GLINER2_LANGUAGE_DECISION_REVISION",
    "GROUP_ROW_COLUMNS",
    "LANGUAGE",
    "LANGUAGE_IN_SCRIPT",
    "LANGUAGE_NAMES",
    "LANGUAGE_ROW_COLUMNS",
    "LANGUAGE_SCRIPTS",
    "LOCALE_LABELS",
    "LOW_RESOURCE_TIERS",
    "MIX_CATEGORIES",
    "MULTIPLE_LANGUAGES",
    "POLICIES",
    "RESOURCE_TIERS",
    "ROUTER_EVIDENCE_CONFIDENCE",
    "SCHEMA_VERSION",
    "SCRIPT",
    "SCRIPT_LABELS",
    "LanguageDecisionEvalError",
    "LanguageDecisionEvaluation",
    "LanguageDecisionRecord",
    "LanguageDecisionSample",
    "ModelOutcome",
    "RouterOutcome",
    "build_language_decision_report",
    "build_language_decision_samples",
    "dataset_digest",
    "evaluate_language_decisions",
    "format_language_decision_summary",
    "format_language_table",
    "format_report_json",
    "gliner2_language_decision_backends",
    "model_outcome",
    "option_labels",
    "report_differences",
    "resource_tier",
    "router_outcome",
    "run_gliner2_language_decision_eval",
]
