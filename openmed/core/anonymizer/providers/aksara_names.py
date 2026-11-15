"""Pronounceable aksara-based Devanagari and Telugu name surrogates.

Devanagari and Telugu write a name as a run of aksara (orthographic
syllables). An aksara is either a word-initial independent vowel or a
consonant, optionally joined to further consonants by a virama, followed by
at most one dependent vowel sign and at most one anusvara or visarga. The
generators here compose surrogate names from those parts instead of sampling
unrelated letters, so each token reads as a plausible name.

The syllable inventories are authored in this module. They hold consonants,
vowel signs, and a few common conjuncts, never a list of real names. Every
surrogate:

* keeps the source's whitespace token count and order, and turns a
  one-aksara initial such as ``क.`` into another one-aksara initial;
* shares no script code point with the source surface, so no source
  consonant, independent vowel, vowel sign, virama, anusvara, or visarga is
  reused;
* segments into well-formed aksara under the grapheme engine in
  :mod:`openmed.core.decoding.spans`; and
* depends only on the caller's seeded ``faker.random`` stream.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from ...decoding.spans import iter_grapheme_cluster_spans
from ...language_pack import LanguagePack, get_language_pack

WeightedParts = tuple[tuple[str, int], ...]
"""Inventory entries paired with positive relative sampling weights."""

_MIN_TOKEN_AKSARA: Final = 2
_TOKEN_ATTEMPTS: Final = 16
_INITIAL_VOWEL_RATE: Final = 0.15
_INITIAL_CLUSTER_RATE: Final = 0.1
_MEDIAL_CLUSTER_RATE: Final = 0.2
_HEAVY_CONSONANT_RATE: Final = 0.15
_INHERENT_AFTER_SIGN_RATE: Final = 0.55
_INHERENT_AFTER_INHERENT_RATE: Final = 0.35
_ANUSVARA_RATE: Final = 0.12
_VISARGA_RATE: Final = 0.04
_FINAL_ANUSVARA_RATE: Final = 0.12
_INITIAL_SUFFIX: Final = "."
_PART_SEPARATOR: Final = "-"
_WHITESPACE_RE: Final = re.compile(r"(\s+)")


@dataclass(frozen=True)
class AksaraInventory:
    """Repository-authored syllable inventory for one Brahmic script.

    Attributes:
        script: Script name reported by :mod:`openmed.core.script_detect`.
        onset_consonants: Weighted frequent consonants for any aksara.
        heavy_consonants: Weighted aspirated, retroflex, and similar
            consonants. A token holds at most one of them.
        medial_only: Heavy consonants that never start a token.
        initial_clusters: Weighted virama-joined conjuncts for any aksara.
        medial_clusters: Weighted conjuncts used only after the first aksara.
        initial_vowels: Weighted independent vowels for a token's first
            aksara.
        vowel_signs: Weighted dependent vowel signs for non-final aksara.
        initial_vowel_signs: Weighted dependent vowel signs used only on a
            token's first aksara.
        final_vowel_signs: Weighted dependent vowel signs that end a token.
        initialism_signs: Weighted dependent vowel signs for a one-aksara
            initial such as ``के.``.
        final_inherent_rate: Probability that a token ends in the inherent
            vowel rather than a dependent vowel sign.
        final_anusvara: Whether a token may end in an anusvara.
        max_token_aksara: Upper bound on the aksara count of one token.
        anusvara_blockers: Nasal, semivowel, and liquid consonants that
            never follow an anusvara.
        sibilants: The only consonants that may follow a visarga. Two
            sibilant onsets are never adjacent.
        ra: The ``ra`` consonant, which never takes the vocalic-r sign.
        vocalic_r_sign: Dependent vowel sign limited to a single non-``ra``
            consonant.
        virama: The script's virama.
        anusvara: The script's anusvara.
        visarga: The script's visarga.
        title_tokens: Honorific tokens never emitted as a whole token.
        aksara_pattern: Regular expression for one well-formed aksara.
    """

    script: str
    onset_consonants: WeightedParts
    heavy_consonants: WeightedParts
    medial_only: frozenset[str]
    initial_clusters: WeightedParts
    medial_clusters: WeightedParts
    initial_vowels: WeightedParts
    vowel_signs: WeightedParts
    initial_vowel_signs: WeightedParts
    final_vowel_signs: WeightedParts
    initialism_signs: WeightedParts
    final_inherent_rate: float
    final_anusvara: bool
    max_token_aksara: int
    anusvara_blockers: frozenset[str]
    sibilants: frozenset[str]
    ra: str
    vocalic_r_sign: str
    virama: str
    anusvara: str
    visarga: str
    title_tokens: frozenset[str]
    aksara_pattern: re.Pattern[str]


def _char_class(start: int, end: int) -> str:
    """Return a regex character-class range for ``start`` to ``end``."""

    return f"{re.escape(chr(start))}-{re.escape(chr(end))}"


def _aksara_pattern(
    *,
    consonants: tuple[int, int],
    vowel_signs: tuple[int, int],
    independent_vowels: tuple[int, int],
    modifiers: tuple[int, int],
    virama: int,
) -> re.Pattern[str]:
    """Compile the well-formed aksara grammar for one script block."""

    consonant = f"[{_char_class(*consonants)}]"
    return re.compile(
        f"(?:{consonant}(?:{re.escape(chr(virama))}{consonant}){{0,2}}"
        f"[{_char_class(*vowel_signs)}]?"
        f"|[{_char_class(*independent_vowels)}])"
        f"[{_char_class(*modifiers)}]?"
    )


DEVANAGARI_AKSARA_INVENTORY: Final = AksaraInventory(
    script="Devanagari",
    onset_consonants=(
        ("क", 4),
        ("ग", 3),
        ("च", 2),
        ("ज", 3),
        ("त", 3),
        ("द", 3),
        ("न", 4),
        ("प", 3),
        ("ब", 2),
        ("म", 4),
        ("य", 2),
        ("र", 4),
        ("ल", 3),
        ("व", 3),
        ("श", 3),
        ("स", 4),
        ("ह", 3),
    ),
    heavy_consonants=(
        ("ख", 1),
        ("घ", 1),
        ("छ", 1),
        ("झ", 1),
        ("ट", 1),
        ("ड", 1),
        ("थ", 1),
        ("ध", 2),
        ("फ", 1),
        ("भ", 2),
        ("ण", 1),
        ("ष", 1),
    ),
    medial_only=frozenset({"ण", "ष"}),
    initial_clusters=(
        ("क्र", 1),
        ("ग्र", 1),
        ("त्र", 1),
        ("द्र", 1),
        ("प्र", 2),
        ("श्र", 1),
        ("क्ष", 1),
        ("स्व", 1),
    ),
    medial_clusters=(
        ("न्द", 2),
        ("न्त", 2),
        ("र्म", 1),
        ("र्ज", 1),
        ("ष्ण", 1),
        ("स्त", 1),
        ("त्य", 1),
        ("म्ब", 1),
        ("क्त", 1),
        ("र्श", 1),
    ),
    initial_vowels=(
        ("अ", 3),
        ("आ", 1),
        ("इ", 1),
        ("ई", 1),
        ("उ", 2),
        ("ए", 1),
        ("ओ", 1),
    ),
    vowel_signs=(
        ("ा", 5),
        ("ि", 4),
        ("ी", 3),
        ("ु", 3),
        ("े", 3),
        ("ो", 2),
        ("ू", 1),
    ),
    initial_vowel_signs=(("ै", 1), ("ौ", 1), ("ृ", 1)),
    final_vowel_signs=(("ा", 5), ("ी", 4), ("ु", 1), ("े", 1)),
    initialism_signs=(("े", 2), ("ी", 2), ("ा", 1)),
    final_inherent_rate=0.45,
    final_anusvara=False,
    max_token_aksara=4,
    anusvara_blockers=frozenset({"ङ", "ञ", "ण", "न", "म", "य", "र", "ल"}),
    sibilants=frozenset({"श", "ष", "स"}),
    ra="र",
    vocalic_r_sign="ृ",
    virama="्",
    anusvara="ं",
    visarga="ः",
    title_tokens=frozenset(
        {
            "कुमारी",
            "जी",
            "डॉ",
            "माननीय",
            "श्री",
            "श्रीमती",
            "श्रीमान",
            "संमानित",
            "सौ",
        }
    ),
    aksara_pattern=_aksara_pattern(
        consonants=(0x0915, 0x0939),
        vowel_signs=(0x093E, 0x094C),
        independent_vowels=(0x0904, 0x0914),
        modifiers=(0x0901, 0x0903),
        virama=0x094D,
    ),
)
"""Devanagari syllable inventory for Hindi name surrogates."""

TELUGU_AKSARA_INVENTORY: Final = AksaraInventory(
    script="Telugu",
    onset_consonants=(
        ("క", 3),
        ("గ", 3),
        ("చ", 2),
        ("జ", 2),
        ("త", 3),
        ("ద", 2),
        ("న", 3),
        ("ప", 3),
        ("బ", 2),
        ("మ", 3),
        ("య", 2),
        ("ర", 4),
        ("ల", 3),
        ("వ", 4),
        ("శ", 2),
        ("స", 3),
        ("హ", 2),
    ),
    heavy_consonants=(
        ("ఖ", 1),
        ("ఘ", 1),
        ("ఛ", 1),
        ("ఝ", 1),
        ("ట", 1),
        ("డ", 1),
        ("థ", 1),
        ("ధ", 1),
        ("ఫ", 1),
        ("భ", 2),
        ("ణ", 1),
        ("ష", 1),
        ("ళ", 1),
    ),
    medial_only=frozenset({"ణ", "ష", "ళ"}),
    initial_clusters=(
        ("క్ర", 1),
        ("గ్ర", 1),
        ("త్ర", 1),
        ("ద్ర", 1),
        ("ప్ర", 2),
        ("శ్ర", 1),
        ("క్ష", 1),
        ("స్వ", 1),
    ),
    medial_clusters=(
        ("ల్ల", 2),
        ("మ్మ", 2),
        ("న్న", 2),
        ("య్య", 2),
        ("ట్ట", 1),
        ("క్క", 1),
        ("ర్మ", 1),
        ("ష్ణ", 1),
        ("స్త", 1),
        ("త్య", 1),
    ),
    initial_vowels=(
        ("అ", 3),
        ("ఆ", 1),
        ("ఇ", 1),
        ("ఈ", 1),
        ("ఉ", 2),
        ("ఎ", 1),
        ("ఏ", 1),
        ("ఒ", 1),
        ("ఓ", 1),
    ),
    vowel_signs=(
        ("ా", 5),
        ("ి", 4),
        ("ీ", 3),
        ("ు", 4),
        ("ూ", 1),
        ("ె", 2),
        ("ే", 3),
        ("ొ", 1),
        ("ో", 2),
    ),
    initial_vowel_signs=(("ై", 1), ("ౌ", 1), ("ృ", 1)),
    final_vowel_signs=(("ు", 5), ("ి", 4), ("ా", 1), ("ీ", 1)),
    initialism_signs=(("ె", 2), ("ి", 2), ("ా", 1)),
    final_inherent_rate=0.35,
    final_anusvara=True,
    max_token_aksara=5,
    anusvara_blockers=frozenset({"ఙ", "ఞ", "ణ", "న", "మ", "య", "ర", "ల"}),
    sibilants=frozenset({"శ", "ష", "స"}),
    ra="ర",
    vocalic_r_sign="ృ",
    virama="్",
    anusvara="ం",
    visarga="ః",
    title_tokens=frozenset(
        {
            "కుమారి",
            "గారు",
            "డాక్టర్",
            "శ్రీ",
            "శ్రీమతి",
            "శ్రీమాన్",
        }
    ),
    aksara_pattern=_aksara_pattern(
        consonants=(0x0C15, 0x0C39),
        vowel_signs=(0x0C3E, 0x0C4C),
        independent_vowels=(0x0C05, 0x0C14),
        modifiers=(0x0C01, 0x0C03),
        virama=0x0C4D,
    ),
)
"""Telugu syllable inventory for Telugu name surrogates."""

AKSARA_INVENTORIES: Final[Mapping[str, AksaraInventory]] = MappingProxyType(
    {
        DEVANAGARI_AKSARA_INVENTORY.script: DEVANAGARI_AKSARA_INVENTORY,
        TELUGU_AKSARA_INVENTORY.script: TELUGU_AKSARA_INVENTORY,
    }
)
"""Aksara inventories keyed by script name."""


@dataclass(frozen=True)
class _Onset:
    text: str
    kind: str  # "vowel", "consonant", or "cluster"


@dataclass(frozen=True)
class _Pools:
    """One inventory with every part that reuses a source code point removed."""

    onset_consonants: WeightedParts
    heavy_consonants: WeightedParts
    initial_clusters: WeightedParts
    medial_clusters: WeightedParts
    initial_vowels: WeightedParts
    vowel_signs: WeightedParts
    initial_vowel_signs: WeightedParts
    final_vowel_signs: WeightedParts
    initialism_signs: WeightedParts
    anusvara: str
    visarga: str


def _disjoint(parts: WeightedParts, forbidden: frozenset[str]) -> WeightedParts:
    return tuple((text, weight) for text, weight in parts if forbidden.isdisjoint(text))


def _source_pools(inventory: AksaraInventory, source: str) -> _Pools:
    forbidden = frozenset(source)
    pools = _Pools(
        onset_consonants=_disjoint(inventory.onset_consonants, forbidden),
        heavy_consonants=_disjoint(inventory.heavy_consonants, forbidden),
        initial_clusters=_disjoint(inventory.initial_clusters, forbidden),
        medial_clusters=_disjoint(inventory.medial_clusters, forbidden),
        initial_vowels=_disjoint(inventory.initial_vowels, forbidden),
        vowel_signs=_disjoint(inventory.vowel_signs, forbidden),
        initial_vowel_signs=_disjoint(inventory.initial_vowel_signs, forbidden),
        final_vowel_signs=_disjoint(inventory.final_vowel_signs, forbidden),
        initialism_signs=_disjoint(inventory.initialism_signs, forbidden),
        anusvara="" if inventory.anusvara in forbidden else inventory.anusvara,
        visarga="" if inventory.visarga in forbidden else inventory.visarga,
    )
    if not pools.onset_consonants and not pools.heavy_consonants:
        # Names cannot plausibly exhaust the inventory; never echo the source.
        raise ValueError("source surface exhausts the synthetic aksara inventory")
    return pools


def _draw(
    rng,
    parts: WeightedParts,
    *,
    exclude: frozenset[str] = frozenset(),
) -> str | None:
    """Return one weighted entry of ``parts`` outside ``exclude``, or ``None``."""

    eligible = tuple((text, weight) for text, weight in parts if text not in exclude)
    if not eligible:
        return None
    return rng.choices(
        [text for text, _weight in eligible],
        weights=[weight for _text, weight in eligible],
        k=1,
    )[0]


def _draw_consonant(
    rng,
    inventory: AksaraInventory,
    pools: _Pools,
    *,
    first: bool,
    allow_heavy: bool,
    previous: str,
) -> str:
    excluded = {previous}
    if previous in inventory.sibilants:
        excluded.update(inventory.sibilants)
    if first:
        excluded.update(inventory.medial_only)
    exclude = frozenset(excluded)
    if allow_heavy and rng.random() < _HEAVY_CONSONANT_RATE:
        heavy = _draw(rng, pools.heavy_consonants, exclude=exclude)
        if heavy is not None:
            return heavy
    for candidate in (
        _draw(rng, pools.onset_consonants, exclude=exclude),
        _draw(rng, pools.heavy_consonants, exclude=exclude),
        _draw(rng, pools.onset_consonants),
    ):
        if candidate is not None:
            return candidate
    # Only heavy consonants survived the source filter.
    return str(_draw(rng, pools.heavy_consonants))


def _draw_onsets(
    rng,
    inventory: AksaraInventory,
    pools: _Pools,
    aksara_count: int,
) -> list[_Onset]:
    heavy = frozenset(text for text, _weight in inventory.heavy_consonants)
    onsets: list[_Onset] = []
    for position in range(aksara_count):
        first = position == 0
        if first and pools.initial_vowels and rng.random() < _INITIAL_VOWEL_RATE:
            onsets.append(_Onset(str(_draw(rng, pools.initial_vowels)), "vowel"))
            continue
        clusters = (
            pools.initial_clusters
            if first
            else pools.initial_clusters + pools.medial_clusters
        )
        has_cluster = any(onset.kind == "cluster" for onset in onsets)
        cluster_rate = _INITIAL_CLUSTER_RATE if first else _MEDIAL_CLUSTER_RATE
        if clusters and not has_cluster and rng.random() < cluster_rate:
            onsets.append(_Onset(str(_draw(rng, clusters)), "cluster"))
            continue
        consonant = _draw_consonant(
            rng,
            inventory,
            pools,
            first=first,
            allow_heavy=not any(onset.text in heavy for onset in onsets),
            previous=onsets[-1].text if onsets else "",
        )
        onsets.append(_Onset(consonant, "consonant"))
    return onsets


def _draw_nucleus(
    rng,
    inventory: AksaraInventory,
    pools: _Pools,
    onset: _Onset,
    *,
    first: bool,
    last: bool,
    previous: str | None,
) -> str:
    """Draw a vowel sign, or ``""`` for the inherent vowel.

    ``previous`` is the preceding aksara's vowel sign (``""`` for its
    inherent vowel, ``None`` for the first aksara). A sign never repeats on
    adjacent aksara, and a sign is more often followed by the inherent vowel,
    which keeps the long and short vowels alternating as in natural names.
    """

    if onset.kind == "vowel":
        return ""
    excluded = set() if previous is None else {previous}
    if onset.kind != "consonant" or onset.text == inventory.ra:
        excluded.add(inventory.vocalic_r_sign)
    exclude = frozenset(excluded)
    if first and last:
        # A one-aksara initial reads like a spelled letter, e.g. ``के.``.
        signs = pools.initialism_signs or pools.final_vowel_signs
        return _draw(rng, signs, exclude=exclude) or ""
    if last:
        if rng.random() < inventory.final_inherent_rate:
            return ""
        return _draw(rng, pools.final_vowel_signs, exclude=exclude) or ""
    inherent_rate = (
        _INHERENT_AFTER_SIGN_RATE if previous else _INHERENT_AFTER_INHERENT_RATE
    )
    if rng.random() < inherent_rate:
        return ""
    signs = (
        pools.vowel_signs + pools.initial_vowel_signs if first else pools.vowel_signs
    )
    return _draw(rng, signs, exclude=exclude) or ""


def _draw_coda(
    rng,
    inventory: AksaraInventory,
    pools: _Pools,
    nucleus: str,
    following: _Onset | None,
) -> str:
    if nucleus and any(nucleus == sign for sign, _ in inventory.initial_vowel_signs):
        return ""
    if following is None:
        if (
            inventory.final_anusvara
            and pools.anusvara
            and not nucleus
            and rng.random() < _FINAL_ANUSVARA_RATE
        ):
            return pools.anusvara
        return ""
    if following.kind != "consonant":
        return ""
    if (
        pools.anusvara
        and following.text not in inventory.anusvara_blockers
        and rng.random() < _ANUSVARA_RATE
    ):
        return pools.anusvara
    if (
        pools.visarga
        and following.text in inventory.sibilants
        and rng.random() < _VISARGA_RATE
    ):
        return pools.visarga
    return ""


def _build_token(
    rng,
    inventory: AksaraInventory,
    pools: _Pools,
    aksara_count: int,
) -> str:
    onsets = _draw_onsets(rng, inventory, pools, aksara_count)
    aksara: list[str] = []
    previous: str | None = None
    for position, onset in enumerate(onsets):
        last = position == aksara_count - 1
        nucleus = _draw_nucleus(
            rng,
            inventory,
            pools,
            onset,
            first=position == 0,
            last=last,
            previous=previous,
        )
        following = None if last else onsets[position + 1]
        coda = _draw_coda(rng, inventory, pools, nucleus, following)
        aksara.append(f"{onset.text}{nucleus}{coda}")
        previous = nucleus
    return "".join(aksara)


def _has_mark(token: str) -> bool:
    return any(unicodedata.category(character).startswith("M") for character in token)


def _compose_token(
    rng,
    inventory: AksaraInventory,
    pools: _Pools,
    aksara_count: int,
) -> str:
    """Return one well-formed token of exactly ``aksara_count`` aksara.

    Well-formedness, aksara count, and code-point disjointness hold by
    construction and are re-checked. Avoiding honorific tokens and runs of
    bare consonants are preferences, retried a bounded number of times.
    """

    fallback = ""
    for _attempt in range(_TOKEN_ATTEMPTS):
        token = _build_token(rng, inventory, pools, aksara_count)
        if not _is_well_formed_part(token, inventory) or (
            sum(1 for _cluster in iter_aksara(token)) != aksara_count
        ):  # pragma: no cover - construction always yields whole aksara
            continue
        if token in inventory.title_tokens:
            continue
        if aksara_count < _MIN_TOKEN_AKSARA or _has_mark(token):
            return token
        fallback = fallback or token
    if fallback:
        return fallback
    raise ValueError(  # pragma: no cover - needs an exhausted inventory
        "could not compose a well-formed aksara token"
    )


def _content_aksara_count(part: str) -> int:
    """Count grapheme clusters in ``part`` that carry a letter, mark, or digit."""

    return sum(
        any(
            character.isalnum() or unicodedata.category(character).startswith("M")
            for character in cluster
        )
        for cluster in iter_aksara(part)
    )


def _replace_token(
    rng,
    inventory: AksaraInventory,
    pools: _Pools,
    token: str,
) -> tuple[str, bool]:
    """Replace one whitespace token, keeping its initial dot and hyphens.

    Returns:
        The rendered token and whether any part of it was regenerated.
    """

    core = token.rstrip(_INITIAL_SUFFIX)
    suffix = token[len(core) :]
    parts = core.split(_PART_SEPARATOR)
    rendered: list[str] = []
    regenerated = False
    for part in parts:
        count = _content_aksara_count(part)
        if count == 0:
            rendered.append(part)
            continue
        if suffix and len(parts) == 1 and count == 1:
            size = 1
        else:
            size = min(max(count, _MIN_TOKEN_AKSARA), inventory.max_token_aksara)
        rendered.append(_compose_token(rng, inventory, pools, size))
        regenerated = True
    return _PART_SEPARATOR.join(rendered) + suffix, regenerated


def generate_aksara_name(
    faker,
    original: str,
    *,
    inventory: AksaraInventory,
) -> str:
    """Return a pronounceable name built from well-formed aksara.

    Each whitespace token of ``original`` is replaced in place by a token of
    the same aksara count, clamped to two aksara and the inventory's
    ``max_token_aksara``. A one-aksara token ending in ``.`` stays a
    one-aksara initial, hyphenated parts keep their hyphens, and whitespace
    between tokens is preserved.

    Args:
        faker: Faker-like object whose ``random`` attribute is a seeded
            :class:`random.Random`.
        original: Detected name surface. It only shapes the output and
            excludes its own code points; it is never returned or logged.
        inventory: Syllable inventory for the target script.

    Returns:
        A surrogate with the source's token count and order that shares no
        script code point with ``original`` and never equals it.

    Raises:
        ValueError: If ``original`` uses so many consonants that no onset is
            left. The message never contains source text.
    """

    rng = faker.random
    source = str(original).strip()
    pools = _source_pools(inventory, source)
    pieces = _WHITESPACE_RE.split(source)
    rendered: list[str] = []
    regenerated = False
    for index, piece in enumerate(pieces):
        if index % 2 or not piece:
            rendered.append(piece)
            continue
        token, token_regenerated = _replace_token(rng, inventory, pools, piece)
        rendered.append(token)
        regenerated = regenerated or token_regenerated
    surrogate = "".join(rendered)
    if not regenerated or surrogate == source:
        # A source without letters still needs a name-shaped replacement.
        return _compose_token(rng, inventory, pools, _MIN_TOKEN_AKSARA)
    return surrogate


def generate_devanagari_aksara_name(faker, original: str, *, locale: str) -> str:
    """Return a pronounceable Devanagari name surrogate for ``original``.

    Args:
        faker: Seeded Faker-like object exposing ``random``.
        original: Detected Devanagari name surface.
        locale: Resolved Faker locale. Unused; names are script-driven.

    Returns:
        A Devanagari surrogate built from well-formed aksara.
    """

    del locale
    return generate_aksara_name(
        faker,
        original,
        inventory=DEVANAGARI_AKSARA_INVENTORY,
    )


def generate_telugu_aksara_name(faker, original: str, *, locale: str) -> str:
    """Return a pronounceable Telugu name surrogate for ``original``.

    Args:
        faker: Seeded Faker-like object exposing ``random``.
        original: Detected Telugu name surface.
        locale: Resolved Faker locale. Unused; names are script-driven.

    Returns:
        A Telugu surrogate built from well-formed aksara.
    """

    del locale
    return generate_aksara_name(
        faker,
        original,
        inventory=TELUGU_AKSARA_INVENTORY,
    )


def iter_aksara(text: str) -> Iterator[str]:
    """Yield the extended grapheme clusters of ``text``.

    Args:
        text: Text to segment without normalization.

    Yields:
        Each whole cluster, as segmented by
        :func:`openmed.core.decoding.spans.iter_grapheme_cluster_spans`.
    """

    for start, end in iter_grapheme_cluster_spans(text):
        yield text[start:end]


def _is_well_formed_part(part: str, inventory: AksaraInventory) -> bool:
    return bool(part) and all(
        inventory.aksara_pattern.fullmatch(cluster) for cluster in iter_aksara(part)
    )


def is_well_formed_aksara_name(value: str, *, script: str) -> bool:
    """Return whether every token of ``value`` is a run of well-formed aksara.

    A token may end in ``.`` (an initial) and may join parts with ``-``. Each
    part must segment, under the shared grapheme engine, into clusters that
    are a word-initial independent vowel or a consonant joined to at most two
    more consonants by virama, each followed by at most one dependent vowel
    sign and one candrabindu, anusvara, or visarga.

    Args:
        value: Candidate name surface.
        script: ``"Devanagari"`` or ``"Telugu"``.

    Returns:
        ``True`` when ``value`` has at least one token and every part of
        every token is well formed.

    Raises:
        ValueError: If ``script`` has no aksara inventory.
    """

    inventory = AKSARA_INVENTORIES.get(script)
    if inventory is None:
        raise ValueError(f"no aksara inventory for script {script!r}")
    tokens = str(value).split()
    return bool(tokens) and all(
        _is_well_formed_part(part, inventory)
        for token in tokens
        for part in token.rstrip(_INITIAL_SUFFIX).split(_PART_SEPARATOR)
    )


def _require_language_pack(code: str) -> LanguagePack:
    language_pack = get_language_pack(code)
    if language_pack is None:  # pragma: no cover - catalog import is mandatory
        raise RuntimeError(f"language pack {code!r} is not registered")
    return language_pack


# Marathi keeps the native three-part generators installed by the registry.
AKSARA_NAME_PACKS: Final[Sequence[tuple[LanguagePack, str, Callable[..., str]]]] = (
    (
        _require_language_pack("hi"),
        DEVANAGARI_AKSARA_INVENTORY.script,
        generate_devanagari_aksara_name,
    ),
    (
        _require_language_pack("te"),
        TELUGU_AKSARA_INVENTORY.script,
        generate_telugu_aksara_name,
    ),
)
"""Language packs whose name labels use the aksara generators."""


__all__ = [
    "AKSARA_INVENTORIES",
    "AKSARA_NAME_PACKS",
    "DEVANAGARI_AKSARA_INVENTORY",
    "TELUGU_AKSARA_INVENTORY",
    "AksaraInventory",
    "WeightedParts",
    "generate_aksara_name",
    "generate_devanagari_aksara_name",
    "generate_telugu_aksara_name",
    "is_well_formed_aksara_name",
    "iter_aksara",
]
