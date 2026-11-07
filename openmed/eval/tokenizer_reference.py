"""Pure-Python reference semantics for portable tokenizer conformance vectors.

The reference implements the subset of Hugging Face ``tokenizer.json`` that the
synthetic conformance tokenizer uses: ``BertNormalizer``, ``BertPreTokenizer``,
``WordPiece`` and ``TemplateProcessing`` plus non-normalized special added
tokens and right-side truncation windows with stride. It follows the Rust
``tokenizers`` crate, which is the declared reference that Python
``transformers``, DJL, swift-transformers and Transformers.js tokenizers aim
to reproduce.

The implementation depends only on the standard library so fixture generation
and ``--check`` runs stay offline and free of optional dependencies. Any
component, option or input outside the supported subset fails closed with
:class:`UnsupportedTokenizerFeature` and a controlled reason code instead of
guessing. Offsets are half-open Unicode scalar offsets into the original text.
"""

from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "ReferenceEncoding",
    "ReferenceTokenizer",
    "ReferenceWindow",
    "UnsupportedTokenizerFeature",
]

# Unicode White_Space, matching Rust ``char::is_whitespace``.
_WHITE_SPACE = frozenset(
    "\t\n\v\f\r \x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
_ASCII_PUNCTUATION = frozenset(string.punctuation)
# CJK ideograph blocks padded by the Rust ``BertNormalizer``.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B920, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)
_REMOVED_BY_CLEAN_TEXT = frozenset("\x00\ufffd")
_SUPPORTED_ADDED_TOKEN_FLAGS = {
    "single_word": False,
    "lstrip": False,
    "rstrip": False,
    "normalized": False,
}


class UnsupportedTokenizerFeature(ValueError):
    """Raised when a tokenizer or input is outside the reference subset.

    Attributes:
        code: Controlled reason code. It never contains tokenizer content or
            input text.
    """

    def __init__(self, code: str) -> None:
        super().__init__(f"unsupported tokenizer feature: {code}")
        self.code = code


@dataclass(frozen=True)
class _Char:
    """One normalized character aligned to an original scalar span."""

    value: str
    start: int
    end: int


@dataclass(frozen=True)
class ReferenceEncoding:
    """Model inputs produced by the reference tokenizer for one text.

    Attributes:
        tokens: Token strings, including template special tokens.
        input_ids: Vocabulary ids for ``tokens``.
        attention_mask: One per token; the reference never pads.
        token_type_ids: Segment ids from the post-processor template.
        special_tokens_mask: ``1`` for template special tokens, else ``0``.
        offsets: Half-open Unicode scalar spans in the original text;
            template special tokens use ``(0, 0)``.
        sequence_indices: Positions that came from the input sequence rather
            than from the post-processor template.
    """

    tokens: tuple[str, ...]
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_type_ids: tuple[int, ...]
    special_tokens_mask: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    sequence_indices: tuple[int, ...]


@dataclass(frozen=True)
class ReferenceWindow:
    """One truncation window over the sequence tokens of an encoding.

    Attributes:
        content_start: First sequence-token index covered by the window.
        content_end: Exclusive end sequence-token index.
        input_ids: Window ids after the post-processor template is applied.
    """

    content_start: int
    content_end: int
    input_ids: tuple[int, ...]


@dataclass(frozen=True)
class _SequenceToken:
    token: str
    token_id: int
    start: int
    end: int


class ReferenceTokenizer:
    """Reference tokenizer for the supported ``tokenizer.json`` subset."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        """Build the reference from a parsed ``tokenizer.json`` payload.

        Args:
            payload: Parsed ``tokenizer.json`` mapping.

        Raises:
            UnsupportedTokenizerFeature: If a component or option is outside
                the reference subset.
        """
        if payload.get("truncation") is not None:
            raise UnsupportedTokenizerFeature("embedded_truncation")
        if payload.get("padding") is not None:
            raise UnsupportedTokenizerFeature("embedded_padding")
        self._normalizer = _parse_normalizer(payload.get("normalizer"))
        pre_tokenizer = payload.get("pre_tokenizer")
        if (
            not isinstance(pre_tokenizer, Mapping)
            or pre_tokenizer.get("type") != "BertPreTokenizer"
        ):
            raise UnsupportedTokenizerFeature("pre_tokenizer")
        model = payload.get("model")
        if not isinstance(model, Mapping) or model.get("type") != "WordPiece":
            raise UnsupportedTokenizerFeature("model")
        vocab = model.get("vocab")
        if not isinstance(vocab, Mapping) or not vocab:
            raise UnsupportedTokenizerFeature("model_vocab")
        self._vocab: dict[str, int] = {str(k): int(v) for k, v in vocab.items()}
        self._unk_token = str(model.get("unk_token", "[UNK]"))
        if self._unk_token not in self._vocab:
            raise UnsupportedTokenizerFeature("model_unk_token")
        self._prefix = str(model.get("continuing_subword_prefix", "##"))
        self._max_chars = int(model.get("max_input_chars_per_word", 100))
        self._added = _parse_added_tokens(payload.get("added_tokens"), self._vocab)
        self._template = _parse_template(payload.get("post_processor"), self._vocab)

    @property
    def vocab(self) -> Mapping[str, int]:
        """Return the WordPiece vocabulary."""
        return dict(self._vocab)

    @property
    def special_token_ids(self) -> Mapping[str, int]:
        """Return special added tokens and their ids."""
        return dict(self._added)

    @property
    def template_special_count(self) -> int:
        """Return how many special tokens the single-sequence template adds."""
        return sum(1 for piece in self._template if piece[0] == "special")

    def normalize(self, text: str) -> str:
        """Return the normalizer output for ``text``.

        Args:
            text: Original text.

        Returns:
            Normalized text, before pre-tokenization and added-token splitting.
        """
        chars = [_Char(char, index, index + 1) for index, char in enumerate(text)]
        return "".join(item.value for item in self._normalize(chars))

    def encode(self, text: str) -> ReferenceEncoding:
        """Encode ``text`` with special tokens and without truncation.

        Args:
            text: Original text.

        Returns:
            The reference encoding.
        """
        return self._post_process(self._sequence(text))

    def windows(
        self, text: str, *, max_length: int, stride: int
    ) -> tuple[ReferenceWindow, ...]:
        """Return right-side truncation windows for ``text``.

        Windows follow Rust ``Encoding::truncate``: the sequence capacity is
        ``max_length`` minus the template special tokens, consecutive windows
        start ``capacity - stride`` tokens apart, and the last window ends at
        the final sequence token.

        Args:
            text: Original text.
            max_length: Maximum window length including special tokens.
            stride: Number of sequence tokens shared by consecutive windows.

        Returns:
            Windows in order.

        Raises:
            UnsupportedTokenizerFeature: If the window parameters are invalid.
        """
        capacity = max_length - self.template_special_count
        if stride < 0 or capacity <= 0 or stride >= capacity:
            raise UnsupportedTokenizerFeature("window_parameters")
        sequence = self._sequence(text)
        if len(sequence) <= capacity:
            ranges = [(0, len(sequence))]
        else:
            ranges = []
            for start in range(0, len(sequence), capacity - stride):
                stop = min(start + capacity, len(sequence))
                ranges.append((start, stop))
                if stop == len(sequence):
                    break
        return tuple(
            ReferenceWindow(
                start,
                stop,
                self._post_process(sequence[start:stop]).input_ids,
            )
            for start, stop in ranges
        )

    def _sequence(self, text: str) -> list[_SequenceToken]:
        tokens: list[_SequenceToken] = []
        for start, end, special in self._split_added(text):
            if special is not None:
                tokens.append(_SequenceToken(special, self._added[special], start, end))
                continue
            chars = [
                _Char(text[index], index, index + 1) for index in range(start, end)
            ]
            for word in _pre_tokenize(self._normalize(chars)):
                tokens.extend(self._wordpiece(word))
        return tokens

    def _split_added(self, text: str) -> list[tuple[int, int, str | None]]:
        # Leftmost-longest matching on the original text, like the Rust
        # added-vocabulary splitter for non-normalized tokens.
        contents = sorted(self._added, key=len, reverse=True)
        segments: list[tuple[int, int, str | None]] = []
        cursor = 0
        index = 0
        while index < len(text):
            match = next(
                (token for token in contents if text.startswith(token, index)), None
            )
            if match is None:
                index += 1
                continue
            if cursor < index:
                segments.append((cursor, index, None))
            segments.append((index, index + len(match), match))
            index += len(match)
            cursor = index
        if cursor < len(text):
            segments.append((cursor, len(text), None))
        return segments

    def _normalize(self, chars: list[_Char]) -> list[_Char]:
        options = self._normalizer
        if options["clean_text"]:
            # Removed characters drop their alignment; whitespace maps to " ".
            chars = [
                _Char(
                    " " if _is_clean_whitespace(item.value) else item.value,
                    item.start,
                    item.end,
                )
                for item in chars
                if not _is_removed_by_clean_text(item.value)
            ]
        if options["handle_chinese_chars"]:
            padded: list[_Char] = []
            for item in chars:
                if _is_chinese_char(item.value):
                    padded.extend(
                        (
                            _Char(" ", item.start, item.end),
                            item,
                            _Char(" ", item.start, item.end),
                        )
                    )
                else:
                    padded.append(item)
            chars = padded
        if options["strip_accents"]:
            chars = _strip_accents(chars)
        if options["lowercase"]:
            # Per-character lowercasing, like Rust ``char::to_lowercase``.
            chars = [
                _Char(lowered, item.start, item.end)
                for item in chars
                for lowered in item.value.lower()
            ]
        return chars

    def _wordpiece(self, word: list[_Char]) -> list[_SequenceToken]:
        values = [item.value for item in word]
        unknown = [
            _SequenceToken(
                self._unk_token,
                self._vocab[self._unk_token],
                word[0].start,
                word[-1].end,
            )
        ]
        if len(values) > self._max_chars:
            return unknown
        pieces: list[_SequenceToken] = []
        start = 0
        while start < len(values):
            end = len(values)
            found: str | None = None
            while start < end:
                candidate = "".join(values[start:end])
                if start > 0:
                    candidate = self._prefix + candidate
                if candidate in self._vocab:
                    found = candidate
                    break
                end -= 1
            if found is None:
                return unknown
            pieces.append(
                _SequenceToken(
                    found,
                    self._vocab[found],
                    word[start].start,
                    word[end - 1].end,
                )
            )
            start = end
        return pieces

    def _post_process(self, sequence: Sequence[_SequenceToken]) -> ReferenceEncoding:
        tokens: list[str] = []
        ids: list[int] = []
        types: list[int] = []
        special_mask: list[int] = []
        offsets: list[tuple[int, int]] = []
        sequence_indices: list[int] = []
        for kind, value, type_id in self._template:
            if kind == "special":
                tokens.append(value)
                ids.append(self._vocab[value])
                types.append(type_id)
                special_mask.append(1)
                offsets.append((0, 0))
                continue
            for item in sequence:
                sequence_indices.append(len(tokens))
                tokens.append(item.token)
                ids.append(item.token_id)
                types.append(type_id)
                special_mask.append(0)
                offsets.append((item.start, item.end))
        return ReferenceEncoding(
            tokens=tuple(tokens),
            input_ids=tuple(ids),
            attention_mask=tuple(1 for _ in ids),
            token_type_ids=tuple(types),
            special_tokens_mask=tuple(special_mask),
            offsets=tuple(offsets),
            sequence_indices=tuple(sequence_indices),
        )


def _parse_normalizer(payload: Any) -> dict[str, bool]:
    if not isinstance(payload, Mapping) or payload.get("type") != "BertNormalizer":
        raise UnsupportedTokenizerFeature("normalizer")
    lowercase = payload.get("lowercase", True)
    strip_accents = payload.get("strip_accents")
    options = {
        "clean_text": payload.get("clean_text", True),
        "handle_chinese_chars": payload.get("handle_chinese_chars", True),
        "lowercase": lowercase,
        "strip_accents": lowercase if strip_accents is None else strip_accents,
    }
    if not all(isinstance(value, bool) for value in options.values()):
        raise UnsupportedTokenizerFeature("normalizer_options")
    return options


def _parse_added_tokens(payload: Any, vocab: Mapping[str, int]) -> dict[str, int]:
    if payload is None:
        return {}
    if not isinstance(payload, list):
        raise UnsupportedTokenizerFeature("added_tokens")
    added: dict[str, int] = {}
    for entry in payload:
        if not isinstance(entry, Mapping) or entry.get("special") is not True:
            raise UnsupportedTokenizerFeature("added_token_not_special")
        for flag, expected in _SUPPORTED_ADDED_TOKEN_FLAGS.items():
            if entry.get(flag, expected) != expected:
                raise UnsupportedTokenizerFeature("added_token_flags")
        content = entry.get("content")
        token_id = entry.get("id")
        if (
            not isinstance(content, str)
            or not content
            or vocab.get(content) != token_id
        ):
            raise UnsupportedTokenizerFeature("added_token_vocab")
        added[content] = int(token_id)
    return added


def _parse_template(
    payload: Any, vocab: Mapping[str, int]
) -> tuple[tuple[str, str, int], ...]:
    if payload is None:
        return (("sequence", "A", 0),)
    if not isinstance(payload, Mapping) or payload.get("type") != "TemplateProcessing":
        raise UnsupportedTokenizerFeature("post_processor")
    special_tokens = payload.get("special_tokens")
    single = payload.get("single")
    if not isinstance(special_tokens, Mapping) or not isinstance(single, list):
        raise UnsupportedTokenizerFeature("post_processor_template")
    pieces: list[tuple[str, str, int]] = []
    for item in single:
        if not isinstance(item, Mapping) or len(item) != 1:
            raise UnsupportedTokenizerFeature("post_processor_template")
        if "SpecialToken" in item:
            spec = item["SpecialToken"]
            token = str(spec.get("id"))
            declared = special_tokens.get(token)
            if (
                not isinstance(declared, Mapping)
                or list(declared.get("tokens", ())) != [token]
                or list(declared.get("ids", ())) != [vocab.get(token)]
            ):
                raise UnsupportedTokenizerFeature("post_processor_special_tokens")
            pieces.append(("special", token, int(spec.get("type_id", 0))))
        elif "Sequence" in item and item["Sequence"].get("id") == "A":
            pieces.append(("sequence", "A", int(item["Sequence"].get("type_id", 0))))
        else:
            raise UnsupportedTokenizerFeature("post_processor_template")
    if sum(1 for piece in pieces if piece[0] == "sequence") != 1:
        raise UnsupportedTokenizerFeature("post_processor_template")
    return tuple(pieces)


def _pre_tokenize(chars: Sequence[_Char]) -> list[list[_Char]]:
    words: list[list[_Char]] = []
    current: list[_Char] = []
    for item in chars:
        if item.value in _WHITE_SPACE:
            if current:
                words.append(current)
                current = []
        elif _is_bert_punctuation(item.value):
            if current:
                words.append(current)
                current = []
            words.append([item])
        else:
            current.append(item)
    if current:
        words.append(current)
    return words


def _strip_accents(chars: Sequence[_Char]) -> list[_Char]:
    decomposed = [
        _Char(part, item.start, item.end)
        for item in chars
        for part in unicodedata.normalize("NFD", item.value)
    ]
    whole = unicodedata.normalize("NFD", "".join(item.value for item in chars))
    if whole != "".join(item.value for item in decomposed):
        # Canonical reordering across characters would change alignments.
        raise UnsupportedTokenizerFeature("nfd_reordering")
    return [item for item in decomposed if unicodedata.category(item.value) != "Mn"]


def _is_removed_by_clean_text(char: str) -> bool:
    if char in "\t\n\r":
        return False
    return char in _REMOVED_BY_CLEAN_TEXT or unicodedata.category(char).startswith("C")


def _is_clean_whitespace(char: str) -> bool:
    return char in "\t\n\r" or char in _WHITE_SPACE


def _is_chinese_char(char: str) -> bool:
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in _CJK_RANGES)


def _is_bert_punctuation(char: str) -> bool:
    return char in _ASCII_PUNCTUATION or unicodedata.category(char).startswith("P")
