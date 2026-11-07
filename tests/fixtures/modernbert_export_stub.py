"""Offline tiny ModernBERT token classifier for export qualification tests.

Everything here is built in-test from seeded generators: no downloads, no
committed weights. The tokenizer is character level, so no fixture surface
form ever appears in a vocabulary file. The classifier is wired so that
uppercase letters decode as ``I-NAME`` and digits as ``I-ID_NUM`` while every
attention and MLP block keeps random weights. Exported graphs therefore still
exercise full and sliding-window attention, rotary embeddings, and padding
masks, and a broken mask or rotary rewrite shows up as logit drift.

The synthetic fixtures spell names in capitals and identifiers as digit runs,
so the reference checkpoint reaches full direct-identifier recall and a
degraded candidate is caught by the recall and leakage gates.
"""

from __future__ import annotations

import json
import socket
import string
from pathlib import Path
from typing import Any

SEED = 3636
LABELS = {0: "O", 1: "I-NAME", 2: "I-ID_NUM"}
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
PUNCTUATION = ".,:;-/#()'"

# Synthetic sentinels: surface forms that must never reach logs, errors,
# manifests, reports, or any other artifact written during export.
_ROWS = (
    ("patient {n0} was seen today, mrn {i0}.", ("ZORVAK QUILLAN",), ("44172290",)),
    ("{n0} called about the visit on unit {i0}.", ("MERIDETH ASHGROVE",), ("7731",)),
    (
        "dr {n0} reviewed chart {i0} for {n1}.",
        ("TOVIN BRAKKE", "ELSPETH NUROVA"),
        ("90021",),
    ),
    ("guardian {n0}, member id {i0}, consented.", ("QUESNA DORVILLE",), ("5528193",)),
    ("follow up with {n0} next week.", ("HALVARD PENNIMORE",), ()),
    ("record {i0} was merged into {i1}.", (), ("88310", "88311")),
)
SENTINELS = tuple(
    sorted({value for _, names, ids in _ROWS for value in (*names, *ids)})
)


def rule_fixture_rows() -> list[dict[str, Any]]:
    """Return synthetic fixtures whose gold spans match the wired classifier."""
    rows: list[dict[str, Any]] = []
    for index, (template, names, ids) in enumerate(_ROWS, start=1):
        values = {f"n{i}": name for i, name in enumerate(names)}
        values.update({f"i{i}": value for i, value in enumerate(ids)})
        text = template.format(**values)
        spans = []
        cursor = 0
        ordered = sorted(
            [(text.index(name), name, "PERSON") for name in names]
            + [(text.index(value), value, "ID_NUM") for value in ids]
        )
        for _, surface, label in ordered:
            start = text.index(surface, cursor)
            spans.append(
                {
                    "start": start,
                    "end": start + len(surface),
                    "label": label,
                    "text": surface,
                }
            )
            cursor = start + len(surface)
        rows.append(
            {
                "id": f"stub-{index:02d}",
                "language": "en",
                "text": text,
                "gold_spans": spans,
                "metadata": {"synthetic": True},
            }
        )
    return rows


def write_rule_fixtures(path: Path) -> Path:
    """Write :func:`rule_fixture_rows` as JSONL and return the path."""
    lines = [json.dumps(row, sort_keys=True) for row in rule_fixture_rows()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _vocabulary() -> dict[str, int]:
    symbols = [
        *SPECIAL_TOKENS,
        *string.ascii_lowercase,
        *string.ascii_uppercase,
        *string.digits,
        *PUNCTUATION,
    ]
    return {symbol: index for index, symbol in enumerate(symbols)}


def build_tokenizer(directory: Path) -> Any:
    """Save and return a character-level fast tokenizer."""
    from tokenizers import Regex, Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab = _vocabulary()
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Split(Regex("."), behavior="isolated"),
        ]
    )
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=128,
    )
    tokenizer.save_pretrained(str(directory))
    return tokenizer


def build_tiny_modernbert(directory: Path, *, seed: int = SEED) -> Path:
    """Write a deterministic tiny ModernBERT token classifier to *directory*.

    Returns:
        The checkpoint directory, loadable with ``from_pretrained``.
    """

    import torch
    from transformers import ModernBertConfig, ModernBertForTokenClassification

    directory.mkdir(parents=True, exist_ok=True)
    build_tokenizer(directory)
    vocab = _vocabulary()
    config = ModernBertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=128,
        local_attention=16,
        global_attn_every_n_layers=3,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        classifier_dropout=0.0,
        num_labels=len(LABELS),
        id2label=dict(LABELS),
        label2id={label: index for index, label in LABELS.items()},
        pad_token_id=vocab["[PAD]"],
        cls_token_id=vocab["[CLS]"],
        sep_token_id=vocab["[SEP]"],
        bos_token_id=vocab["[CLS]"],
        eos_token_id=vocab["[SEP]"],
    )
    torch.manual_seed(seed)
    model = ModernBertForTokenClassification(config).eval()
    generator = torch.Generator().manual_seed(seed)
    hidden = config.hidden_size
    with torch.no_grad():
        embedding = torch.randn(len(vocab), hidden, generator=generator) * 0.05
        for symbol, index in vocab.items():
            if symbol in string.ascii_uppercase:
                embedding[index, 0] += 3.0
            elif symbol in string.digits:
                embedding[index, 1] += 3.0
            else:
                embedding[index, 2] += 3.0
        model.model.embeddings.tok_embeddings.weight.copy_(embedding)
        for layer in model.model.layers:
            for linear, scale in (
                (layer.attn.Wqkv, 0.3),
                (layer.attn.Wo, 0.03),
                (layer.mlp.Wi, 0.3),
                (layer.mlp.Wo, 0.03),
            ):
                linear.weight.copy_(
                    torch.randn(linear.weight.shape, generator=generator) * scale
                )
        model.head.dense.weight.copy_(torch.eye(hidden))
        classifier = torch.zeros(len(LABELS), hidden)
        classifier[0, 2] = 1.0
        classifier[1, 0] = 1.0
        classifier[2, 1] = 1.0
        model.classifier.weight.copy_(classifier)
        model.classifier.bias.zero_()
    model.save_pretrained(str(directory))
    return directory


class NetworkGuard:
    """Record and refuse every outbound socket connection attempt."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def install(self, monkeypatch: Any) -> "NetworkGuard":
        def refuse(kind: str):
            def blocked(*_args: Any, **_kwargs: Any) -> Any:
                self.attempts.append(kind)
                raise OSError("network access is disabled in this test")

            return blocked

        monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
        monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
        monkeypatch.setattr(socket.socket, "connect", refuse("connect"))
        monkeypatch.setattr(socket.socket, "connect_ex", refuse("connect_ex"))
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            monkeypatch.setenv(name, "1")
        return self


def files_containing_sentinels(root: Path, sentinels: tuple[str, ...]) -> list[str]:
    """Return relative paths under *root* whose bytes contain a sentinel."""
    needles = [value.encode("utf-8") for value in sentinels]
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if any(needle in payload for needle in needles):
            hits.append(path.relative_to(root).as_posix())
    return hits


def text_contains_sentinels(text: str, sentinels: tuple[str, ...]) -> bool:
    """Return whether *text* contains any sentinel surface form."""
    return any(value in text for value in sentinels)
