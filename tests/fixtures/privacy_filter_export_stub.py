"""Offline tiny Privacy Filter token classifier for export qualification tests.

Everything is built in-test from seeded generators: no downloads and no
committed weights. The checkpoint uses the Transformers-native
``openai_privacy_filter`` architecture with the upstream 33-label BIOES space,
so it is an eligible local source for the export adapter.

The tokenizer is character level and adds no special tokens (like the
``o200k_base`` encoding the real family uses), so no fixture surface form
ever appears in a vocabulary file. Characters that only occur inside gold
names steer the classifier towards ``private_person`` and characters that
only occur inside gold account numbers towards ``account_number``; inside
tags dominate, so the BIOES Viterbi decoder must place ``B``/``E`` boundaries
itself. Every attention, sink, router, and expert weight stays random, so a
broken mask, rotary rewrite, or expert routing shows up as logit drift.

The synthetic fixtures cover Latin, accented Latin, katakana, Han,
Devanagari-digit, and Arabic-Indic-digit identifiers.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

from tests.fixtures.modernbert_export_stub import (
    NetworkGuard,
    files_containing_sentinels,
    text_contains_sentinels,
)

SEED = 3638
SPAN_LABELS = (
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
)
LABELS = {
    index: label
    for index, label in enumerate(
        ("O",) + tuple(f"{tag}-{name}" for name in SPAN_LABELS for tag in "BIES")
    )
}
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
VITERBI_BIASES = {
    "transition_bias_background_stay": 0.05,
    "transition_bias_background_to_start": -0.05,
    "transition_bias_end_to_background": 0.02,
    "transition_bias_end_to_start": -0.02,
    "transition_bias_inside_to_continue": 0.03,
    "transition_bias_inside_to_end": 0.01,
}

# (language, template, names, account numbers). Names and identifiers are
# synthetic; their characters never occur elsewhere in the suite.
_ROWS = (
    (
        "en",
        "patient {n0} was seen today, account {i0}.",
        ("ZORVAK QUILLAN",),
        ("44172290",),
    ),
    (
        "en",
        "{n0} called about invoice {i0} yesterday.",
        ("MERIDETH ASHGROVE",),
        ("77310045",),
    ),
    (
        "fr",
        "la patiente {n0} a été reçue, dossier {i0}.",
        ("ÉLODIE VARNESSE",),
        ("55281930",),
    ),
    (
        "de",
        "der patient {n0} kam heute, konto {i0}.",
        ("JÖRG BRÄUNLICH",),
        ("90021377",),
    ),
    ("es", "la señora {n0} pagó la cuenta {i0}.", ("NÚRIA OCAÑAVERDE",), ("31415926",)),
    (
        "ja",
        "患者{n0}さんは本日来院しました。番号{i0}。",
        ("ゾルヴァク・キラン",),
        ("88310442",),
    ),
    ("zh", "患者{n0}今日就诊，病历号{i0}。", ("欧阳璇珂宁",), ("60718293",)),
    ("hi", "रोगी {n0} का खाता {i0} है।", ("RAVIKIRAN THAPAR",), ("४४१७२२९०",)),
    ("ar", "المريض {n0} رقم الحساب {i0}.", ("TOVIN BRAKKE",), ("٧٧٣١٠٠٤٥",)),
    ("en", "follow up next week with the clinic.", (), ()),
)
SENTINELS = tuple(
    sorted({value for _, _, names, ids in _ROWS for value in (*names, *ids)})
)


def rule_fixture_rows() -> list[dict[str, Any]]:
    """Return synthetic multilingual fixtures matching the wired classifier."""
    rows: list[dict[str, Any]] = []
    for index, (language, template, names, ids) in enumerate(_ROWS, start=1):
        values = {f"n{i}": name for i, name in enumerate(names)}
        values.update({f"i{i}": value for i, value in enumerate(ids)})
        text = template.format(**values)
        spans = []
        for surface, label in [(name, "PERSON") for name in names] + [
            (value, "ACCOUNT_NUMBER") for value in ids
        ]:
            start = text.index(surface)
            spans.append(
                {
                    "start": start,
                    "end": start + len(surface),
                    "label": label,
                    "text": surface,
                }
            )
        rows.append(
            {
                "id": f"pf-stub-{index:02d}",
                "language": language,
                "text": text,
                "gold_spans": sorted(spans, key=lambda span: span["start"]),
                "metadata": {"synthetic": True},
            }
        )
    return rows


def write_rule_fixtures(path: Path) -> Path:
    """Write :func:`rule_fixture_rows` as JSONL and return the path."""
    lines = [
        json.dumps(row, sort_keys=True, ensure_ascii=False)
        for row in rule_fixture_rows()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def character_classes() -> dict[str, str]:
    """Return ``{character: "person" | "id" | "other"}`` for the suite.

    Raises:
        AssertionError: If a character occurs both inside and outside gold
            spans of different kinds, which would make the stub ambiguous.
    """

    classes: dict[str, str] = {}
    for row in rule_fixture_rows():
        text = row["text"]
        inside: dict[int, str] = {}
        for span in row["gold_spans"]:
            kind = "person" if span["label"] == "PERSON" else "id"
            for position in range(span["start"], span["end"]):
                inside[position] = kind
        for position, char in enumerate(text):
            if char.isspace():
                continue
            kind = inside.get(position, "other")
            previous = classes.setdefault(char, kind)
            assert previous == kind, "stub fixture characters must be unambiguous"
    for char in string.ascii_lowercase + ".,":
        classes.setdefault(char, "other")
    return classes


def _vocabulary() -> dict[str, int]:
    symbols = [PAD_TOKEN, UNK_TOKEN, *sorted(character_classes())]
    return {symbol: index for index, symbol in enumerate(symbols)}


def build_tokenizer(directory: Path) -> Any:
    """Save and return a character-level fast tokenizer without specials."""
    from tokenizers import Regex, Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    vocab = _vocabulary()
    backend = Tokenizer(WordLevel(vocab, unk_token=UNK_TOKEN))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Split(Regex("."), behavior="isolated"),
        ]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        model_max_length=256,
    )
    tokenizer.save_pretrained(str(directory))
    return tokenizer


def tiny_config(**overrides: Any) -> Any:
    """Return a tiny native Privacy Filter configuration."""
    from transformers import OpenAIPrivacyFilterConfig

    vocab = _vocabulary()
    values: dict[str, Any] = {
        "vocab_size": len(vocab),
        "hidden_size": 32,
        "intermediate_size": 16,
        "head_dim": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "sliding_window": 4,
        "max_position_embeddings": 256,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 150000.0,
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 4096,
        },
        "id2label": dict(LABELS),
        "label2id": {label: index for index, label in LABELS.items()},
        "pad_token_id": vocab[PAD_TOKEN],
        "eos_token_id": vocab[PAD_TOKEN],
        "classifier_dropout": 0.0,
    }
    values.update(overrides)
    return OpenAIPrivacyFilterConfig(**values)


def build_tiny_privacy_filter(
    directory: Path,
    *,
    seed: int = SEED,
    calibration: bool = True,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a deterministic tiny Privacy Filter checkpoint to *directory*.

    Returns:
        The checkpoint directory, loadable with ``from_pretrained``.
    """

    import torch
    from transformers import OpenAIPrivacyFilterForTokenClassification

    directory.mkdir(parents=True, exist_ok=True)
    build_tokenizer(directory)
    vocab = _vocabulary()
    classes = character_classes()
    config = tiny_config(**(config_overrides or {}))
    torch.manual_seed(seed)
    model = OpenAIPrivacyFilterForTokenClassification(config).eval()
    generator = torch.Generator().manual_seed(seed)
    hidden = config.hidden_size

    def randn(*shape: int, scale: float) -> Any:
        return torch.randn(*shape, generator=generator) * scale

    feature = {"person": 0, "id": 1, "other": 2}
    with torch.no_grad():
        embedding = randn(len(vocab), hidden, scale=0.05)
        for symbol, index in vocab.items():
            embedding[index, feature[classes.get(symbol, "other")]] += 3.0
        model.model.embed_tokens.weight.copy_(embedding)
        for layer in model.model.layers:
            attention = layer.self_attn
            for linear, scale in (
                (attention.q_proj, 0.4),
                (attention.k_proj, 0.4),
                (attention.v_proj, 0.4),
                (attention.o_proj, 0.03),
            ):
                linear.weight.copy_(randn(*linear.weight.shape, scale=scale))
                linear.bias.copy_(randn(*linear.bias.shape, scale=0.01))
            attention.sinks.copy_(randn(*attention.sinks.shape, scale=0.5))
            router = layer.mlp.router
            router.weight.copy_(randn(*router.weight.shape, scale=0.6))
            router.bias.copy_(randn(*router.bias.shape, scale=0.1))
            experts = layer.mlp.experts
            experts.gate_up_proj.copy_(randn(*experts.gate_up_proj.shape, scale=0.3))
            experts.gate_up_proj_bias.copy_(
                randn(*experts.gate_up_proj_bias.shape, scale=0.05)
            )
            experts.down_proj.copy_(randn(*experts.down_proj.shape, scale=0.02))
            experts.down_proj_bias.copy_(
                randn(*experts.down_proj_bias.shape, scale=0.005)
            )
            layer.input_layernorm.weight.fill_(1.0)
            layer.post_attention_layernorm.weight.fill_(1.0)
        model.model.norm.weight.fill_(1.0)
        classifier = torch.zeros(len(LABELS), hidden)
        classifier[0, feature["other"]] = 1.45
        for name, kind in (("private_person", "person"), ("account_number", "id")):
            for tag, weight in (("B", 0.7), ("I", 1.45), ("E", 0.7), ("S", 0.35)):
                label_id = next(
                    index for index, label in LABELS.items() if label == f"{tag}-{name}"
                )
                classifier[label_id, feature[kind]] = weight
        model.score.weight.copy_(classifier)
        model.score.bias.zero_()
    model.save_pretrained(str(directory))
    if calibration:
        (directory / "viterbi_calibration.json").write_text(
            json.dumps(
                {"operating_points": {"default": {"biases": VITERBI_BIASES}}},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return directory


def content_revision(directory: Path) -> str:
    """Return the ``sha256:`` content revision of a local checkpoint."""
    from transformers import AutoConfig

    from openmed.eval.export_qualification import source_identity

    config = AutoConfig.from_pretrained(directory)
    return str(source_identity(directory, config=config)["revision"])


__all__ = [
    "LABELS",
    "SENTINELS",
    "VITERBI_BIASES",
    "NetworkGuard",
    "build_tiny_privacy_filter",
    "build_tokenizer",
    "character_classes",
    "content_revision",
    "files_containing_sentinels",
    "rule_fixture_rows",
    "text_contains_sentinels",
    "tiny_config",
    "write_rule_fixtures",
]
