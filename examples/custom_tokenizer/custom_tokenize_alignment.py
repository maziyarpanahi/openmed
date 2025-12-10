"""Custom tokenization with alignment back to OpenMed encoder outputs.

Run:
    python examples/custom_tokenize_alignment.py

This keeps a lightweight custom tokenizer (regex-based) for clinical text,
feeds the original text through an OpenMed HF encoder, and remaps the model's
wordpiece predictions back onto the custom tokens via span overlap.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import torch

from openmed.core.models import ModelLoader


# --- Custom tokenization (fast, dependency-free) ----------------------------

TOKEN_PATTERN = re.compile(
    r"\d+\.?\d*%?"  # numbers/measurements like 5, 5.0, 5mg, 92%
    r"|[A-Za-z]+(?:-[A-Za-z0-9]+)*"  # words, allowing hyphen chains (COVID-19)
    r"|[^\s]"  # any other non-space char as its own token
)


@dataclass
class CustomToken:
    text: str
    start: int
    end: int


def custom_tokenize(text: str) -> List[CustomToken]:
    """Regex tokenizer that returns text plus character offsets."""
    return [CustomToken(m.group(0), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


# --- Alignment logic --------------------------------------------------------

def map_wordpieces_to_custom(
    wordpiece_tokens: List[str],
    wp_offsets: List[Tuple[int, int]],
    pred_ids: List[int],
    id2label: dict,
    custom_tokens: List[CustomToken],
) -> List[Tuple[str, str]]:
    """Project wordpiece predictions back to custom tokens via span overlap."""

    mapped: List[Tuple[str, str]] = []
    seen = set()
    for tok, (start, end), pred in zip(wordpiece_tokens, wp_offsets, pred_ids):
        if start == end:  # special tokens like [CLS]/[SEP]
            continue
        for ct in custom_tokens:
            if start < ct.end and end > ct.start:  # spans intersect
                key = (ct.start, ct.end)
                if key in seen:
                    break
                seen.add(key)
                mapped.append((ct.text, id2label[pred]))
                break
    return mapped


# --- Demo -------------------------------------------------------------------

def main():
    # Use a text that actually contains oncology/clinical entities.
    text = (
        "58-year-old male with acute myeloid leukemia relapsed after induction; "
        "now on doxorubicin and cytarabine chemotherapy, reported neutropenic fever."
    )

    # Allow overriding the model from the environment (default: oncology SuperClinical)
    model_key = os.getenv("OPENMED_MODEL", "oncology_detection_superclinical")

    # 1) Custom tokens for downstream UI/analysis
    custom_tokens = custom_tokenize(text)

    # 2) Load a fast OpenMed encoder (non-GLiNER)
    loader = ModelLoader()
    bundle = loader.load_model(model_key)
    tokenizer = bundle["tokenizer"]
    model = bundle["model"].eval()

    # 3) Run the original text through the HF tokenizer/model
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        padding=False,
    )

    offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]

    with torch.no_grad():
        logits = model(**encoded).logits

    pred_ids = logits.argmax(-1)[0].tolist()
    wp_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    id2label = model.config.id2label

    mapped = map_wordpieces_to_custom(wp_tokens, offsets, pred_ids, id2label, custom_tokens)

    # --- Display ----------------------------------------------------------------
    print(f"Model: {model_key}\n")

    print("Custom tokens (before):")
    print(" ".join([ct.text for ct in custom_tokens]))
    print()

    print("WordPiece predictions (model view):")
    for tok, (s, e), pred in zip(wp_tokens, offsets, pred_ids):
        if s == e:
            continue
        print(f"{tok:20s} [{s:02d},{e:02d}] -> {id2label[pred]}")
    print()

    print("Remapped to custom tokens (after):")
    for tok, label in mapped:
        marker = "*" if label != "O" else " "
        print(f"{marker} {tok:18s} -> {label}")

    print("\nLegend: '*' marks tokens whose label changed from O after alignment.")


if __name__ == "__main__":
    main()
