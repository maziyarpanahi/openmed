"""
Compare tokenization strategies on hard clinical text:
1) HF WordPiece (OpenMed model tokenizer)
2) spaCy tokenizer configured with SciSpaCy-style rules
3) Hugging Face `tokenizers` pre-tokenizer port of the same rules

Run:
    .venv-openmed/bin/python examples/eval_tokenization_comparison.py

Outputs token lists (with offsets) per sentence so you can see where
boundaries diverge.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Iterable

import spacy
import torch
from spacy.lang import char_classes
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex

from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Split, Sequence as PreSeq

from openmed.core.models import ModelLoader


# --------------------------------------------------------------------------- #
# SciSpaCy-like rules
# --------------------------------------------------------------------------- #

# Minimal abbreviation set to keep dots attached (SciSpaCy uses a larger list)
ABBREVIATIONS = {"vs.", "dr.", "prof.", "pt.", "hr.", "min.", "sec.", "mg.", "ml."}


def _combined_rule_prefixes() -> List[str]:
    prefix_punct = char_classes.PUNCT.replace("|", " ")
    prefix_punct = prefix_punct.replace(r"\(", r"\((?![^\(\s]+\)\S+)")
    prefix_punct = prefix_punct.replace(r"\[", r"\[(?![^\[\s]+\]\S+)")
    prefix_punct = prefix_punct.replace(r"\{", r"\{(?![^\{\s]+\}\S+)")
    prefixes = (
        ["§", "%", "=", r"\+"]
        + char_classes.split_chars(prefix_punct)
        + char_classes.LIST_ELLIPSES
        + char_classes.LIST_QUOTES
        + char_classes.LIST_CURRENCY
        + char_classes.LIST_ICONS
    )
    return prefixes


def build_scispacy_spacy_tokenizer() -> Tokenizer:
    """SpaCy tokenizer that mirrors SciSpaCy's custom rules."""
    nlp = spacy.blank("en")
    hyphens = char_classes.HYPHENS.replace("-|", "", 1)

    infixes = (
        char_classes.LIST_ELLIPSES
        + char_classes.LIST_ICONS
        + [
            r"×",
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            rf"(?<=[{char_classes.ALPHA_LOWER}])\.(?=[{char_classes.ALPHA_UPPER}])",
            rf"(?<=[{char_classes.ALPHA}]),(?=[{char_classes.ALPHA}])",
            rf'(?<=[{char_classes.ALPHA}])[?";:=,.]*(?:{hyphens})(?=[{char_classes.ALPHA}])',
            rf'(?<=[{char_classes.ALPHA}"])[:<>=](?=[{char_classes.ALPHA}])',
        ]
    )

    prefixes = _combined_rule_prefixes()

    quotes = char_classes.LIST_QUOTES.copy() + ["’"]
    suffix_punct = char_classes.PUNCT.replace("|", " ")
    suffixes = (
        char_classes.split_chars(suffix_punct)
        + char_classes.LIST_ELLIPSES
        + quotes
        + char_classes.LIST_ICONS
        + ["'s", "'S", "’s", "’S", "’s", "’S"]
        + [
            r"(?<=[0-9])\+",
            r"(?<=°[FfCcKk])\.",
            rf"(?<=[0-9])(?:{char_classes.CURRENCY})",
            rf"(?<=[0-9])(?:{char_classes.UNITS})",
            rf"(?<=[0-9{char_classes.ALPHA_LOWER}%²\-\)\]\+{'|'.join(quotes)}])\.",
            rf"(?<=[{char_classes.ALPHA_UPPER}|\d][{char_classes.ALPHA_UPPER}])\.",
        ]
    )

    infix_re = compile_infix_regex(infixes)
    prefix_re = compile_prefix_regex(prefixes)
    suffix_re = compile_suffix_regex(suffixes)

    tokenizer_exceptions = nlp.Defaults.tokenizer_exceptions.copy()
    tokenizer_exceptions.update({abbr: [{spacy.symbols.ORTH: abbr}] for abbr in ABBREVIATIONS})

    return Tokenizer(
        nlp.vocab,
        tokenizer_exceptions,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.tokenizer.token_match,  # type: ignore
    )


def build_scispacy_hf_pre_tokenizer() -> PreSeq:
    """Port of the above rules to `tokenizers` Split pre-tokenizers."""
    prefix_pattern = "(" + "|".join(_combined_rule_prefixes()) + ")"
    hyphens = char_classes.HYPHENS.replace("-|", "", 1)
    infix_patterns = [
        r"…", r"\.\.\.", r"×",
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        rf"(?<=[{char_classes.ALPHA_LOWER}])\.(?=[{char_classes.ALPHA_UPPER}])",
        rf"(?<=[{char_classes.ALPHA}]),(?=[{char_classes.ALPHA}])",
        rf'(?<=[{char_classes.ALPHA}])[?";:=,.]*(?:{hyphens})(?=[{char_classes.ALPHA}])',
        rf'(?<=[{char_classes.ALPHA}"])[:<>=](?=[{char_classes.ALPHA}])',
    ]
    suffix_patterns = [
        r"'s", r"'S", r"’s", r"’S",
        r"(?<=[0-9])\+",
        r"(?<=°[FfCcKk])\.",
        rf"(?<=[0-9])(?:{char_classes.CURRENCY})",
        rf"(?<=[0-9])(?:{char_classes.UNITS})",
        rf"(?<=[0-9{char_classes.ALPHA_LOWER}%²\-\)\]\+])\.",
        rf"(?<=[{char_classes.ALPHA_UPPER}|\d][{char_classes.ALPHA_UPPER}])\.",
    ]

    return PreSeq(
        [
            pre_tokenizers.Whitespace(),
            Split(prefix_pattern, behavior="isolated"),
            Split("|".join(infix_patterns), behavior="isolated"),
            Split("|".join(suffix_patterns), behavior="isolated"),
        ]
    )


# --------------------------------------------------------------------------- #
# Evaluation corpus
# --------------------------------------------------------------------------- #

CORPUS = [
    "t(8;21) AML patient on daunorubicin/cytarabine 90mg/m2/day x3; WBC 12.3 x10^3/µL, HbA1c 7.5%.",
    "IL-6-mediated cytokine storm post-CAR-T; given tocilizumab 8mg/kg and methylpred 1 mg/kg.",
    "COVID-19+ pt with O2 sat 88%, on 40% FiO2 via HFNC; dexamethasone 6 mg qd started.",
]


# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #

def run_model_with_offsets(text: str):
    """Tokenize with the model tokenizer and return tokens, offsets, labels."""
    loader = ModelLoader()
    bundle = loader.load_model("oncology_detection_tiny")
    tok = bundle["tokenizer"]
    model = bundle["model"].eval()

    encoded = tok(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        padding=False,
    )
    offsets = [tuple(x) for x in encoded.pop("offset_mapping")[0].tolist()]

    with torch.no_grad():
        logits = model(**encoded).logits

    pred_ids = logits.argmax(-1)[0].tolist()
    id2label = model.config.id2label
    tokens = tok.convert_ids_to_tokens(encoded["input_ids"][0])

    spans = []
    for tok_str, (s, e), pid in zip(tokens, offsets, pred_ids):
        if s == e:
            continue  # special tokens
        spans.append((tok_str, s, e, id2label[pid]))
    return spans


def spacy_tokens(tok: Tokenizer, text: str):
    return [(t.text, t.idx, t.idx + len(t.text)) for t in tok(text)]


def pretokenizer_tokens(pre_tok: PreSeq, text: str):
    raw = [(tok, start, end) for tok, (start, end) in pre_tok.pre_tokenize_str(text)]

    merged = []
    i = 0
    while i < len(raw):
        tok, s, e = raw[i]
        # Merge biomedical hyphenated patterns like COVID-19, IL-6, BCR-ABL1, post-CAR-T
        if (
            i + 2 < len(raw)
            and raw[i + 1][0] == "-"
            and re.match(r"^[A-Za-z]{2,}$", tok)
            and re.match(r"^[A-Za-z0-9]{1,}$", raw[i + 2][0])
        ):
            # Base pattern: word - word/number
            merged_tok = tok + "-" + raw[i + 2][0]
            merged.append((merged_tok, s, raw[i + 2][2]))
            i += 3
            # Handle trailing "-XYZ" chains (e.g., post-CAR-T)
            while (
                i + 1 < len(raw)
                and raw[i][0] == "-"
                and re.match(r"^[A-Za-z0-9]{1,}$", raw[i + 1][0])
            ):
                merged_tok += "-" + raw[i + 1][0]
                merged[-1] = (merged_tok, merged[-1][1], raw[i + 1][2])
                i += 2
            continue
        merged.append((tok, s, e))
        i += 1

    return merged


def map_labels(target_spans: Iterable[Tuple[str, int, int]], wp_spans: List[Tuple[str, int, int, str]]):
    """Assign labels to target spans by overlapping WordPiece-labeled spans."""
    labeled = []
    for tok, s, e in target_spans:
        overlaps = [lab for _, ws, we, lab in wp_spans if ws < e and we > s]
        if not overlaps:
            label = "O"
        else:
            # majority vote preferring non-O
            counts: Dict[str, int] = {}
            for lab in overlaps:
                counts[lab] = counts.get(lab, 0) + 1
            # sort by count then put O last
            label = sorted(counts.items(), key=lambda x: (-x[1], x[0] == "O"))[0][0]
        labeled.append((tok, s, e, label))
    return labeled


def print_table(title: str, rows: List[Tuple[str, int, int, str]]):
    col_widths = [max(len(str(r[i])) for r in rows + [("token", 0, 0, "label")]) for i in range(4)]
    header = f"{'token'.ljust(col_widths[0])}  {'start'.ljust(col_widths[1])}  {'end'.ljust(col_widths[2])}  label"
    print(f"{title}\n{header}")
    print("-" * len(header))
    for tok, s, e, lab in rows:
        print(f"{tok.ljust(col_widths[0])}  {str(s).ljust(col_widths[1])}  {str(e).ljust(col_widths[2])}  {lab}")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    spacy_tok = build_scispacy_spacy_tokenizer()
    pre_tok = build_scispacy_hf_pre_tokenizer()

    for idx, text in enumerate(CORPUS, 1):
        print(f"\n=== Example {idx} ===")
        print(text)
        wp_spans = run_model_with_offsets(text)

        spacy_spans = spacy_tokens(spacy_tok, text)
        port_spans = pretokenizer_tokens(pre_tok, text)

        wp_labeled = wp_spans
        spacy_labeled = map_labels(spacy_spans, wp_spans)
        port_labeled = map_labels(port_spans, wp_spans)

        print_table("HF WordPiece (tokens + labels)", wp_labeled)
        print_table("spaCy SciSpaCy-style", spacy_labeled)
        print_table("HF `tokenizers` port", port_labeled)
        print("-" * 60)


if __name__ == "__main__":
    main()
