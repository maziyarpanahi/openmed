"""Write a tiny invented corpus in the i2b2/VA 2010 file format.

Every note and annotation here is synthetic. Files are written only into the
directory the caller passes (a pytest ``tmp_path``); nothing derived from the
DUA corpus is committed. The layout mirrors the release, with text and each
annotation kind in its own folder, so the loader has to pair files by stem.
"""

from __future__ import annotations

from pathlib import Path

NOTES: dict[str, list[str]] = {
    "note-001": [
        "The patient reports chest pain and shortness of breath .",
        "An ECG showed ST elevation and aspirin was given for chest pain .",
        "There is no evidence of pneumonia on chest x-ray .",
        "Rule out myocardial infarction .",
        "Family history of diabetes in her mother .",
        "If the cough worsens , start albuterol .",
        "She has wheezing when climbing stairs .",
    ],
    "note-002": [
        "Blood cultures were drawn this morning .",
    ],
}

# (note, c text, start line:token, end line:token, type, assertion or None)
CONCEPTS: list[tuple[str, str, str, str, str, str | None]] = [
    ("note-001", "chest pain", "1:3", "1:4", "problem", "present"),
    ("note-001", "shortness of breath", "1:6", "1:8", "problem", "present"),
    ("note-001", "an ecg", "2:0", "2:1", "test", None),
    ("note-001", "st elevation", "2:3", "2:4", "problem", "present"),
    ("note-001", "aspirin", "2:6", "2:6", "treatment", None),
    ("note-001", "chest pain", "2:10", "2:11", "problem", "present"),
    ("note-001", "pneumonia", "3:5", "3:5", "problem", "absent"),
    ("note-001", "chest x-ray", "3:7", "3:8", "test", None),
    ("note-001", "myocardial infarction", "4:2", "4:3", "problem", "possible"),
    ("note-001", "diabetes", "5:3", "5:3", "problem", "associated_with_someone_else"),
    ("note-001", "the cough", "6:1", "6:2", "problem", "hypothetical"),
    ("note-001", "albuterol", "6:6", "6:6", "treatment", None),
    ("note-001", "wheezing", "7:2", "7:2", "problem", "conditional"),
    ("note-002", "blood cultures", "1:0", "1:1", "test", None),
]

# (note, head concept index into CONCEPTS, relation, tail concept index)
RELATIONS: list[tuple[str, int, str, int]] = [
    ("note-001", 0, "PIP", 1),
    ("note-001", 2, "TeRP", 3),
    ("note-001", 4, "TrAP", 5),
    ("note-001", 7, "TeCP", 6),
]

EXPECTED_CONCEPT_COUNTS = {"PROBLEM": 9, "LAB_TEST": 3, "OTHER": 2}


def concept_line(index: int) -> str:
    """Return the ``.con`` line for one entry of :data:`CONCEPTS`."""

    _, text, start, end, kind, _ = CONCEPTS[index]
    return f'c="{text}" {start} {end}||t="{kind}"'


def write_i2b2_2010_corpus(root: Path) -> Path:
    """Write the synthetic notes and their ``.con``/``.ast``/``.rel`` files."""

    for folder in ("txt", "concept", "ast", "rel"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for stem, lines in NOTES.items():
        (root / "txt" / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        indexes = [i for i, concept in enumerate(CONCEPTS) if concept[0] == stem]
        (root / "concept" / f"{stem}.con").write_text(
            "".join(concept_line(i) + "\n" for i in indexes), encoding="utf-8"
        )
        assertions = [
            f'{concept_line(i)}||a="{CONCEPTS[i][5]}"\n'
            for i in indexes
            if CONCEPTS[i][5] is not None
        ]
        relations = [
            _relation_line(head, relation, tail)
            for note, head, relation, tail in RELATIONS
            if note == stem
        ]
        if assertions:
            (root / "ast" / f"{stem}.ast").write_text(
                "".join(assertions), encoding="utf-8"
            )
        if relations:
            (root / "rel" / f"{stem}.rel").write_text(
                "".join(relations), encoding="utf-8"
            )
    return root


def _relation_line(head: int, relation: str, tail: int) -> str:
    _, head_text, head_start, head_end, _, _ = CONCEPTS[head]
    _, tail_text, tail_start, tail_end, _, _ = CONCEPTS[tail]
    return (
        f'c="{head_text}" {head_start} {head_end}||r="{relation}"||'
        f'c="{tail_text}" {tail_start} {tail_end}\n'
    )
