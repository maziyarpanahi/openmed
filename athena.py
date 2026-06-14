"""
openmed.interop.athena
~~~~~~~~~~~~~~~~~~~~~~
Loaders for user-downloaded OHDSI Athena vocabulary bundles and Usagi mapping
exports.

**License notice**
Athena vocabulary content is © OHDSI contributors and distributed under the
Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
licence.  Restricted vocabularies (e.g. CPT-4, SNOMED US Edition) are
user-supplied only; OpenMed never bundles them.  When loading vocabulary data
the caller is responsible for complying with the licence of each vocabulary_id
present in the export.  Provenance is recorded in the returned index under the
``_meta`` key.

**No mandatory extra dependencies** – this module uses only the Python
standard-library ``csv`` module.  If you wish to use a pandas-based workflow,
install ``openmed[omop]`` (adds ``pandas``), though it is not required here.

Typical usage::

    from openmed.interop.athena import load_athena_vocab, load_usagi_mapping

    vocab = load_athena_vocab("/path/to/athena_download/")
    # vocab["SNOMED"]["73211009"] -> {"concept_id": 201826, "concept_name": "...", ...}

    mapping = load_usagi_mapping("/path/to/usagi_export.csv")
    # mapping["ICD10CM:E11.9"] -> 201826
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Optional, Union

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

# concept index:  vocab_id -> concept_code -> concept_dict
ConceptIndex = Dict[str, Dict[str, dict]]

# usagi mapping:  "VOCAB:code" -> standard_concept_id (int)
UsagiMapping = Dict[str, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_tsv(path: Path):
    """Open a tab-delimited Athena CSV file.

    Returns ``(file_handle, DictReader)`` so the caller can use the file handle
    as a context manager::

        fh, reader = _open_tsv(path)
        with fh:
            for row in reader:
                ...
    """
    fh = open(path, encoding="utf-8", newline="")
    reader = csv.DictReader(fh, delimiter="\t")
    return fh, reader


def _resolve_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Athena export path does not exist: {p}")
    if p.is_file():
        # Caller may pass the CONCEPT.csv directly – use its parent directory.
        p = p.parent
    return p


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_athena_vocab(
    path: Union[str, Path],
    *,
    include_synonyms: bool = True,
    vocabulary_ids: Optional[list] = None,
) -> ConceptIndex:
    """Load a user-downloaded Athena vocabulary export into a concept index.

    Parameters
    ----------
    path:
        Path to the directory that contains ``CONCEPT.csv`` (and optionally
        ``CONCEPT_SYNONYM.csv``) as produced by the Athena download portal
        (https://athena.ohdsi.org).  A path to ``CONCEPT.csv`` directly is
        also accepted – the parent directory is used.
    include_synonyms:
        When *True* (default) and ``CONCEPT_SYNONYM.csv`` is present, synonym
        strings are added to each concept entry under the ``"synonyms"`` key.
    vocabulary_ids:
        Optional allow-list of vocabulary IDs to load (e.g.
        ``["SNOMED", "LOINC"]``).  *None* loads every vocabulary present in
        the file.

    Returns
    -------
    ConceptIndex
        Nested dict ``{vocabulary_id: {concept_code: concept_dict}}``.

        Each ``concept_dict`` contains at minimum:

        * ``concept_id``       – int  (OMOP standard concept_id)
        * ``concept_name``     – str
        * ``domain_id``        – str
        * ``vocabulary_id``    – str
        * ``concept_class_id`` – str
        * ``standard_concept`` – str | None  (``"S"`` = standard, ``"C"`` = classification)
        * ``concept_code``     – str
        * ``synonyms``         – list[str]  (populated when ``include_synonyms=True``)

        A special ``"_meta"`` entry at the top level records provenance::

            index["_meta"] = {
                "source": str(path),
                "vocabulary_ids": [...],
                "licence": "CC BY-SA 4.0 (user-supplied; restricted vocabs are user-side only)",
            }

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist or ``CONCEPT.csv`` is absent.
    """
    export_dir = _resolve_dir(path)
    concept_csv = export_dir / "CONCEPT.csv"
    synonym_csv = export_dir / "CONCEPT_SYNONYM.csv"

    if not concept_csv.exists():
        raise FileNotFoundError(
            f"CONCEPT.csv not found in {export_dir}. "
            "Download the vocabulary bundle from https://athena.ohdsi.org and "
            "point this function at the extracted directory."
        )

    vocab_set = set(vocabulary_ids) if vocabulary_ids else None
    index: ConceptIndex = {}

    # -- 1. Parse CONCEPT.csv ------------------------------------------------
    # concept_id -> (vocab_id, code) lookup used to attach synonyms efficiently
    concept_id_to_key: dict = {}

    fh, reader = _open_tsv(concept_csv)
    with fh:
        for row in reader:
            vid = row.get("vocabulary_id", "").strip()
            if vocab_set and vid not in vocab_set:
                continue
            code = row.get("concept_code", "").strip()
            if not code:
                continue
            cid_raw = row.get("concept_id", "").strip()
            try:
                cid = int(cid_raw)
            except ValueError:
                cid = 0

            entry = {
                "concept_id": cid,
                "concept_name": row.get("concept_name", "").strip(),
                "domain_id": row.get("domain_id", "").strip(),
                "vocabulary_id": vid,
                "concept_class_id": row.get("concept_class_id", "").strip(),
                "standard_concept": row.get("standard_concept") or None,
                "concept_code": code,
                "synonyms": [],
            }
            index.setdefault(vid, {})[code] = entry
            if cid:
                concept_id_to_key[cid] = (vid, code)

    # -- 2. Parse CONCEPT_SYNONYM.csv (optional) -----------------------------
    if include_synonyms and synonym_csv.exists():
        fh2, reader2 = _open_tsv(synonym_csv)
        with fh2:
            for row in reader2:
                cid_raw = row.get("concept_id", "").strip()
                try:
                    cid = int(cid_raw)
                except ValueError:
                    continue
                key = concept_id_to_key.get(cid)
                if key is None:
                    continue
                vid, code = key
                syn = row.get("concept_synonym_name", "").strip()
                if syn:
                    index[vid][code]["synonyms"].append(syn)

    # -- 3. Record provenance ------------------------------------------------
    loaded_vids = sorted(index.keys())
    index["_meta"] = {  # type: ignore[assignment]
        "source": str(export_dir),
        "vocabulary_ids": loaded_vids,
        "licence": (
            "CC BY-SA 4.0 (OHDSI Athena, user-supplied). "
            "Restricted vocabularies (e.g. CPT-4) are user-side only; "
            "OpenMed does not bundle them."
        ),
    }

    return index


def load_usagi_mapping(
    path: Union[str, Path],
    *,
    min_equivalence: Optional[str] = None,
) -> UsagiMapping:
    """Load an OHDSI Usagi mapping export into a source-to-standard concept dict.

    Usagi (https://github.com/OHDSI/Usagi) produces CSV exports where each row
    maps a source code to an OMOP standard ``concept_id``.  Only rows with
    ``mappingStatus == "APPROVED"`` are included by default; rows mapped to
    concept_id ``0`` (unmapped) are skipped.

    Parameters
    ----------
    path:
        Path to the Usagi export CSV file.
    min_equivalence:
        Optional equivalence level filter.  Usagi stores equivalence as a
        string (``"EQUIVALENT"``, ``"BROADER"``, ``"NARROWER"``,
        ``"INEXACT"``, ``"UNREVIEWED"``).  Pass e.g. ``"EQUIVALENT"`` to keep
        only exact matches.  *None* (default) keeps all approved rows
        regardless of equivalence.

    Returns
    -------
    UsagiMapping
        Dict ``{"VOCAB:sourceCode": standard_concept_id}``.
        The key format mirrors common OMOP conventions; if the export does not
        include a ``sourceVocabularyId`` column the key is just the raw
        ``sourceCode``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Usagi mapping file not found: {p}")

    mapping: UsagiMapping = {}

    with open(p, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            status = row.get("mappingStatus", "").strip().upper()
            if status != "APPROVED":
                continue

            cid_raw = row.get("conceptId", row.get("concept_id", "")).strip()
            try:
                cid = int(cid_raw)
            except ValueError:
                continue
            if cid == 0:
                continue  # unmapped / explicitly rejected

            if min_equivalence:
                equiv = row.get("equivalence", "").strip().upper()
                if equiv != min_equivalence.upper():
                    continue

            src_code = row.get("sourceCode", row.get("source_code", "")).strip()
            src_vocab = row.get(
                "sourceVocabularyId", row.get("source_vocabulary_id", "")
            ).strip()
            key = f"{src_vocab}:{src_code}" if src_vocab else src_code

            mapping[key] = cid

    return mapping
