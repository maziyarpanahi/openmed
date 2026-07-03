"""Repository checks for terminology isolation."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
NORMALIZATION_ROOT = REPO_ROOT / "openmed" / "clinical" / "normalization"
PAYLOAD_SUFFIXES = {
    ".csv",
    ".db",
    ".json",
    ".jsonl",
    ".parquet",
    ".sqlite",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
RESTRICTED_EXPORT_MARKERS = (
    "MRCONSO",
    "RXNCONSO",
    "sct2_Description",
    "LOINC_NUM",
    "HCPCS",
)


def test_normalization_package_bundles_no_terminology_payload_files():
    payload_files = [
        path.relative_to(REPO_ROOT)
        for path in NORMALIZATION_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in PAYLOAD_SUFFIXES
    ]

    assert payload_files == []


def test_only_synthetic_backend_is_shipped_in_normalization_package():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in NORMALIZATION_ROOT.rglob("*.py")
        if path.is_file()
    )

    assert "SyntheticTerminologyBackend" in source
    for marker in RESTRICTED_EXPORT_MARKERS:
        assert marker not in source
