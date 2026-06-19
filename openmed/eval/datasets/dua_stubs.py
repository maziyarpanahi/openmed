"""Eval-only stubs for corpora that require a local data-use agreement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .public import DatasetLoadResult


DUA_GATED_CORPORA: tuple[str, ...] = (
    "i2b2",
    "n2c2",
    "shac",
    "thyme",
    "mednli",
    "made",
    "mimic",
)


class DUACredentialRequired(PermissionError):
    """Raised when a gated corpus is requested without a credentialed path."""


@dataclass(frozen=True)
class DUACorpusStub:
    name: str
    eval_only: bool = True

    def load(self, credentialed_path: str | Path | None = None) -> DatasetLoadResult:
        if credentialed_path is None:
            raise DUACredentialRequired(
                f"{self.name} requires a credentialed local path and cannot be bundled"
            )
        path = Path(credentialed_path)
        if not path.exists():
            raise DUACredentialRequired(
                f"{self.name} credentialed path does not exist: {path}"
            )
        return DatasetLoadResult(
            dataset=self.name,
            records=(),
            skipped=True,
            reason="eval-only gated corpus stub; local loader is intentionally not bundled",
        )


def dua_stub_for(name: str) -> DUACorpusStub:
    key = name.lower()
    if key not in DUA_GATED_CORPORA:
        raise ValueError(f"unknown gated corpus: {name}")
    return DUACorpusStub(key)


def load_dua_corpus(name: str, credentialed_path: str | Path | None = None) -> DatasetLoadResult:
    return dua_stub_for(name).load(credentialed_path)


def all_dua_stubs() -> Mapping[str, DUACorpusStub]:
    return {name: DUACorpusStub(name) for name in DUA_GATED_CORPORA}


__all__ = [
    "DUA_GATED_CORPORA",
    "DUACorpusStub",
    "DUACredentialRequired",
    "all_dua_stubs",
    "dua_stub_for",
    "load_dua_corpus",
]
