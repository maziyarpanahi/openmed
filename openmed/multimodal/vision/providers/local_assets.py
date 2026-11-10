"""Content-addressed local assets for the bounded SmolVLM reference adapter."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .base import VLMContractError

_REQUIRED = frozenset(
    {
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.json",
        "special_tokens_map.json",
        "model.safetensors",
    }
)
_OPTIONAL = frozenset(
    {"added_tokens.json", "merges.txt", "vocab.json", "model.safetensors.index.json"}
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True, slots=True)
class LocalVLMAssets:
    """Verified assets, with caller-supplied revision and license provenance.

    Args:
        directory: Local, owner-controlled model directory; never a Hub ID or URL.
        revision: Immutable source commit, or a SHA-256 for a local conversion.
        license_metadata: Local model card/license snapshot, hashed without interpreting
            terms. The caller is responsible for permission to use the weights.

    Only a single safetensors checkpoint and declarative processor files are supported.
    Extra executable/configuration files are rejected. The directory must remain
    immutable during loading; this is not a sandbox for hostile concurrent writers.
    """

    directory: Path = field(repr=False)
    revision: str
    license_metadata: bytes = field(repr=False)
    file_digests: tuple[tuple[str, str], ...] = field(init=False)
    model_digest: str = field(init=False)
    preprocessing_digest: str = field(init=False)
    license_digest: str = field(init=False)

    def __post_init__(self) -> None:
        failed = False
        try:
            if (
                not isinstance(self.directory, Path)
                or not self.directory.is_dir()
                or type(self.revision) is not str
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.revision) is None
                or type(self.license_metadata) is not bytes
                or not 0 < len(self.license_metadata) <= 128 * 1024
            ):
                raise ValueError
            directory = self.directory.resolve(strict=True)
            names = {p.name for p in directory.iterdir()}
            allowed = (
                _REQUIRED
                | _OPTIONAL
                | {
                    "README.md",
                    "LICENSE",
                    "LICENSE.txt",
                    "hub-metadata.json",
                    ".gitattributes",
                }
            )
            if not _REQUIRED <= names or names - allowed:
                raise ValueError
            digests = []
            for name in sorted(names & (_REQUIRED | _OPTIONAL)):
                path = directory / name
                # Materialize Hub snapshots before use; do not follow asset symlinks.
                if path.is_symlink() or not path.is_file():
                    raise ValueError
                maximum = 8 * 1024**3 if name.endswith(".safetensors") else 16 * 1024**2
                if not 0 < path.stat().st_size <= maximum:
                    raise ValueError
                digest = hashlib.sha256()
                count = 0
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        count += len(chunk)
                        if count > maximum:
                            raise ValueError
                        digest.update(chunk)
                digests.append((name, digest.hexdigest()))
                if name.endswith(".json"):
                    config = json.loads(path.read_bytes())
                    if type(config) is not dict or "auto_map" in config:
                        raise ValueError
            config = json.loads((directory / "config.json").read_bytes())
            if config.get("model_type") != "smolvlm":
                raise ValueError
            # Version one qualifies the unquantized reference; converted weights
            # need their own profile and measured quality/privacy evidence.
            if config.get("quantization") or config.get("quantization_config"):
                raise ValueError
            object.__setattr__(self, "directory", directory)
            object.__setattr__(self, "file_digests", tuple(digests))
            license_digest = hashlib.sha256(self.license_metadata).hexdigest()
            object.__setattr__(self, "license_digest", license_digest)
            object.__setattr__(
                self,
                "model_digest",
                hashlib.sha256(
                    _canonical(
                        [
                            "openmed.smolvlm.assets.v1",
                            self.revision,
                            license_digest,
                            digests,
                        ]
                    )
                ).hexdigest(),
            )
            preprocessing = [
                (n, h) for n, h in digests if not n.endswith(".safetensors")
            ]
            object.__setattr__(
                self,
                "preprocessing_digest",
                hashlib.sha256(
                    _canonical(
                        [
                            "openmed.smolvlm.preprocess.v1",
                            "opaque-rgb-no-exif",
                            preprocessing,
                        ]
                    )
                ).hexdigest(),
            )
        except Exception:
            failed = True
        if failed:
            raise VLMContractError("local VLM assets rejected")

    def verify_current(self) -> None:
        """Rehash assets before loading; reject a changed checkpoint or processor."""
        current = LocalVLMAssets(self.directory, self.revision, self.license_metadata)
        if current.file_digests != self.file_digests:
            raise VLMContractError("local VLM assets changed")

    def to_json(self) -> str:
        """Return digest-only provenance; no local paths or license text."""
        return _canonical(
            {
                "schema_version": "openmed.smolvlm.assets.v1",
                "revision": self.revision,
                "model_digest": self.model_digest,
                "preprocessing_digest": self.preprocessing_digest,
                "license_digest": self.license_digest,
                "files": dict(self.file_digests),
                "quantization": "none",
            }
        ).decode()
