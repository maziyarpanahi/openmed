"""Offline integrity verification for cached OpenMed model artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .offline import env_flag_enabled

logger = logging.getLogger(__name__)

ARTIFACT_MANIFEST_FILENAME = ".openmed-integrity.json"
ARTIFACT_MANIFEST_SCHEMA = "openmed.model_integrity.v1"
INTEGRITY_CACHE_DIRNAME = "integrity"
SKIP_MODEL_VERIFY_ENV_VAR = "OPENMED_SKIP_MODEL_VERIFY"
STRICT_MODEL_VERIFY_ENV_VAR = "OPENMED_MODEL_VERIFY_STRICT"
SIGSTORE_BUNDLE_SUFFIX = ".sigstore.json"
SIGSTORE_MANIFEST_IDENTITY = (
    "https://github.com/maziyarpanahi/openmed/"
    ".github/workflows/manifest-refresh.yml@refs/heads/master"
)
# SHA-256 of the DER SubjectPublicKeyInfo pinned for model-manifest bundles.
# Signing infrastructure is deliberately outside OM-713; bundles are optional.
SIGSTORE_MANIFEST_KEY_SHA256 = (
    "sha256:86eb2319be26d7cabe775ce344cdbc0edcf63fa3ed038b318472755bd19ffed1"
)

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}$")
_HASH_CHUNK_SIZE = 8 * 1024 * 1024
_RUNTIME_METADATA_FILES = frozenset(
    {
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "spm.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)


class ModelIntegrityError(RuntimeError):
    """Raised when model or model-manifest integrity cannot be established."""

    def __init__(
        self,
        model_id: str,
        *,
        expected_sha256: str,
        actual_sha256: str,
        artifact: str | Path | None = None,
        reason: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256
        self.artifact = str(artifact) if artifact is not None else None
        detail = reason or "artifact digest mismatch"
        location = f" ({self.artifact})" if self.artifact else ""
        super().__init__(
            f"Model integrity verification failed for {model_id}{location}: {detail}; "
            f"expected {expected_sha256}, actual {actual_sha256}. Remove the cached "
            f"artifact and run `hf download {model_id} --cache-dir "
            "~/.cache/openmed --force-download`, then retry. "
            "Report suspected tampering at "
            "https://openmed.life/security/disclosure-policy/."
        )


@dataclass(frozen=True)
class ModelVerificationResult:
    """Successful verification result for one cached artifact set."""

    model_id: str
    artifact_root: Path
    expected_sha256: str
    actual_sha256: str
    files_checked: int
    manifest_path: Path


def model_verification_skipped() -> bool:
    """Return whether model verification was explicitly disabled."""
    return env_flag_enabled(os.environ.get(SKIP_MODEL_VERIFY_ENV_VAR))


def strict_model_verification() -> bool:
    """Return whether missing registry integrity metadata must fail closed."""
    return env_flag_enabled(os.environ.get(STRICT_MODEL_VERIFY_ENV_VAR))


def sha256_file(path: str | Path) -> str:
    """Stream *path* and return its ``sha256:`` digest."""
    digest = hashlib.sha256()
    buffer = bytearray(_HASH_CHUNK_SIZE)
    view = memoryview(buffer)
    with Path(path).open("rb") as handle:
        while size := handle.readinto(buffer):
            digest.update(view[:size])
    return f"sha256:{digest.hexdigest()}"


def prepare_model_reference(
    model_reference: str,
    *,
    registry_info: Any | None,
    cache_dir: str | Path,
    local_only: bool,
    token: str | None = None,
) -> str:
    """Return a verified local snapshot path or the unchanged model reference.

    Local directories are checked against an adjacent integrity manifest. Registry
    models use a cache-side integrity manifest after the first verified download.
    A network call is only made when a registry model has not yet been prepared and
    local-only mode is disabled.
    """
    if model_verification_skipped():
        _warn_verification_skipped(model_reference)
        return model_reference

    local_path = _existing_directory(model_reference)
    if local_path is not None:
        sidecar = local_path / ARTIFACT_MANIFEST_FILENAME
        if sidecar.exists():
            verify_artifact_manifest(sidecar)
        else:
            logger.warning(
                "MODEL INTEGRITY NOT VERIFIED for local path %s: %s is absent",
                local_path,
                ARTIFACT_MANIFEST_FILENAME,
            )
        return str(local_path)

    if registry_info is None:
        return model_reference

    model_id = str(registry_info.model_id)
    reproducibility_hash = registry_info.reproducibility_hash
    if not isinstance(reproducibility_hash, str) or not _SHA256_RE.fullmatch(
        reproducibility_hash
    ):
        return _handle_missing_registry_hash(model_id, model_reference)

    sidecar = _registry_sidecar_path(cache_dir, model_id, reproducibility_hash)
    if sidecar.exists():
        result = verify_artifact_manifest(
            sidecar,
            expected_model_id=model_id,
            expected_reproducibility_hash=reproducibility_hash,
        )
        return str(result.artifact_root)

    if local_only:
        return _handle_missing_integrity_sidecar(model_id, model_reference, sidecar)

    return _download_verified_snapshot(
        registry_info,
        cache_dir=Path(cache_dir),
        sidecar=sidecar,
        token=token,
    )


def verify_artifact_manifest(
    manifest_path: str | Path,
    *,
    expected_model_id: str | None = None,
    expected_reproducibility_hash: str | None = None,
) -> ModelVerificationResult:
    """Verify every file named by an offline model integrity manifest."""
    sidecar = Path(manifest_path)
    payload = _load_artifact_manifest(sidecar)
    model_id = str(payload["model_id"])
    if expected_model_id is not None and model_id != expected_model_id:
        raise ModelIntegrityError(
            expected_model_id,
            expected_sha256="sha256:" + "0" * 64,
            actual_sha256="sha256:" + "0" * 64,
            artifact=sidecar,
            reason=f"integrity manifest belongs to {model_id}",
        )
    recorded_reproducibility_hash = str(payload["reproducibility_hash"])
    if (
        expected_reproducibility_hash is not None
        and recorded_reproducibility_hash != expected_reproducibility_hash
    ):
        raise ModelIntegrityError(
            model_id,
            expected_sha256=expected_reproducibility_hash,
            actual_sha256=recorded_reproducibility_hash,
            artifact=sidecar,
            reason="integrity manifest does not match the registry revision",
        )

    artifact_root = _artifact_root(sidecar, payload)
    expected_records: list[tuple[str, str]] = []
    actual_records: list[tuple[str, str]] = []
    for entry in payload["artifacts"]:
        relative_path = _safe_relative_artifact_path(entry["path"])
        expected = str(entry["sha256"])
        artifact = artifact_root / relative_path
        if not artifact.is_file():
            raise ModelIntegrityError(
                model_id,
                expected_sha256=expected,
                actual_sha256="sha256:" + "0" * 64,
                artifact=relative_path,
                reason="cached artifact is missing",
            )
        actual = sha256_file(artifact)
        if actual != expected:
            raise ModelIntegrityError(
                model_id,
                expected_sha256=expected,
                actual_sha256=actual,
                artifact=relative_path,
            )
        expected_records.append((relative_path.as_posix(), expected))
        actual_records.append((relative_path.as_posix(), actual))

    expected_set_hash = _artifact_set_digest(expected_records)
    actual_set_hash = _artifact_set_digest(actual_records)
    return ModelVerificationResult(
        model_id=model_id,
        artifact_root=artifact_root,
        expected_sha256=expected_set_hash,
        actual_sha256=actual_set_hash,
        files_checked=len(expected_records),
        manifest_path=sidecar,
    )


def verify_cached_models(
    *,
    cache_dir: str | Path,
    model_id: str | None = None,
) -> list[ModelVerificationResult]:
    """Verify cached artifact sets without performing network access."""
    manifests = list(_iter_cached_integrity_manifests(cache_dir))
    if model_id is not None:
        local_path = _existing_directory(model_id)
        if local_path is not None:
            sidecar = local_path / ARTIFACT_MANIFEST_FILENAME
            if not sidecar.exists():
                raise ModelIntegrityError(
                    model_id,
                    expected_sha256="sha256:<required>",
                    actual_sha256="sha256:<missing>",
                    artifact=sidecar,
                    reason="integrity manifest is missing",
                )
            return [verify_artifact_manifest(sidecar)]
        manifests = [path for path in manifests if _manifest_model_id(path) == model_id]
        if not manifests:
            raise ModelIntegrityError(
                model_id,
                expected_sha256="sha256:<required>",
                actual_sha256="sha256:<missing>",
                reason="no verified cached artifact set was found",
            )
    return [verify_artifact_manifest(path) for path in manifests]


def verify_manifest_signature_if_present(manifest_path: str | Path) -> bool:
    """Verify the detached Sigstore bundle beside *manifest_path*, when present."""
    path = Path(manifest_path)
    bundle = Path(f"{path}{SIGSTORE_BUNDLE_SUFFIX}")
    if not bundle.exists():
        return False
    if model_verification_skipped():
        _warn_verification_skipped(str(path))
        return False
    verify_manifest_signature(path, bundle)
    return True


def verify_manifest_signature(
    manifest_path: str | Path,
    bundle_path: str | Path,
    *,
    expected_identity: str = SIGSTORE_MANIFEST_IDENTITY,
    expected_key_sha256: str = SIGSTORE_MANIFEST_KEY_SHA256,
) -> str:
    """Verify a detached Sigstore bundle fully offline against pinned identity."""
    manifest = Path(manifest_path)
    bundle = Path(bundle_path)
    manifest_bytes = manifest.read_bytes()
    actual_digest = f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}"
    try:
        payload = json.loads(bundle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelIntegrityError(
            "models.jsonl",
            expected_sha256="sha256:<valid Sigstore bundle>",
            actual_sha256=actual_digest,
            artifact=bundle,
            reason="detached signature bundle is unreadable",
        ) from exc
    try:
        message_signature = payload["messageSignature"]
        digest_info = message_signature["messageDigest"]
        algorithm = digest_info["algorithm"]
        expected_digest = (
            "sha256:" + base64.b64decode(digest_info["digest"], validate=True).hex()
        )
        signature = base64.b64decode(message_signature["signature"], validate=True)
        raw_certificate = _bundle_certificate_bytes(payload)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise ModelIntegrityError(
            "models.jsonl",
            expected_sha256="sha256:<valid Sigstore bundle>",
            actual_sha256=actual_digest,
            artifact=bundle,
            reason="detached signature bundle is malformed",
        ) from exc

    if algorithm not in {"SHA2_256", "SHA_256", "SHA256"}:
        raise ModelIntegrityError(
            "models.jsonl",
            expected_sha256="sha256:<SHA2_256 bundle>",
            actual_sha256=actual_digest,
            artifact=bundle,
            reason=f"unsupported bundle digest algorithm {algorithm}",
        )
    if expected_digest != actual_digest:
        raise ModelIntegrityError(
            "models.jsonl",
            expected_sha256=expected_digest,
            actual_sha256=actual_digest,
            artifact=manifest,
            reason="signed manifest digest mismatch",
        )

    try:
        from cryptography import x509
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
        from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise ModelIntegrityError(
            "models.jsonl",
            expected_sha256=expected_digest,
            actual_sha256=actual_digest,
            artifact=bundle,
            reason="install openmed[integrity] to verify the local Sigstore bundle",
        ) from exc

    try:
        certificate = x509.load_der_x509_certificate(raw_certificate)
    except ValueError as exc:
        raise ModelIntegrityError(
            "models.jsonl",
            expected_sha256=expected_digest,
            actual_sha256=actual_digest,
            artifact=bundle,
            reason="signing certificate is invalid",
        ) from exc
    public_key = certificate.public_key()
    public_key_der = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    actual_key_hash = f"sha256:{hashlib.sha256(public_key_der).hexdigest()}"
    if actual_key_hash != expected_key_sha256:
        raise ModelIntegrityError(
            "models.jsonl",
            expected_sha256=expected_key_sha256,
            actual_sha256=actual_key_hash,
            artifact=bundle,
            reason="signing key is not the pinned model-manifest key",
        )

    try:
        san = certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        identities = set(san.get_values_for_type(x509.UniformResourceIdentifier))
        identities.update(san.get_values_for_type(x509.RFC822Name))
    except x509.ExtensionNotFound as exc:
        raise ModelIntegrityError(
            "models.jsonl",
            expected_sha256=expected_digest,
            actual_sha256=actual_digest,
            artifact=bundle,
            reason="signing certificate has no identity",
        ) from exc
    if expected_identity not in identities:
        raise ModelIntegrityError(
            "models.jsonl",
            expected_sha256=expected_digest,
            actual_sha256=actual_digest,
            artifact=bundle,
            reason=f"signer identity is not pinned to {expected_identity}",
        )

    try:
        if isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(signature, manifest_bytes, ec.ECDSA(hashes.SHA256()))
        elif isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(signature, manifest_bytes, PKCS1v15(), hashes.SHA256())
        elif isinstance(public_key, ed25519.Ed25519PublicKey):
            public_key.verify(signature, manifest_bytes)
        else:  # pragma: no cover - defensive against future key types
            raise TypeError(f"unsupported public key type: {type(public_key).__name__}")
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise ModelIntegrityError(
            "models.jsonl",
            expected_sha256=expected_digest,
            actual_sha256=actual_digest,
            artifact=bundle,
            reason="detached signature is invalid",
        ) from exc
    return actual_digest


def _download_verified_snapshot(
    registry_info: Any,
    *,
    cache_dir: Path,
    sidecar: Path,
    token: str | None,
) -> str:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - guarded by model extras
        raise ModelIntegrityError(
            str(registry_info.model_id),
            expected_sha256=str(registry_info.reproducibility_hash),
            actual_sha256="sha256:<unavailable>",
            reason="huggingface-hub is required for a verified model download",
        ) from exc

    model_id = str(registry_info.model_id)
    try:
        remote_info = HfApi(token=token).model_info(
            model_id,
            files_metadata=True,
            token=token,
        )
        remote_reproducibility_hash = _remote_reproducibility_hash(remote_info)
        if remote_reproducibility_hash != registry_info.reproducibility_hash:
            raise ModelIntegrityError(
                model_id,
                expected_sha256=str(registry_info.reproducibility_hash),
                actual_sha256=remote_reproducibility_hash,
                reason="registry metadata does not match the current repository revision",
            )

        siblings = _select_runtime_siblings(remote_info.siblings or ())
        if not any(_runtime_weight_file(str(item.rfilename)) for item in siblings):
            return _handle_missing_integrity_sidecar(model_id, model_id, sidecar)
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=remote_info.sha,
                cache_dir=str(cache_dir),
                allow_patterns=[str(item.rfilename) for item in siblings],
                token=token,
            )
        )
        records = [_verified_remote_artifact(snapshot, item) for item in siblings]
        _write_artifact_manifest(
            sidecar,
            model_id=model_id,
            reproducibility_hash=str(registry_info.reproducibility_hash),
            artifact_root=snapshot,
            artifacts=records,
        )
        verify_artifact_manifest(
            sidecar,
            expected_model_id=model_id,
            expected_reproducibility_hash=str(registry_info.reproducibility_hash),
        )
        return str(snapshot)
    except ModelIntegrityError:
        raise
    except Exception as exc:
        raise ModelIntegrityError(
            model_id,
            expected_sha256=str(registry_info.reproducibility_hash),
            actual_sha256="sha256:<unverified>",
            reason=f"verified download could not be completed: {exc}",
        ) from exc


def _remote_reproducibility_hash(remote_info: Any) -> str:
    timestamp = remote_info.last_modified or remote_info.created_at
    released = timestamp.date().isoformat() if timestamp is not None else None
    siblings = sorted(str(item.rfilename) for item in remote_info.siblings or ())
    payload = json.dumps(
        {
            "repo_id": str(remote_info.id),
            "sha": remote_info.sha,
            "released": released,
            "siblings": siblings,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _select_runtime_siblings(siblings: Sequence[Any]) -> list[Any]:
    items = [item for item in siblings if _runtime_metadata_file(item.rfilename)]
    safetensors = [
        item
        for item in siblings
        if Path(str(item.rfilename)).name.endswith(".safetensors")
    ]
    if safetensors:
        items.extend(safetensors)
        items.extend(
            item
            for item in siblings
            if Path(str(item.rfilename)).name == "model.safetensors.index.json"
        )
    else:
        items.extend(
            item for item in siblings if _pytorch_weight_file(str(item.rfilename))
        )
        items.extend(
            item
            for item in siblings
            if Path(str(item.rfilename)).name == "pytorch_model.bin.index.json"
        )
    by_name = {str(item.rfilename): item for item in items}
    return [by_name[name] for name in sorted(by_name)]


def _runtime_metadata_file(filename: str) -> bool:
    name = Path(str(filename)).name
    return name in _RUNTIME_METADATA_FILES or name.startswith("chat_template.")


def _pytorch_weight_file(filename: str) -> bool:
    name = Path(filename).name
    return name == "pytorch_model.bin" or (
        name.startswith("pytorch_model-") and name.endswith(".bin")
    )


def _runtime_weight_file(filename: str) -> bool:
    name = Path(filename).name
    return name.endswith(".safetensors") or _pytorch_weight_file(name)


def _verified_remote_artifact(snapshot: Path, sibling: Any) -> dict[str, Any]:
    relative_path = _safe_relative_artifact_path(str(sibling.rfilename))
    artifact = snapshot / relative_path
    if not artifact.is_file():
        raise ModelIntegrityError(
            str(snapshot),
            expected_sha256="sha256:<downloaded artifact>",
            actual_sha256="sha256:<missing>",
            artifact=relative_path,
            reason="snapshot download omitted a runtime artifact",
        )

    actual_sha256, actual_git_sha1 = _hash_file(artifact)
    lfs = getattr(sibling, "lfs", None)
    if lfs is not None and getattr(lfs, "sha256", None):
        expected_sha256 = f"sha256:{lfs.sha256}"
        if actual_sha256 != expected_sha256:
            raise ModelIntegrityError(
                str(snapshot),
                expected_sha256=expected_sha256,
                actual_sha256=actual_sha256,
                artifact=relative_path,
            )
    else:
        expected_git_sha1 = getattr(sibling, "blob_id", None)
        if not expected_git_sha1 or actual_git_sha1 != expected_git_sha1:
            raise ModelIntegrityError(
                str(snapshot),
                expected_sha256="sha256:<content addressed Git blob>",
                actual_sha256=actual_sha256,
                artifact=relative_path,
                reason="Git blob identity mismatch",
            )
        expected_sha256 = actual_sha256
    return {
        "path": relative_path.as_posix(),
        "sha256": expected_sha256,
        "size": artifact.stat().st_size,
    }


def _hash_file(path: Path) -> tuple[str, str]:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(usedforsecurity=False)
    git_sha1.update(f"blob {size}\0".encode("ascii"))
    buffer = bytearray(_HASH_CHUNK_SIZE)
    view = memoryview(buffer)
    with path.open("rb") as handle:
        while count := handle.readinto(buffer):
            chunk = view[:count]
            sha256.update(chunk)
            git_sha1.update(chunk)
    return f"sha256:{sha256.hexdigest()}", git_sha1.hexdigest()


def _write_artifact_manifest(
    sidecar: Path,
    *,
    model_id: str,
    reproducibility_hash: str,
    artifact_root: Path,
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "model_id": model_id,
        "reproducibility_hash": reproducibility_hash,
        "artifact_root": str(artifact_root.resolve()),
        "artifacts": list(artifacts),
    }
    temporary = sidecar.with_suffix(f"{sidecar.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(sidecar)


def _load_artifact_manifest(sidecar: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("integrity manifest must be an object")
        if payload.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA:
            raise ValueError("unsupported integrity manifest schema")
        model_id = payload.get("model_id")
        reproducibility_hash = payload.get("reproducibility_hash")
        artifacts = payload.get("artifacts")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("integrity manifest model_id is invalid")
        if not isinstance(reproducibility_hash, str) or not _SHA256_RE.fullmatch(
            reproducibility_hash
        ):
            raise ValueError("integrity manifest reproducibility_hash is invalid")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("integrity manifest artifacts are empty")
        for entry in artifacts:
            if not isinstance(entry, Mapping):
                raise ValueError("integrity artifact entry is invalid")
            if not isinstance(entry.get("path"), str):
                raise ValueError("integrity artifact path is invalid")
            digest = entry.get("sha256")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ValueError("integrity artifact sha256 is invalid")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ModelIntegrityError(
            str(sidecar),
            expected_sha256="sha256:<valid integrity manifest>",
            actual_sha256="sha256:<invalid>",
            artifact=sidecar,
            reason=f"integrity manifest is invalid: {exc}",
        ) from exc
    return payload


def _artifact_root(sidecar: Path, payload: Mapping[str, Any]) -> Path:
    value = payload.get("artifact_root", ".")
    if not isinstance(value, str) or not value:
        raise ValueError(f"integrity artifact_root is invalid: {sidecar}")
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = sidecar.parent / root
    return root


def _safe_relative_artifact_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"artifact path must stay within its model root: {value}")
    return path


def _artifact_set_digest(records: Iterable[tuple[str, str]]) -> str:
    payload = json.dumps(
        sorted(records),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _registry_sidecar_path(
    cache_dir: str | Path,
    model_id: str,
    reproducibility_hash: str,
) -> Path:
    model_token = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:24]
    revision_token = reproducibility_hash.removeprefix("sha256:")
    return (
        Path(cache_dir)
        / INTEGRITY_CACHE_DIRNAME
        / model_token
        / f"{revision_token}.json"
    )


def _iter_cached_integrity_manifests(cache_dir: str | Path) -> Iterable[Path]:
    root = Path(cache_dir) / INTEGRITY_CACHE_DIRNAME
    if not root.is_dir():
        return ()
    return sorted(root.glob("*/*.json"))


def _manifest_model_id(sidecar: Path) -> str | None:
    try:
        payload = _load_artifact_manifest(sidecar)
    except (ModelIntegrityError, ValueError):
        return None
    return str(payload["model_id"])


def _existing_directory(value: str) -> Path | None:
    try:
        path = Path(value).expanduser()
        return path if path.is_dir() else None
    except (OSError, TypeError, ValueError):
        return None


def _handle_missing_registry_hash(model_id: str, fallback: str) -> str:
    if strict_model_verification():
        raise ModelIntegrityError(
            model_id,
            expected_sha256="sha256:<required>",
            actual_sha256="sha256:<missing>",
            reason="registry model has no valid reproducibility_hash",
        )
    logger.warning(
        "MODEL INTEGRITY NOT VERIFIED for registry model %s: reproducibility_hash "
        "is absent or malformed",
        model_id,
    )
    return fallback


def _handle_missing_integrity_sidecar(
    model_id: str,
    fallback: str,
    sidecar: Path,
) -> str:
    if strict_model_verification():
        raise ModelIntegrityError(
            model_id,
            expected_sha256="sha256:<required artifact set>",
            actual_sha256="sha256:<missing>",
            artifact=sidecar,
            reason="no verified artifact integrity manifest is cached",
        )
    logger.warning(
        "MODEL INTEGRITY NOT VERIFIED for %s: no trusted artifact hashes are "
        "available at %s",
        model_id,
        sidecar,
    )
    return fallback


def _warn_verification_skipped(model_id: str) -> None:
    logger.warning(
        "MODEL INTEGRITY VERIFICATION DISABLED for %s because %s=1; cached "
        "artifacts may be corrupted or tampered with",
        model_id,
        SKIP_MODEL_VERIFY_ENV_VAR,
    )


def _bundle_certificate_bytes(payload: Mapping[str, Any]) -> bytes:
    verification_material = payload["verificationMaterial"]
    certificate = verification_material.get("certificate")
    if certificate is None:
        chain = verification_material["x509CertificateChain"]["certificates"]
        certificate = chain[0]
    return base64.b64decode(certificate["rawBytes"], validate=True)
