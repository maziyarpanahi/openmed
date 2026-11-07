"""Ephemeral synthetic PKI for TLS and mTLS credential-reload tests.

Every key and certificate is generated in memory when a test runs and is only
written below a pytest temporary directory. Names are synthetic workload names;
no key material, production certificate or PHI is committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from openmed.service.tls_reload import certificate_sha256

SERVER_NAME = "openmed-reload.test"
SYNTHETIC_NAME_PREFIX = "reload-synthetic"


@dataclass(frozen=True)
class SyntheticIdentity:
    """One synthetic key pair and the certificate issued for it."""

    key: ec.EllipticCurvePrivateKey
    certificate: x509.Certificate
    common_name: str

    @property
    def cert_pem(self) -> bytes:
        """Return the certificate as PEM."""
        return self.certificate.public_bytes(serialization.Encoding.PEM)

    @property
    def key_pem(self) -> bytes:
        """Return the unencrypted PKCS#8 private key as PEM."""
        return self.key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    @property
    def sha256(self) -> str:
        """Return the certificate fingerprint."""
        return certificate_sha256(self.certificate)


@dataclass(frozen=True)
class SyntheticGeneration:
    """A root CA plus the server and client identities it issued."""

    label: str
    root: SyntheticIdentity
    server: SyntheticIdentity
    client: SyntheticIdentity


@dataclass(frozen=True)
class CredentialFiles:
    """Fixed paths a provider is configured with; rotation rewrites them."""

    directory: Path
    certificate_chain_file: Path
    private_key_file: Path
    client_ca_file: Path


class FakeClock:
    """Deterministic, manually advanced UTC clock."""

    def __init__(self, start: Optional[datetime] = None) -> None:
        self.now = start or datetime.now(timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self.now += timedelta(seconds=seconds)


def issue(
    common_name: str,
    *,
    issuer: Optional[SyntheticIdentity],
    now: datetime,
    ca: bool = False,
    usage: Optional[x509.ObjectIdentifier] = None,
    not_before: Optional[datetime] = None,
    not_after: Optional[datetime] = None,
    key: Optional[ec.EllipticCurvePrivateKey] = None,
) -> SyntheticIdentity:
    """Issue a synthetic certificate, self-signed when ``issuer`` is ``None``."""
    key = key or ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OpenMed Test"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )
    signer_key = key if issuer is None else issuer.key
    issuer_name = subject if issuer is None else issuer.certificate.subject
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before or now - timedelta(days=1))
        .not_valid_after(not_after or now + timedelta(days=30))
        .add_extension(
            x509.BasicConstraints(ca=ca, path_length=0 if ca else None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=not ca,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=ca,
                crl_sign=ca,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(signer_key.public_key()),
            critical=False,
        )
    )
    if usage is not None:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([usage]), critical=False
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName(common_name)]),
            critical=False,
        )
    certificate = builder.sign(signer_key, hashes.SHA256())
    return SyntheticIdentity(key=key, certificate=certificate, common_name=common_name)


def build_generation(
    label: str,
    *,
    now: datetime,
    server_not_before: Optional[datetime] = None,
    server_not_after: Optional[datetime] = None,
    root_not_after: Optional[datetime] = None,
) -> SyntheticGeneration:
    """Build a root CA with one server and one client identity."""
    root = issue(
        f"{SYNTHETIC_NAME_PREFIX}-{label}-root",
        issuer=None,
        now=now,
        ca=True,
        not_after=root_not_after,
    )
    server = issue(
        SERVER_NAME,
        issuer=root,
        now=now,
        usage=ExtendedKeyUsageOID.SERVER_AUTH,
        not_before=server_not_before or now - timedelta(hours=1),
        not_after=server_not_after or now + timedelta(days=7),
    )
    client = issue(
        f"{SYNTHETIC_NAME_PREFIX}-{label}-client",
        issuer=root,
        now=now,
        usage=ExtendedKeyUsageOID.CLIENT_AUTH,
        not_before=now - timedelta(hours=1),
        not_after=now + timedelta(days=7),
    )
    return SyntheticGeneration(label=label, root=root, server=server, client=client)


def credential_files(directory: Path) -> CredentialFiles:
    """Return the fixed credential paths below ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    return CredentialFiles(
        directory=directory,
        certificate_chain_file=directory / "server-chain.pem",
        private_key_file=directory / "server-key.pem",
        client_ca_file=directory / "client-ca.pem",
    )


def write_atomic(path: Path, data: bytes) -> None:
    """Replace ``path`` atomically, like a secret-volume update."""
    staging = path.with_name(f".{path.name}.staging")
    staging.write_bytes(data)
    os.replace(staging, path)


def install(
    files: CredentialFiles,
    generation: SyntheticGeneration,
    *,
    client_ca_pem: Optional[bytes] = None,
) -> None:
    """Write a complete generation to the configured paths."""
    write_atomic(files.certificate_chain_file, generation.server.cert_pem)
    write_atomic(files.private_key_file, generation.server.key_pem)
    write_atomic(
        files.client_ca_file,
        generation.root.cert_pem if client_ca_pem is None else client_ca_pem,
    )


def assert_value_free(
    text: str, files: CredentialFiles, *generations: SyntheticGeneration
) -> None:
    """Assert that rendered output carries no path, PEM, key or identity."""
    assert str(files.directory) not in text
    assert ".pem" not in text
    assert "BEGIN" not in text
    assert SYNTHETIC_NAME_PREFIX not in text
    assert SERVER_NAME not in text
    for generation in generations:
        for identity in (generation.root, generation.server, generation.client):
            assert identity.key_pem.decode("ascii") not in text
            assert identity.certificate.subject.rfc4514_string() not in text


__all__ = [
    "SERVER_NAME",
    "SYNTHETIC_NAME_PREFIX",
    "CredentialFiles",
    "FakeClock",
    "SyntheticGeneration",
    "SyntheticIdentity",
    "assert_value_free",
    "build_generation",
    "credential_files",
    "install",
    "issue",
    "write_atomic",
]
