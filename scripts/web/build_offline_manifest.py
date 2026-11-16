#!/usr/bin/env python3
"""Build the versioned digest manifest for an installable browser profile.

The installable privacy playground precaches only the static files listed in
its ``offline-assets.json``. This tool recomputes the SHA-256 digest and byte
length of every listed file, derives the generation identifier from those
digests and the service-worker source, and embeds the same asset table in the
service worker. The service worker refuses to activate unless the deployed
manifest, every asset and its own embedded table agree, so one generation can
never be assembled from files of another.

The tool is dependency-free and performs no network operation. Messages name
only the public relative asset URL and a fixed reason; they never include
absolute filesystem paths or file contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_ROOT = ROOT / "docs" / "demo" / "privacy-playground"
DEFAULT_SITE_ROOT = ROOT / "docs"
MANIFEST_NAME = "offline-assets.json"
SCHEMA = "openmed.offline-assets.v1"
BLOCK_BEGIN = (
    "// BEGIN GENERATED OFFLINE PROFILE (python scripts/web/build_offline_manifest.py)"
)
BLOCK_END = "// END GENERATED OFFLINE PROFILE"
BLANK_BLOCK_BODY = "const OFFLINE_PROFILE = null;"

ASSET_KINDS = ("shell", "runtime", "model")
CONTENT_TYPES: Mapping[str, str] = {
    ".bin": "application/octet-stream",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".mjs": "text/javascript; charset=utf-8",
    ".onnx": "application/octet-stream",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".wasm": "application/wasm",
    ".webmanifest": "application/manifest+json",
    ".woff2": "font/woff2",
}
KIND_EXTENSIONS: Mapping[str, frozenset[str]] = {
    "shell": frozenset(
        {".css", ".html", ".js", ".json", ".mjs", ".png", ".svg"}
        | {".webmanifest", ".woff2"}
    ),
    "runtime": frozenset({".js", ".mjs", ".wasm"}),
    "model": frozenset({".bin", ".json", ".onnx", ".txt"}),
}
MAX_ASSETS = 512
MAX_ASSET_BYTES = 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MAX_URL_LENGTH = 512
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z", re.ASCII)
_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


class OfflineManifestError(ValueError):
    """Raised when an offline profile declaration cannot be built safely."""


@dataclass(frozen=True)
class OfflineAsset:
    """One precached static file in an offline generation.

    Attributes:
        url: Public URL relative to the profile directory.
        kind: ``shell``, ``runtime`` or ``model``.
        content_type: Media type the service worker serves the bytes with.
        size: Exact byte length.
        sha256: Lower-case hexadecimal SHA-256 digest of the bytes.
    """

    url: str
    kind: str
    content_type: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest representation of the asset."""

        return {
            "url": self.url,
            "kind": self.kind,
            "content_type": self.content_type,
            "bytes": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class OfflineProfile:
    """A complete, digest-pinned offline generation.

    Attributes:
        profile: Stable profile name used to namespace browser caches.
        start_url: Shell URL served for offline navigations.
        service_worker: Service-worker file name relative to the profile.
        service_worker_sha256: Digest of the service worker with its
            generated block blanked.
        generation: Digest that identifies this exact set of files.
        assets: Ordered asset table.
    """

    profile: str
    start_url: str
    service_worker: str
    service_worker_sha256: str
    generation: str
    assets: tuple[OfflineAsset, ...]

    @property
    def total_bytes(self) -> int:
        """Return the summed byte length of every asset."""

        return sum(asset.size for asset in self.assets)

    def manifest_dict(self) -> dict[str, Any]:
        """Return the published ``offline-assets.json`` document."""

        return {
            "schema": SCHEMA,
            "profile": self.profile,
            "start_url": self.start_url,
            "service_worker": self.service_worker,
            "service_worker_sha256": self.service_worker_sha256,
            "generation": self.generation,
            "asset_count": len(self.assets),
            "total_bytes": self.total_bytes,
            "assets": [asset.to_dict() for asset in self.assets],
        }


def sha256_hex(payload: bytes) -> str:
    """Return the lower-case hexadecimal SHA-256 digest of ``payload``."""

    return hashlib.sha256(payload).hexdigest()


def compute_generation(
    *,
    profile: str,
    start_url: str,
    service_worker_sha256: str,
    assets: Sequence[OfflineAsset],
) -> str:
    """Derive the generation identifier shared by the manifest and worker.

    The service worker recomputes the same value from its embedded table, so
    the line format is part of the ``openmed.offline-assets.v1`` contract.

    Args:
        profile: Profile name.
        start_url: Offline navigation URL.
        service_worker_sha256: Digest of the blanked service worker.
        assets: Ordered asset table.

    Returns:
        Lower-case hexadecimal SHA-256 generation identifier.
    """

    lines = [
        SCHEMA,
        f"profile\t{profile}",
        f"start_url\t{start_url}",
        f"service_worker_sha256\t{service_worker_sha256}",
    ]
    lines.extend(
        "\t".join(
            (
                "asset",
                asset.url,
                asset.kind,
                asset.content_type,
                str(asset.size),
                asset.sha256,
            )
        )
        for asset in assets
    )
    return sha256_hex(("\n".join(lines) + "\n").encode("utf-8"))


def validate_asset_url(url: object) -> str:
    """Validate one relative asset URL and return it.

    Accepted URLs are ``./`` (the profile directory), or an optional run of
    leading ``../`` segments followed by plain path segments. Queries,
    fragments, encoded characters, schemes, absolute paths, backslashes and
    inner dot segments are rejected so the browser and this tool always
    resolve the URL to the same file.

    Args:
        url: Candidate URL from the profile declaration.

    Returns:
        The validated URL.

    Raises:
        OfflineManifestError: If the URL is not a safe static asset URL.
    """

    if not isinstance(url, str) or not url or len(url) > MAX_URL_LENGTH:
        raise OfflineManifestError("asset url must be a short non-empty string")
    if url == "./":
        return url
    if url.startswith("/") or any(character in url for character in "?#\\%:|\t"):
        raise OfflineManifestError("asset url must be a plain relative path")
    parts = url.split("/")
    while parts and parts[0] == "..":
        parts.pop(0)
    if not parts:
        raise OfflineManifestError("asset url must name a file")
    directory = parts[-1] == ""
    segments = parts[:-1] if directory else parts
    if not segments or not all(_SEGMENT.fullmatch(part) for part in segments):
        raise OfflineManifestError("asset url contains an unsafe path segment")
    return url


def _asset_file_name(url: str) -> str:
    return "index.html" if url == "./" or url.endswith("/") else ""


def resolve_asset_path(profile_root: Path, site_root: Path, url: str) -> Path:
    """Map an asset URL to the file it is served from.

    Args:
        profile_root: Directory that holds the service worker and manifest.
        site_root: Directory that bounds every precached file.
        url: Validated relative asset URL.

    Returns:
        The regular file inside ``site_root`` that the URL resolves to.

    Raises:
        OfflineManifestError: If the file is missing, a symlink or outside the
            site root.
    """

    relative = url if url != "./" else ""
    candidate = Path(os.path.abspath(profile_root / relative / _asset_file_name(url)))
    site = Path(os.path.abspath(site_root))
    if not candidate.is_relative_to(site):
        raise OfflineManifestError(f"asset {url!r} resolves outside the site root")
    current = candidate
    while current != site:
        if current.is_symlink():
            raise OfflineManifestError(f"asset {url!r} must not use a symlink")
        current = current.parent
    if not candidate.is_file():
        raise OfflineManifestError(f"asset {url!r} is missing")
    return candidate


def content_type_for(url: str, kind: str) -> str:
    """Return the fixed media type for an asset URL.

    Args:
        url: Validated relative asset URL.
        kind: Declared asset kind.

    Returns:
        The media type served for the asset.

    Raises:
        OfflineManifestError: If the extension is not allowed for the kind.
    """

    name = _asset_file_name(url) or url.rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lower()
    if suffix not in KIND_EXTENSIONS[kind]:
        raise OfflineManifestError(f"asset {url!r} has an extension not allowed")
    return CONTENT_TYPES[suffix]


def blank_service_worker(source: str) -> str:
    """Return the service-worker source with its generated block blanked.

    Args:
        source: Current service-worker source text.

    Returns:
        Source text whose generated block contains only the blank body.

    Raises:
        OfflineManifestError: If the generated block markers are missing.
    """

    return _replace_block(source, BLANK_BLOCK_BODY)


def _replace_block(source: str, body: str) -> str:
    begin = source.find(BLOCK_BEGIN)
    end = source.find(BLOCK_END)
    if (
        begin < 0
        or end < begin
        or source.count(BLOCK_BEGIN) != 1
        or source.count(BLOCK_END) != 1
    ):
        raise OfflineManifestError("service worker generated block is missing")
    head = source[: begin + len(BLOCK_BEGIN)]
    tail = source[end:]
    return f"{head}\n{body}\n{tail}"


def render_service_worker(source: str, profile: OfflineProfile, manifest: str) -> str:
    """Embed the verified asset table into the service-worker source.

    Args:
        source: Current service-worker source text.
        profile: Built offline profile.
        manifest: Rendered ``offline-assets.json`` text.

    Returns:
        Service-worker source with the generated block filled in.
    """

    embedded = {
        "schema": SCHEMA,
        "profile": profile.profile,
        "start_url": profile.start_url,
        "service_worker_sha256": profile.service_worker_sha256,
        "generation": profile.generation,
        "manifest_sha256": sha256_hex(manifest.encode("utf-8")),
        "total_bytes": profile.total_bytes,
        "assets": [asset.to_dict() for asset in profile.assets],
    }
    body = "const OFFLINE_PROFILE = " + json.dumps(embedded, indent=2) + ";"
    return _replace_block(source, body)


def render_manifest(profile: OfflineProfile) -> str:
    """Return the canonical ``offline-assets.json`` text for a profile."""

    return json.dumps(profile.manifest_dict(), indent=2) + "\n"


def _declarations(document: object) -> tuple[str, str, str, list[tuple[str, str]]]:
    if not isinstance(document, dict):
        raise OfflineManifestError("offline manifest must be a JSON object")
    if document.get("schema") != SCHEMA:
        raise OfflineManifestError("offline manifest schema is not supported")
    profile = document.get("profile")
    if not isinstance(profile, str) or not _PROFILE_NAME.fullmatch(profile):
        raise OfflineManifestError("offline profile name is invalid")
    start_url = validate_asset_url(document.get("start_url"))
    service_worker = document.get("service_worker")
    if (
        not isinstance(service_worker, str)
        or not _SEGMENT.fullmatch(service_worker)
        or not service_worker.endswith(".js")
    ):
        raise OfflineManifestError("service worker must be a file in the profile")
    raw_assets = document.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise OfflineManifestError("offline manifest must list assets")
    if len(raw_assets) > MAX_ASSETS:
        raise OfflineManifestError("offline manifest lists too many assets")
    assets: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in raw_assets:
        if not isinstance(entry, dict):
            raise OfflineManifestError("asset entries must be JSON objects")
        url = validate_asset_url(entry.get("url"))
        kind = entry.get("kind")
        if kind not in ASSET_KINDS:
            raise OfflineManifestError(f"asset {url!r} has an unknown kind")
        if url in seen:
            raise OfflineManifestError(f"asset {url!r} is listed twice")
        if url == service_worker:
            raise OfflineManifestError("the service worker must not be precached")
        seen.add(url)
        assets.append((url, str(kind)))
    if start_url not in seen or dict(assets)[start_url] != "shell":
        raise OfflineManifestError("start url must be a listed shell asset")
    return profile, start_url, service_worker, assets


def build_profile(
    profile_root: Path = DEFAULT_PROFILE_ROOT,
    site_root: Path = DEFAULT_SITE_ROOT,
) -> tuple[OfflineProfile, dict[Path, str]]:
    """Build the offline profile and the files it should be written to.

    Args:
        profile_root: Directory that holds ``offline-assets.json`` and the
            service worker.
        site_root: Directory that bounds every precached file.

    Returns:
        The built profile and a mapping of output path to rendered text.

    Raises:
        OfflineManifestError: If the declaration or any asset is unsafe.
    """

    manifest_path = profile_root / MANIFEST_NAME
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OfflineManifestError("offline manifest is unreadable") from error
    profile_name, start_url, worker_name, declarations = _declarations(document)

    assets: list[OfflineAsset] = []
    total = 0
    for url, kind in declarations:
        content_type = content_type_for(url, kind)
        payload = resolve_asset_path(profile_root, site_root, url).read_bytes()
        if len(payload) > MAX_ASSET_BYTES:
            raise OfflineManifestError(f"asset {url!r} exceeds the size limit")
        total += len(payload)
        if total > MAX_TOTAL_BYTES:
            raise OfflineManifestError("offline profile exceeds the size limit")
        assets.append(
            OfflineAsset(
                url=url,
                kind=kind,
                content_type=content_type,
                size=len(payload),
                sha256=sha256_hex(payload),
            )
        )

    worker_path = profile_root / worker_name
    if worker_path.is_symlink() or not worker_path.is_file():
        raise OfflineManifestError("service worker is missing")
    worker_source = worker_path.read_text(encoding="utf-8")
    worker_digest = sha256_hex(blank_service_worker(worker_source).encode("utf-8"))
    profile = OfflineProfile(
        profile=profile_name,
        start_url=start_url,
        service_worker=worker_name,
        service_worker_sha256=worker_digest,
        generation=compute_generation(
            profile=profile_name,
            start_url=start_url,
            service_worker_sha256=worker_digest,
            assets=assets,
        ),
        assets=tuple(assets),
    )
    manifest = render_manifest(profile)
    return profile, {
        manifest_path: manifest,
        worker_path: render_service_worker(worker_source, profile, manifest),
    }


def stale_outputs(outputs: Mapping[Path, str]) -> list[Path]:
    """Return the output files whose current text differs from ``outputs``."""

    stale: list[Path] = []
    for path, expected in outputs.items():
        try:
            current = path.read_text(encoding="utf-8")
        except OSError:
            current = None
        if current != expected:
            stale.append(path)
    return stale


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_PROFILE_ROOT,
        help="profile directory holding offline-assets.json and the worker",
    )
    parser.add_argument(
        "--site-root",
        type=Path,
        default=DEFAULT_SITE_ROOT,
        help="directory that bounds every precached file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when generated files are stale",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build or check the offline profile.

    Args:
        argv: Optional command-line arguments.

    Returns:
        ``0`` when the files are current or were written, ``1`` when
        ``--check`` found stale files and ``2`` when the profile is invalid.
    """

    args = build_arg_parser().parse_args(argv)
    try:
        profile, outputs = build_profile(args.root, args.site_root)
    except OfflineManifestError as error:
        print(f"offline profile invalid: {error}", file=sys.stderr)
        return 2
    stale = stale_outputs(outputs)
    if args.check:
        if stale:
            names = ", ".join(sorted(path.name for path in stale))
            print(
                f"offline profile is stale ({names}); run "
                "python scripts/web/build_offline_manifest.py",
                file=sys.stderr,
            )
            return 1
        print(f"offline profile current: generation {profile.generation[:12]}")
        return 0
    for path in stale:
        path.write_text(outputs[path], encoding="utf-8")
    print(
        f"offline profile written: generation {profile.generation[:12]}, "
        f"{len(profile.assets)} assets, {profile.total_bytes} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
