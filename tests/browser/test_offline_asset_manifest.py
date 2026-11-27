"""Offline checks for the installable privacy-playground asset manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.web import build_offline_manifest as builder
from scripts.web.build_offline_manifest import (
    BLANK_BLOCK_BODY,
    OfflineManifestError,
    build_profile,
    compute_generation,
    main,
    stale_outputs,
    validate_asset_url,
)

ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND = ROOT / "docs" / "demo" / "privacy-playground"
DOCS = ROOT / "docs"
SERVICE_WORKER = PLAYGROUND / "offline-sw.js"


def _embedded_profile(source: str) -> dict[str, object]:
    match = re.search(
        r"^const OFFLINE_PROFILE = (\{.*?\});$", source, re.DOTALL | re.MULTILINE
    )
    assert match is not None, "service worker has no embedded offline profile"
    payload = json.loads(match.group(1))
    assert isinstance(payload, dict)
    return payload


def _synthetic_profile(tmp_path: Path, assets: list[dict[str, str]]) -> Path:
    site = tmp_path / "site"
    profile = site / "app"
    profile.mkdir(parents=True)
    (profile / "index.html").write_text("<!doctype html><title>x</title>\n")
    (profile / "app.js").write_text("export const value = 1;\n")
    (site / "shared.css").write_text("body { color: black; }\n")
    (profile / "worker.js").write_text(
        "// BEGIN GENERATED OFFLINE PROFILE "
        "(python scripts/web/build_offline_manifest.py)\n"
        f"{BLANK_BLOCK_BODY}\n"
        "// END GENERATED OFFLINE PROFILE\n"
    )
    (profile / "offline-assets.json").write_text(
        json.dumps(
            {
                "schema": builder.SCHEMA,
                "profile": "synthetic-profile",
                "start_url": "./",
                "service_worker": "worker.js",
                "assets": assets,
            }
        )
    )
    return profile


def test_committed_playground_profile_is_current() -> None:
    """The committed manifest and worker table match the files they pin."""

    profile, outputs = build_profile(PLAYGROUND, DOCS)

    assert stale_outputs(outputs) == [], (
        "run python scripts/web/build_offline_manifest.py to refresh the "
        "installable playground profile"
    )
    assert main(["--check"]) == 0
    assert profile.start_url == "./"
    assert profile.assets[0].url == "./"
    assert {asset.kind for asset in profile.assets} == {"shell"}


def test_worker_embeds_the_published_manifest_generation() -> None:
    """The worker refuses manifests that are not byte-identical to its table."""

    manifest_text = (PLAYGROUND / "offline-assets.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    embedded = _embedded_profile(SERVICE_WORKER.read_text(encoding="utf-8"))

    assert embedded["generation"] == manifest["generation"]
    assert embedded["assets"] == manifest["assets"]
    assert embedded["manifest_sha256"] == builder.sha256_hex(
        manifest_text.encode("utf-8")
    )
    assert embedded["service_worker_sha256"] == manifest["service_worker_sha256"]
    assert manifest["asset_count"] == len(manifest["assets"])
    assert manifest["total_bytes"] == sum(item["bytes"] for item in manifest["assets"])
    assert "offline-sw.js" not in {item["url"] for item in manifest["assets"]}
    assert "offline-assets.json" not in {item["url"] for item in manifest["assets"]}


def test_every_precached_asset_is_a_versioned_static_file() -> None:
    """Only same-origin static files are allowlisted; nothing dynamic."""

    manifest = json.loads((PLAYGROUND / "offline-assets.json").read_text("utf-8"))
    for asset in manifest["assets"]:
        url = validate_asset_url(asset["url"])
        assert "?" not in url and "#" not in url and "://" not in url
        assert re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])
        path = builder.resolve_asset_path(PLAYGROUND, DOCS, url)
        assert path.is_relative_to(DOCS)
        assert builder.sha256_hex(path.read_bytes()) == asset["sha256"]


def test_service_worker_never_writes_page_requests_to_a_cache() -> None:
    """Only the install and repair path calls cache.put, never the fetch path."""

    source = SERVICE_WORKER.read_text(encoding="utf-8")
    fetch_handler = source[
        source.index('self.addEventListener("fetch"') : source.index(
            'self.addEventListener("message"'
        )
    ]

    assert "cache.put" not in fetch_handler
    assert 'request.method !== "GET"' in fetch_handler
    assert "url.search" in fetch_handler
    assert source.count("cache.put(") == 1
    assert "skipWaiting" not in source.split("async function activateWaiting")[0]
    assert "clients.claim" not in source


def test_webmanifest_and_page_declare_the_local_profile() -> None:
    """The install manifest stays same-origin and the page opts in explicitly."""

    manifest = json.loads((PLAYGROUND / "manifest.webmanifest").read_text("utf-8"))
    page = (PLAYGROUND / "index.html").read_text(encoding="utf-8")
    controller = (PLAYGROUND / "offline.js").read_text(encoding="utf-8")

    assert manifest["start_url"] == manifest["scope"] == manifest["id"] == "./"
    assert manifest["display"] == "standalone"
    for icon in manifest["icons"]:
        assert "://" not in icon["src"]
        assert builder.resolve_asset_path(PLAYGROUND, DOCS, icon["src"]).is_file()
    assert '<link rel="manifest" href="./manifest.webmanifest" />' in page
    assert '<script type="module" src="./offline.js"></script>' in page
    assert "connect-src 'self'" in page
    # Registration happens only from the explicit prepare action.
    assert controller.count("serviceWorker.register(") == 1
    assert "async function prepare()" in controller
    assert "fetch(" not in controller


def test_generation_changes_with_any_byte_and_is_deterministic(
    tmp_path: Path,
) -> None:
    """One changed byte yields a new generation; rebuilding is a no-op."""

    profile_root = _synthetic_profile(
        tmp_path,
        [
            {"url": "./", "kind": "shell"},
            {"url": "app.js", "kind": "shell"},
            {"url": "../shared.css", "kind": "shell"},
        ],
    )
    site = profile_root.parent
    assert main(["--root", str(profile_root), "--site-root", str(site)]) == 0
    first = json.loads((profile_root / "offline-assets.json").read_text())
    assert main(["--root", str(profile_root), "--site-root", str(site), "--check"]) == 0

    (profile_root / "app.js").write_text("export const value = 2;\n")
    assert main(["--root", str(profile_root), "--site-root", str(site), "--check"]) == 1
    assert main(["--root", str(profile_root), "--site-root", str(site)]) == 0
    second = json.loads((profile_root / "offline-assets.json").read_text())

    assert first["generation"] != second["generation"]
    assert first["assets"][0] == second["assets"][0]
    assert first["assets"][1]["sha256"] != second["assets"][1]["sha256"]
    embedded = _embedded_profile((profile_root / "worker.js").read_text())
    assert embedded["generation"] == second["generation"]


def test_service_worker_logic_is_part_of_the_generation(tmp_path: Path) -> None:
    """Changing worker code outside the generated block starts a new generation."""

    profile_root = _synthetic_profile(tmp_path, [{"url": "./", "kind": "shell"}])
    site = profile_root.parent
    assert main(["--root", str(profile_root), "--site-root", str(site)]) == 0
    first = json.loads((profile_root / "offline-assets.json").read_text())

    worker = profile_root / "worker.js"
    worker.write_text(worker.read_text() + "// synthetic logic change\n")
    assert main(["--root", str(profile_root), "--site-root", str(site)]) == 0
    second = json.loads((profile_root / "offline-assets.json").read_text())

    assert first["service_worker_sha256"] != second["service_worker_sha256"]
    assert first["generation"] != second["generation"]


def test_generation_formula_matches_the_worker_contract() -> None:
    """The documented line format is the one the worker recomputes."""

    asset = builder.OfflineAsset(
        url="./",
        kind="shell",
        content_type="text/html; charset=utf-8",
        size=3,
        sha256="a" * 64,
    )
    expected = builder.sha256_hex(
        (
            "openmed.offline-assets.v1\n"
            "profile\tsynthetic\n"
            "start_url\t./\n"
            f"service_worker_sha256\t{'b' * 64}\n"
            f"asset\t./\tshell\ttext/html; charset=utf-8\t3\t{'a' * 64}\n"
        ).encode("utf-8")
    )

    assert (
        compute_generation(
            profile="synthetic",
            start_url="./",
            service_worker_sha256="b" * 64,
            assets=[asset],
        )
        == expected
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/app.js",
        "/docs/app.js",
        "app.js?version=1",
        "app.js#fragment",
        "a/../app.js",
        "./app.js",
        "app%2ejs",
        "..\\app.js",
        "../",
        "",
        "app.js|x",
    ],
)
def test_unsafe_asset_urls_are_rejected(url: str) -> None:
    """Queries, schemes, encodings and inner dot segments never enter the table."""

    with pytest.raises(OfflineManifestError):
        validate_asset_url(url)


@pytest.mark.parametrize(
    ("assets", "reason"),
    [
        ([{"url": "./", "kind": "shell"}, {"url": "./", "kind": "shell"}], "twice"),
        ([{"url": "./", "kind": "dynamic"}], "unknown kind"),
        ([{"url": "app.js", "kind": "shell"}], "start url"),
        (
            [{"url": "./", "kind": "shell"}, {"url": "worker.js", "kind": "shell"}],
            "must not be precached",
        ),
        (
            [
                {"url": "./", "kind": "shell"},
                {"url": "../../outside.css", "kind": "shell"},
            ],
            "outside the site root",
        ),
        (
            [{"url": "./", "kind": "shell"}, {"url": "missing.js", "kind": "shell"}],
            "missing",
        ),
        (
            [{"url": "./", "kind": "shell"}, {"url": "app.js", "kind": "model"}],
            "extension not allowed",
        ),
    ],
)
def test_invalid_declarations_fail_closed(
    tmp_path: Path, assets: list[dict[str, str]], reason: str
) -> None:
    """Duplicates, unknown kinds, escapes and missing files are refused."""

    profile_root = _synthetic_profile(tmp_path, assets)
    (tmp_path / "outside.css").write_text("body {}\n")

    site = str(profile_root.parent)
    with pytest.raises(OfflineManifestError, match=reason):
        build_profile(profile_root, profile_root.parent)
    assert main(["--root", str(profile_root), "--site-root", site]) == 2


def test_symlinked_assets_are_rejected(tmp_path: Path) -> None:
    """A symlink cannot smuggle an unrelated file into a generation."""

    profile_root = _synthetic_profile(
        tmp_path,
        [{"url": "./", "kind": "shell"}, {"url": "linked.js", "kind": "shell"}],
    )
    (profile_root / "linked.js").symlink_to(profile_root / "app.js")

    with pytest.raises(OfflineManifestError, match="symlink"):
        build_profile(profile_root, profile_root.parent)


def test_errors_never_include_absolute_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Command output names only public relative URLs and fixed reasons."""

    profile_root = _synthetic_profile(
        tmp_path,
        [{"url": "./", "kind": "shell"}, {"url": "missing.js", "kind": "shell"}],
    )

    assert main(["--root", str(profile_root), "--site-root", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert str(tmp_path) not in captured.err + captured.out
    assert "missing.js" in captured.err


def test_profile_can_include_runtime_and_model_assets(tmp_path: Path) -> None:
    """WASM, worker and model-bundle files are pinned like the app shell."""

    profile_root = _synthetic_profile(
        tmp_path,
        [
            {"url": "./", "kind": "shell"},
            {"url": "runtime/worker.mjs", "kind": "runtime"},
            {"url": "runtime/scorer.wasm", "kind": "runtime"},
            {"url": "models/synthetic/rules.json", "kind": "model"},
        ],
    )
    (profile_root / "runtime").mkdir()
    (profile_root / "runtime" / "worker.mjs").write_text("self.onmessage = null;\n")
    (profile_root / "runtime" / "scorer.wasm").write_bytes(b"\0asm\x01\0\0\0")
    (profile_root / "models" / "synthetic").mkdir(parents=True)
    (profile_root / "models" / "synthetic" / "rules.json").write_text("{}\n")

    profile, _ = build_profile(profile_root, profile_root.parent)

    by_url = {asset.url: asset for asset in profile.assets}
    assert by_url["runtime/scorer.wasm"].content_type == "application/wasm"
    assert by_url["runtime/worker.mjs"].kind == "runtime"
    assert by_url["models/synthetic/rules.json"].kind == "model"
    assert profile.total_bytes == sum(asset.size for asset in profile.assets)


def test_missing_generated_block_is_reported(tmp_path: Path) -> None:
    """A worker without the generated block cannot be stamped silently."""

    profile_root = _synthetic_profile(tmp_path, [{"url": "./", "kind": "shell"}])
    (profile_root / "worker.js").write_text("// no generated block\n")

    with pytest.raises(OfflineManifestError, match="generated block"):
        build_profile(profile_root, profile_root.parent)


def test_check_mode_does_not_write(tmp_path: Path) -> None:
    """``--check`` reports drift without modifying the profile."""

    source = _synthetic_profile(tmp_path, [{"url": "./", "kind": "shell"}])
    before = (source / "offline-assets.json").read_bytes()

    assert (
        main(["--root", str(source), "--site-root", str(source.parent), "--check"]) == 1
    )
    assert (source / "offline-assets.json").read_bytes() == before
