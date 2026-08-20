#!/usr/bin/env python3
"""Build the offline OpenMed public-claims snapshot from repository truth."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "docs/brand/system/claims.yml"
GITHUB_EVIDENCE = REPO_ROOT / "docs/brand/system/evidence/github-repository.json"
GITHUB_API_URL = "https://api.github.com/repos/maziyarpanahi/openmed"
AS_OF = "2026-07-29"
REVIEW_BY = "2026-10-29"
UNVERIFIED_FOLLOW_UP_BY = "2026-10-29"
MODEL_MANIFEST_AS_OF = "2026-08-11"
MODEL_MANIFEST_REVIEW_BY = "2026-11-11"
OWNER = "repository-owner"


def _read_github_evidence() -> dict[str, Any]:
    evidence = json.loads(GITHUB_EVIDENCE.read_text(encoding="utf-8"))
    if evidence["repository"] != "maziyarpanahi/openmed":
        raise RuntimeError("GitHub evidence targets the wrong repository")
    if evidence["api_url"] != GITHUB_API_URL:
        raise RuntimeError("GitHub evidence API URL is not canonical")
    count = evidence["stargazers_count"]
    display = evidence["display"]
    if not isinstance(count, int) or count < 0:
        raise RuntimeError("GitHub stargazers_count must be a non-negative integer")
    if display["method"] != "floor" or display["quantum"] != 100:
        raise RuntimeError("GitHub stars display must floor to a 100-star quantum")
    if display["value"] != count // 100 * 100:
        raise RuntimeError("GitHub stars display value is not conservatively rounded")
    if display["label"] != f"{display['value']:,}+ GitHub stars":
        raise RuntimeError("GitHub stars display label is stale")
    return evidence


def _refresh_github_evidence() -> None:
    request = urllib.request.Request(
        GITHUB_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "openmed-offline-claims-refresh",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    if payload.get("full_name") != "maziyarpanahi/openmed":
        raise RuntimeError("GitHub API returned an unexpected repository")
    count = payload.get("stargazers_count")
    if not isinstance(count, int) or count < 0:
        raise RuntimeError("GitHub API returned an invalid stargazers_count")

    captured = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    display_value = count // 100 * 100
    evidence = {
        "schema_version": 2,
        "repository": payload["full_name"],
        "api_url": GITHUB_API_URL,
        "html_url": payload["html_url"],
        "captured_at": captured.isoformat().replace("+00:00", "Z"),
        "repository_updated_at": payload["updated_at"],
        "stargazers_count": count,
        "display": {
            "method": "floor",
            "quantum": 100,
            "value": display_value,
            "label": f"{display_value:,}+ GitHub stars",
        },
        "owner": OWNER,
        "review_by": (captured.date() + dt.timedelta(days=31)).isoformat(),
        "refresh": {
            "command": ("python scripts/brand/update_claims.py --refresh-github-stars"),
            "network": "explicit opt-in only",
            "ci": "forbidden",
        },
    }
    GITHUB_EVIDENCE.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_version() -> str:
    source = (REPO_ROOT / "openmed/__about__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', source, re.MULTILINE)
    if not match:
        raise RuntimeError("openmed/__about__.py does not declare __version__")
    return match.group(1)


def _read_entity_types() -> list[str]:
    source = (REPO_ROOT / "openmed/core/model_registry.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "_PII_ENTITY_TYPES"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            break
        return value
    raise RuntimeError("could not read _PII_ENTITY_TYPES from model_registry.py")


def _read_language_claims() -> dict[str, list[str]]:
    sys.path.insert(0, str(REPO_ROOT))
    from openmed.core.language_pack_catalog import (  # noqa: PLC0415
        DEFAULT_MODEL_PLACEHOLDER_LANGUAGES,
        DEFAULT_PII_MODELS,
        NATIONAL_ID_ONLY_LANGUAGES,
        USER_SUPPLIED_MODEL_LANGUAGES,
    )

    supported = sorted(DEFAULT_PII_MODELS)
    placeholders = sorted(DEFAULT_MODEL_PLACEHOLDER_LANGUAGES)
    model_backed = sorted(set(supported) - set(placeholders))
    return {
        "supported": supported,
        "model_backed": model_backed,
        "placeholder": placeholders,
        "user_supplied": sorted(USER_SUPPLIED_MODEL_LANGUAGES),
        "national_id_only": sorted(NATIONAL_ID_ONLY_LANGUAGES),
    }


def _read_model_manifest() -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "models.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    repo_ids = [row["repo_id"] for row in rows]
    if len(repo_ids) != len(set(repo_ids)):
        raise RuntimeError("models.jsonl contains duplicate repo_id values")
    license_counts: dict[str, int] = {}
    for row in rows:
        license_name = str(row.get("license") or "unknown")
        license_counts[license_name] = license_counts.get(license_name, 0) + 1

    return {
        "rows": len(rows),
        "openmed_owned": sum(repo_id.startswith("OpenMed/") for repo_id in repo_ids),
        "pii_family": sum(row.get("family") == "PII" for row in rows),
        "mlx": sum(
            any(str(fmt).startswith("mlx") for fmt in row.get("formats", []))
            for row in rows
        ),
        "licenses": dict(sorted(license_counts.items())),
    }


def _claim(
    *,
    status: str,
    value: Any,
    display: str | None,
    definition: str,
    source: str,
    public_wording: str,
    qualification: str,
    as_of: str | None = AS_OF,
    review_by: str | None = REVIEW_BY,
    owner: str = OWNER,
) -> dict[str, Any]:
    return {
        "status": status,
        "value": value,
        "display": display,
        "definition": definition,
        "source": source,
        "as_of": as_of,
        "owner": owner,
        "public_wording": public_wording,
        "qualification": qualification,
        "review_by": review_by,
        "follow_up_by": (UNVERIFIED_FOLLOW_UP_BY if status == "unverified" else None),
    }


def build_registry() -> dict[str, Any]:
    """Return the complete committed public-claims snapshot."""

    languages = _read_language_claims()
    manifest = _read_model_manifest()
    entity_types = _read_entity_types()
    version = _read_version()
    github = _read_github_evidence()

    claims = {
        "package_version": _claim(
            status="verified",
            value=version,
            display=f"OpenMed SDK {version}",
            definition="Version declared by openmed/__about__.py.",
            source="openmed/__about__.py",
            public_wording=f"OpenMed SDK {version}",
            qualification="Package version, not proof of PyPI publication.",
        ),
        "github_stars_snapshot": _claim(
            status="verified",
            value=github["stargazers_count"],
            display=github["display"]["label"],
            definition=(
                "Exact stargazers_count from the committed GitHub API evidence; "
                "public display is conservatively floored to the preceding "
                f"{github['display']['quantum']}-star quantum."
            ),
            source="docs/brand/system/evidence/github-repository.json",
            public_wording=github["display"]["label"],
            qualification=(
                "Dated offline snapshot, not a live counter; the exact raw "
                f"value was {github['stargazers_count']:,} when captured."
            ),
            as_of=github["captured_at"][:10],
            review_by=github["review_by"],
            owner=github["owner"],
        ),
        "repository_model_snapshot": _claim(
            status="verified",
            value=manifest["rows"],
            display=f"{manifest['rows']:,} manifest entries",
            definition=(
                "Unique repo_id rows in committed models.jsonl; no rounding. "
                "This is a repository snapshot, not a live Hugging Face count."
            ),
            source="models.jsonl",
            public_wording=(
                f"The committed catalog snapshot contains "
                f"{manifest['rows']:,} unique entries."
            ),
            qualification="Dated offline snapshot; availability can change.",
            as_of=MODEL_MANIFEST_AS_OF,
            review_by=MODEL_MANIFEST_REVIEW_BY,
        ),
        "hugging_face_openmed_owned_snapshot": _claim(
            status="verified",
            value=manifest["openmed_owned"],
            display=f"{manifest['openmed_owned']:,} OpenMed-owned entries",
            definition=(
                "Committed manifest repo_id rows whose exact namespace is "
                "OpenMed; de-duplicated by repo_id."
            ),
            source="models.jsonl",
            public_wording=(
                f"The committed snapshot records "
                f"{manifest['openmed_owned']:,} OpenMed-namespace entries."
            ),
            qualification="Not a live organization inventory.",
            as_of=MODEL_MANIFEST_AS_OF,
            review_by=MODEL_MANIFEST_REVIEW_BY,
        ),
        "broader_compatible_model_count": _claim(
            status="unverified",
            value=None,
            display=None,
            definition=(
                "Curated compatible models across owners after explicit "
                "license, availability, and repo_id de-duplication."
            ),
            source="No current cross-owner committed registry.",
            public_wording="",
            qualification="Do not publish a count until a source is committed.",
            as_of=None,
            review_by=None,
        ),
        "supported_pii_languages": _claim(
            status="verified",
            value=len(languages["supported"]),
            display=f"{len(languages['supported'])} supported PII routes",
            definition=(
                "Exact keys in DEFAULT_PII_MODELS, including named placeholder "
                "routes; no locale aliases."
            ),
            source="openmed/core/language_pack_catalog.py",
            public_wording=(
                f"{len(languages['supported'])} supported PII language routes"
            ),
            qualification=(
                "Supported routes include the separately identified placeholder "
                "route; do not call every route model-backed."
            ),
        ),
        "supported_pii_language_codes": _claim(
            status="verified",
            value=languages["supported"],
            display=None,
            definition="Sorted exact keys in DEFAULT_PII_MODELS.",
            source="openmed/core/language_pack_catalog.py",
            public_wording=", ".join(languages["supported"]),
            qualification="Use codes only where the definition is visible.",
        ),
        "model_backed_pii_languages": _claim(
            status="verified",
            value=len(languages["model_backed"]),
            display=f"{len(languages['model_backed'])} model-backed PII languages",
            definition=(
                "DEFAULT_PII_MODELS keys minus DEFAULT_MODEL_PLACEHOLDER_LANGUAGES."
            ),
            source="openmed/core/language_pack_catalog.py",
            public_wording=(
                f"{len(languages['model_backed'])} model-backed PII languages"
            ),
            qualification="Excludes placeholder and user-supplied routes.",
        ),
        "model_backed_pii_language_codes": _claim(
            status="verified",
            value=languages["model_backed"],
            display=None,
            definition=(
                "Sorted DEFAULT_PII_MODELS keys after removing placeholder languages."
            ),
            source="openmed/core/language_pack_catalog.py",
            public_wording=", ".join(languages["model_backed"]),
            qualification="Codes are model-backed, not benchmark equivalence.",
        ),
        "placeholder_pii_languages": _claim(
            status="verified",
            value=languages["placeholder"],
            display=None,
            definition="Named built-in routes using a placeholder model.",
            source="openmed/core/language_pack_catalog.py",
            public_wording="",
            qualification="Never advertise as trained/model-backed coverage.",
        ),
        "user_supplied_model_languages": _claim(
            status="verified",
            value=languages["user_supplied"],
            display=None,
            definition=(
                "Script-routable language codes requiring a user-configured "
                "model; not built-in model support."
            ),
            source="openmed/core/language_pack_catalog.py",
            public_wording="",
            qualification="Call these optional user-configured routes.",
        ),
        "national_id_only_languages": _claim(
            status="verified",
            value=languages["national_id_only"],
            display=None,
            definition=(
                "Compatibility declarations with deterministic national-ID "
                "recognition but no complete default PII language pack."
            ),
            source="openmed/core/language_pack_catalog.py",
            public_wording="",
            qualification="Do not include in model-backed language totals.",
        ),
        "pii_family_manifest_entries": _claim(
            status="verified",
            value=manifest["pii_family"],
            display=f"{manifest['pii_family']} PII-family manifest entries",
            definition="Rows in models.jsonl whose family is exactly PII.",
            source="models.jsonl",
            public_wording=(
                f"The committed snapshot has {manifest['pii_family']} "
                "PII-family entries."
            ),
            qualification="Snapshot entries, not live checkpoints or downloads.",
            as_of=MODEL_MANIFEST_AS_OF,
            review_by=MODEL_MANIFEST_REVIEW_BY,
        ),
        "pii_checkpoint_count": _claim(
            status="unverified",
            value=None,
            display=None,
            definition=(
                "Distinct PII checkpoints after repository, revision, and "
                "artifact de-duplication."
            ),
            source="No approved committed checkpoint inventory.",
            public_wording="",
            qualification=(
                "Do not treat PII-family manifest rows as distinct checkpoints."
            ),
            as_of=None,
            review_by=None,
        ),
        "dataset_count": _claim(
            status="unverified",
            value=None,
            display=None,
            definition=(
                "Distinct datasets after source, license, access, and "
                "de-duplication rules."
            ),
            source="No approved committed dataset registry.",
            public_wording="",
            qualification=(
                "Do not publish a dataset total until the governed inventory "
                "and inclusion rules are reviewed."
            ),
            as_of=None,
            review_by=None,
        ),
        "mlx_manifest_entries": _claim(
            status="verified",
            value=manifest["mlx"],
            display=f"{manifest['mlx']} MLX-format manifest entries",
            definition=(
                "Rows in models.jsonl with at least one format value beginning "
                "with mlx; each row counted once."
            ),
            source="models.jsonl",
            public_wording=(
                f"The committed snapshot has {manifest['mlx']} MLX-format entries."
            ),
            qualification="Manifest format availability can change upstream.",
            as_of=MODEL_MANIFEST_AS_OF,
            review_by=MODEL_MANIFEST_REVIEW_BY,
        ),
        "pii_entity_types": _claim(
            status="verified",
            value=len(entity_types),
            display=f"{len(entity_types)} PII entity labels",
            definition=(
                "Exact unique labels in model_registry._PII_ENTITY_TYPES; "
                "not HIPAA Safe Harbor identifier classes."
            ),
            source="openmed/core/model_registry.py",
            public_wording=f"{len(entity_types)} PII entity labels",
            qualification="Entity-label coverage varies by selected model.",
        ),
        "pii_entity_type_codes": _claim(
            status="verified",
            value=entity_types,
            display=None,
            definition="Ordered model-registry PII label vocabulary.",
            source="openmed/core/model_registry.py",
            public_wording="",
            qualification="Do not imply every model detects every label.",
        ),
        "model_license_population": _claim(
            status="verified",
            value=manifest["licenses"],
            display="mixed and partially unknown",
            definition=(
                "Exact license field values in the committed model manifest; "
                "missing values remain unknown."
            ),
            source="models.jsonl",
            public_wording="Model licenses vary; review each model repository.",
            qualification=("Never inherit the SDK's Apache-2.0 license onto models."),
            as_of=MODEL_MANIFEST_AS_OF,
            review_by=MODEL_MANIFEST_REVIEW_BY,
        ),
        "sdk_license": _claim(
            status="verified",
            value="Apache-2.0",
            display="Apache-2.0 SDK",
            definition="License of the OpenMed repository software.",
            source="LICENSE",
            public_wording="Apache-2.0 SDK",
            qualification=(
                "Applies to the SDK source, not every model, dataset, Agent, "
                "Welna, or external product."
            ),
            review_by="2027-07-29",
        ),
        "runtime_locality": _claim(
            status="verified",
            value="local-first",
            display="local-first",
            definition=(
                "Core inference can run on controlled hardware after required "
                "artifacts are present; adapters and downloads may use a network."
            ),
            source="README.md and openmed runtime configuration",
            public_wording="Local-first clinical AI on hardware you control.",
            qualification=(
                "Do not promise no network calls for download, remote-provider, "
                "telemetry-enabled, or user-configured integration paths."
            ),
        ),
        "runtime_behavior_by_surface": _claim(
            status="verified",
            value={
                "core_sdk": (
                    "Local processing after required artifacts are available; "
                    "local_only mode blocks outbound sockets after model loading."
                ),
                "artifact_acquisition": (
                    "Model, evaluation-data, and optional vocabulary downloads "
                    "may use their configured network sources."
                ),
                "optional_integrations": (
                    "Remote-provider adapters, telemetry-enabled paths, and "
                    "user-configured integrations may use a network."
                ),
                "browser_demo": (
                    "Accepts same-origin runtime and model URLs only; the page "
                    "does not upload entered text."
                ),
            },
            display="runtime-specific local and network boundaries",
            definition=(
                "Approved locality and network behavior for each published "
                "runtime surface."
            ),
            source=(
                "README.md, docs/configuration.md, "
                "docs/security/no-telemetry.md, and docs/demo/web/README.md"
            ),
            public_wording=(
                "Core processing is local after required artifacts are present; "
                "downloads and explicitly configured integrations may use a network."
            ),
            qualification=(
                "Describe the selected runtime and configuration; never turn "
                "local-first into an unconditional no-network promise."
            ),
        ),
        "compliance_wording": _claim(
            status="verified",
            value="qualified",
            display=None,
            definition="Approved boundary for compliance language.",
            source="docs/compliance.md and docs/security/no-telemetry.md",
            public_wording=(
                "OpenMed can support a deployment's de-identification controls."
            ),
            qualification=(
                "The SDK does not make a deployment HIPAA/GDPR compliant and "
                "does not replace expert review."
            ),
        ),
        "product_maturity": _claim(
            status="verified",
            value="software-library",
            display=None,
            definition="OpenMed is a software library and research toolkit.",
            source="README.md and docs",
            public_wording="Software library and research toolkit.",
            qualification=(
                "Not a medical device; must not auto-trigger clinical decisions."
            ),
        ),
        "dataset_access": _claim(
            status="verified",
            value="mixed",
            display=None,
            definition="Dataset access and licenses are source-specific.",
            source="NOTICE and documentation dataset references",
            public_wording="Dataset access and terms vary by source.",
            qualification=(
                "DUA-, credential-, UMLS-, SNOMED-, CPT-, MIMIC-, i2b2-, and "
                "n2c2-gated data is not bundled."
            ),
        ),
        "license_by_product_surface": _claim(
            status="verified",
            value={
                "openmed_sdk_source": "Apache-2.0",
                "catalog_models": "mixed and partially unknown",
                "referenced_datasets": "source-specific access and license terms",
                "openmed_agent": "separate product terms",
                "welna": "separate product terms",
                "external_integrations": "upstream terms",
            },
            display="surface-specific terms",
            definition=(
                "The approved license boundary for each named product or "
                "dependency surface."
            ),
            source="LICENSE, models.jsonl, NOTICE, README.md, and documentation",
            public_wording=(
                "The OpenMed SDK source is Apache-2.0; model, dataset, OpenMed "
                "Agent, Welna, and integration terms are separate."
            ),
            qualification=(
                "Never inherit the SDK license onto a model, dataset, product, "
                "or external integration."
            ),
        ),
    }

    for name, definition in {
        "cumulative_model_downloads": "Live cumulative model downloads.",
        "monthly_model_downloads": "Live monthly model downloads.",
        "cumulative_package_installs": "Live cumulative package installs.",
        "release_cadence": "Measured release cadence over a declared interval.",
        "community_timeline": "Founding and community timeline facts.",
        "benchmark_performance": (
            "Benchmark claim with hardware, versions, method, result, and date."
        ),
        "research_sota": (
            "Research or SOTA claim with paper and reproducibility evidence."
        ),
        "competitive_matrix": (
            "Competitor comparison with per-cell sources and an as-of date."
        ),
    }.items():
        claims[name] = _claim(
            status="unverified",
            value=None,
            display=None,
            definition=definition,
            source="No approved committed snapshot.",
            public_wording="",
            qualification="Do not publish until evidence is reviewed.",
            as_of=None,
            review_by=None,
        )

    return {
        "schema_version": 2,
        "generated_at": MODEL_MANIFEST_AS_OF,
        "generation": {
            "command": "python scripts/brand/update_claims.py --write",
            "network": "forbidden",
            "network_refresh_command": (
                "python scripts/brand/update_claims.py --refresh-github-stars"
            ),
            "network_refresh_ci_policy": "never invoke from CI",
            "rounding": "none unless a claim definition explicitly says otherwise",
        },
        "claims": claims,
    }


def _serialized_registry() -> str:
    return json.dumps(build_registry(), ensure_ascii=False, indent=2) + "\n"


def _json_script(payload: dict[str, Any]) -> str:
    return (
        '<script type="application/ld+json">\n'
        + json.dumps(payload, ensure_ascii=False, indent=4)
        + "\n</script>"
    )


def _website_fragments(registry: dict[str, Any]) -> dict[str, str]:
    claims = registry["claims"]
    version = claims["package_version"]["value"]
    stars = claims["github_stars_snapshot"]
    supported_language_count = claims["supported_pii_languages"]["value"]
    supported_language_codes = claims["supported_pii_language_codes"]["value"]
    model_backed_language_count = claims["model_backed_pii_languages"]["value"]
    supported_language_codes_text = (
        ", ".join(supported_language_codes[:-1])
        + f", and {supported_language_codes[-1]}"
    )
    as_of = dt.date.fromisoformat(claims["repository_model_snapshot"]["as_of"])
    compact_stars = stars["display"].split()[0]
    numeric_stars = int(compact_stars.rstrip("+").replace(",", ""))
    display_stars = f"{numeric_stars / 1000:.1f}k"
    metadata_title = "OpenMed — local-first clinical AI"
    metadata_description = (
        "OpenMed reads clinical text and removes 55+ PHI types on hardware "
        "you control. Explore 2,000+ open models, on-device runtimes, and "
        "reproducible biomedical NER benchmarks."
    )
    identity_answer = (
        "Apache-2.0 clinical NLP and de-identification software that runs on "
        "hardware you control, from phones and browsers to GPU servers."
    )

    public_metadata = f"""<title>{metadata_title}</title>
<meta
    name="description"
    content="{metadata_description}"
>
<meta
    name="keywords"
    content="OpenMed, clinical NLP, biomedical NER, PHI de-identification, local-first AI, healthcare privacy, MLX, OpenMedKit, WebGPU"
>
<meta name="author" content="OpenMed">
<meta name="robots" content="index,follow">
<link rel="canonical" href="https://openmed.life/">

<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="icon" href="favicon-64.png" type="image/png" sizes="64x64">
<link rel="icon" href="favicon.ico" sizes="48x48">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
<link rel="manifest" href="site.webmanifest">
<link
    rel="preload"
    href="assets/fonts/IBMPlexSans-Regular.woff2"
    as="font"
    type="font/woff2"
    crossorigin
>
<link
    rel="preload"
    href="assets/fonts/IBMPlexSans-SemiBold.woff2"
    as="font"
    type="font/woff2"
    crossorigin
>
<link
    rel="preload"
    href="assets/fonts/IBMPlexMono-Regular.woff2"
    as="font"
    type="font/woff2"
    crossorigin
>
<meta
    id="themeColorLight"
    data-theme-color
    name="theme-color"
    content="#F4F7F8"
    media="(prefers-color-scheme: light)"
>
<meta
    id="themeColorDark"
    data-theme-color
    name="theme-color"
    content="#0B0E13"
    media="(prefers-color-scheme: dark)"
>

<meta property="og:type" content="website">
<meta property="og:title" content="{metadata_title}">
<meta property="og:description" content="{metadata_description}">
<meta property="og:url" content="https://openmed.life/">
<meta property="og:site_name" content="OpenMed">
<meta property="og:image" content="https://openmed.life/og.png">
<meta property="og:image:secure_url" content="https://openmed.life/og.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta
    property="og:image:alt"
    content="OpenMed social card with the Open Cross and the words “Your data. Your model. Your hardware.”; its footer lists openmed.life, 2,000+ models, 340M+ downloads, 10M+ installs, and Apache-2.0."
>
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@OpenMed_AI">
<meta name="twitter:title" content="{metadata_title}">
<meta name="twitter:description" content="{metadata_description}">
<meta name="twitter:image" content="https://openmed.life/og.png">
<meta
    name="twitter:image:alt"
    content="OpenMed social card with the Open Cross and the words “Your data. Your model. Your hardware.”; its footer lists openmed.life, 2,000+ models, 340M+ downloads, 10M+ installs, and Apache-2.0."
>"""

    software_source = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode",
        "name": "OpenMed",
        "description": identity_answer,
        "codeRepository": "https://github.com/maziyarpanahi/openmed",
        "license": "https://www.apache.org/licenses/LICENSE-2.0",
        "programmingLanguage": ["Python", "Swift", "Kotlin", "TypeScript"],
        "softwareVersion": version,
        "url": "https://openmed.life/",
        "isAccessibleForFree": True,
    }
    faq_items = [
        (
            "How is this different from a cloud medical NLP API?",
            "Cloud APIs bill per character and require sending clinical text to a "
            "vendor region — every new use case becomes a privacy review. OpenMed "
            "downloads once and runs in your process: laptop, VPC, or air-gapped "
            "box, with no telemetry and no licence check-in. It is also the only "
            "option here that runs inside an iPhone or a browser tab.",
        ),
        (
            "And versus closed-source healthcare NLP?",
            "Those are mature platforms with strong published accuracy and support "
            "contracts — if you need a vendor SLA, that is a real answer. OpenMed "
            "competes on ownership: no per-server subscription, Apache-2.0 weights "
            "and recipes, mobile and browser runtimes, and a release most weeks "
            "instead of quarterly. Our benchmarks you can rerun; theirs you take "
            "on trust.",
        ),
        (
            "Can we fine-tune the models on our own vocabulary?",
            "Yes. Every model ships with its full training recipe: domain-adaptive "
            "pre-training plus LoRA, updating under 1.5% of parameters. A full run "
            "finishes in under 12 hours on a single GPU (under 1.2 kg CO2e), and "
            "starter notebooks live in the openmed-starter repo.",
        ),
        (
            "Is a weekly release cadence safe for clinical use?",
            "Releases are versioned, signed and documented, with pinned model "
            "artifacts, calibrated thresholds and release gates that fail closed. "
            "Pin a version and upgrade on your schedule — the cadence just means a "
            "gap you report closes in days, not in the next procurement cycle.",
        ),
        (
            'What does "community-driven" actually mean here?',
            "Language packs, national-ID validators, translations and mobile "
            "runtimes have all come from people outside the original author. Issues "
            "and PRs are public, security disclosure is documented, and PHI-leak "
            "reports get a private channel. If your locale is missing, requesting "
            "it is a realistic path to it existing.",
        ),
        (
            "Are these generative models?",
            "No. They are compact encoder transformers fine-tuned for token "
            "classification — they label spans of text rather than generating it, "
            "so there is nothing to hallucinate. Small enough to run on a phone; "
            "the Privacy Filter family adds the de-identification policy layer on "
            "top.",
        ),
    ]
    faq = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": question,
                "acceptedAnswer": {"@type": "Answer", "text": answer},
            }
            for question, answer in faq_items
        ],
    }

    package_version_header = f"""<a
    class="release-chip"
    href="https://github.com/maziyarpanahi/openmed/releases"
    aria-label="OpenMed SDK version {version}, current release"
    data-release-label
>
    <span class="status-dot" aria-hidden="true"></span>
    v{version} current release
</a>"""
    github_stars = f"""<a
    class="release-chip repository-stars"
    href="https://github.com/maziyarpanahi/openmed/stargazers"
    aria-label="{stars["display"]}"
    data-repository-stars
>
    <span class="star-symbol" aria-hidden="true">★</span>
    <span data-star-count>{display_stars}</span>
</a>"""
    hero = f"""<p class="eyebrow">
    <span class="status-dot status-dot-accent" aria-hidden="true"></span>
    OpenMed {version} · on-device clinical AI
</p>
<h1>
    Your data. Your model. Your
    <span class="sr-only">hardware.</span>
    <span class="rotating-wrap" aria-hidden="true">
        <span class="rotating-word" data-rotating-word aria-hidden="true">
            hardware.
        </span>
    </span>
</h1>
<p class="hero-lead">
    OpenMed reads clinical text and removes 55+ PHI types on the
    hardware you control, so patient data never leaves the device.
    2,000+ Apache-2.0 models, {model_backed_language_count} model-backed PII languages,
    state of the art on 10 of 12 biomedical NER benchmarks —
    and a new release most weeks.
</p>
<div class="button-row">
    <a
        class="button button-ink button-large"
        href="https://github.com/maziyarpanahi/openmed"
        target="_blank"
        rel="noopener"
    >
        View on GitHub <span aria-hidden="true">↗</span>
    </a>
    <a class="button button-outline button-large" href="#compare">
        Compare with your vendor <span aria-hidden="true">→</span>
    </a>
</div>
<button
    class="install-command js-only"
    type="button"
    aria-label="Copy pip install command"
    data-copy-text="pip install openmed"
    data-copy-label="Copy pip install command"
>
    <span class="prompt" aria-hidden="true">$</span>
    <code>pip install openmed</code>
    <span class="copy-label" data-copy-feedback aria-hidden="true">⧉</span>
</button>"""
    repository_snapshot = f"""<div class="community-grid">
    <div class="community-lead">
        <p class="mono-label">Model downloads · all-time</p>
        <p class="community-number">340<span>M</span></p>
        <p>
            One person on lunch breaks in July 2025; the largest open
            medical-AI collection by July 2026. Founded by
            <span class="accent-text">Maziyar Panahi</span>.
            <a href="https://github.com/maziyarpanahi/openmed">
                Join on GitHub <span aria-hidden="true">↗</span>
            </a>
        </p>
    </div>
    <div class="numbers-wall" aria-label="OpenMed community statistics">
        <div>
            <strong>30<span>M</span></strong>
            <small>Every month</small>
        </div>
        <div>
            <strong>9.4<span>M</span></strong>
            <small>PyPI installs</small>
        </div>
        <div>
            <strong>2,000<span>+</span></strong>
            <small>Open models</small>
        </div>
        <div>
                            <strong>{supported_language_count}</strong>
                            <small>Languages</small>
        </div>
    </div>
</div>

<div class="facts-rail">
    <div>
        <strong>A release every week</strong>
        <span>13 releases in the first six months</span>
    </div>
    <div>
        <strong>{supported_language_count} languages supported</strong>
        <span>Docs · models · community-maintained</span>
    </div>
    <div>
        <strong>Free forever</strong>
        <span>Apache-2.0 · nothing to sign</span>
    </div>
</div>
<div class="snapshot-note">
    <span>Counted, not claimed · Hugging Face + PyPI · July 2026</span>
    <span data-community-stars>★ {display_stars.upper()} GitHub stars · counted live</span>
</div>"""
    privacy_contract = f"""<h2>De-identification you can audit, not just trust.</h2>
<p class="section-lead">
    All 18 HIPAA Safe Harbor identifiers inside a 55+ entity
    catalog: {model_backed_language_count} model-backed PII languages,
    {supported_language_count} supported codes,
    600+ PII models, plus validator-backed ID-only locales.
    Policy-aware pipelines add calibrated thresholds, signed audit
    reports, re-identification risk scoring, and release gates that
    fail closed when the evidence is missing.
    <span class="sr-only">
        {supported_language_count} supported PII language codes:
        {supported_language_codes_text}
    </span>
</p>
"""

    return {
        "public_metadata": public_metadata,
        "software_source_jsonld": _json_script(software_source),
        "faq_jsonld": _json_script(faq),
        "package_version_header": package_version_header,
        "github_stars": github_stars,
        "hero_claims": hero,
        "repository_snapshot": repository_snapshot,
        "package_version_quickstart": (f"<span>openmed {version} · quickstart</span>"),
        "competitive_matrix_boundary": """<p class="table-note">
    <span>Capability and cadence rows describe publicly documented positions · July 2026</span>
    <span>Sources: vendor docs · arXiv:2508.01630 · OpenMed release feed</span>
</p>""",
        "privacy_contract": privacy_contract,
        "footer_license": f"""<span>
    © <span id="year">{as_of.year}</span> OpenMed · Apache-2.0 · openmed.life
</span>""",
    }


def _replace_website_marker(text: str, name: str, content: str) -> str:
    pattern = re.compile(
        rf"(?ms)^(?P<indent>[ \t]*)"
        rf"<!-- openmed-claim:{re.escape(name)}:start -->\n"
        rf".*?"
        rf"^(?P=indent)<!-- openmed-claim:{re.escape(name)}:end -->"
    )

    def replacement(match: re.Match[str]) -> str:
        indent = match.group("indent")
        indented = "\n".join(
            indent + line if line else "" for line in content.strip().splitlines()
        )
        return (
            f"{indent}<!-- openmed-claim:{name}:start -->\n"
            f"{indented}\n"
            f"{indent}<!-- openmed-claim:{name}:end -->"
        )

    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"website marker {name!r} is missing or duplicated")
    return updated


def _organization_jsonld(text: str) -> dict[str, Any]:
    marker = re.search(
        r"(?s)<!-- openmed-claim:organization_jsonld:start -->"
        r"\s*<script type=\"application/ld\+json\">\s*(.*?)\s*</script>\s*"
        r"<!-- openmed-claim:organization_jsonld:end -->",
        text,
    )
    if not marker:
        raise RuntimeError("website organization_jsonld marker is invalid")
    return json.loads(marker.group(1))


def _sync_website(
    registry: dict[str, Any],
    *,
    write: bool,
) -> list[str]:
    path = REPO_ROOT / "docs/website/index.html"
    current = path.read_text(encoding="utf-8")
    expected = current
    fragments = _website_fragments(registry)
    for name, content in fragments.items():
        expected = _replace_website_marker(expected, name, content)

    organization = _organization_jsonld(expected)
    required_organization = {
        "@context": "https://schema.org",
        "@type": "Organization",
        "name": "OpenMed",
        "url": "https://openmed.life/",
        "logo": "https://openmed.life/logo.svg",
        "sameAs": [
            "https://github.com/maziyarpanahi/openmed",
            "https://huggingface.co/OpenMed",
            "https://x.com/OpenMed_AI",
            "https://www.linkedin.com/company/openmed-ai/",
        ],
        "founder": {"@type": "Person", "name": "Maziyar Panahi"},
    }
    errors: list[str] = []
    if organization != required_organization:
        errors.append("website organization JSON-LD identity links are not governed")
    if write and expected != current:
        path.write_text(expected, encoding="utf-8")
        print(f"updated {path.relative_to(REPO_ROOT)} claim fragments")
    elif not write and expected != current:
        errors.append("website claim fragments are stale")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--refresh-github-stars", action="store_true")
    args = parser.parse_args()

    if args.refresh_github_stars:
        _refresh_github_evidence()
        registry = build_registry()
        OUTPUT.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        website_errors = _sync_website(registry, write=True)
        if website_errors:
            print("\n".join(website_errors), file=sys.stderr)
            return 1
        print(
            "refreshed the opt-in GitHub evidence and rebuilt "
            f"{OUTPUT.relative_to(REPO_ROOT)}"
        )
        return 0

    registry = build_registry()
    expected = json.dumps(registry, ensure_ascii=False, indent=2) + "\n"
    if args.write:
        OUTPUT.write_text(expected, encoding="utf-8")
        website_errors = _sync_website(registry, write=True)
        if website_errors:
            print("\n".join(website_errors), file=sys.stderr)
            return 1
        print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
        return 0

    website_errors = _sync_website(registry, write=False)
    actual = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
    if actual != expected or website_errors:
        for error in website_errors:
            print(error, file=sys.stderr)
        print(
            "claims/website snapshot is stale; run "
            "python scripts/brand/update_claims.py --write",
            file=sys.stderr,
        )
        return 1
    print("claims snapshot is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
