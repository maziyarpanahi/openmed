"""Source-of-truth guards for the service incident runbooks.

Every environment variable, module path, imported name, CLI command, route,
reason code, Helm value and Compose object cited by the runbooks must exist in
the repository. Each checker has a negative control so a renamed or invented
setting fails the suite instead of shipping as stale operator guidance.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
RUNBOOK_DIR = DOCS / "operations" / "runbooks"
OVERVIEW = RUNBOOK_DIR / "service-runbooks.md"
SCENARIO_PAGES = {
    "credential-compromise.md": 3,
    "client-certificate-compromise.md": 1,
    "disk-exhaustion.md": 3,
    "store-corruption.md": 3,
}
RUNBOOKS = (OVERVIEW, *(RUNBOOK_DIR / name for name in SCENARIO_PAGES))
MKDOCS = ROOT / "mkdocs.yml"
PUBLICATION = DOCS / "brand" / "system" / "publication.yml"
HELM_VALUES = ROOT / "deploy" / "helm" / "openmed-service" / "values.yaml"
COMPOSE = ROOT / "deploy" / "openmed-compose.yaml"
SOURCE_ROOTS = (ROOT / "openmed", ROOT / "deploy")
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".tpl", ".toml"}

REQUIRED_SCENARIO_SECTIONS = (
    "### Detection signals",
    "### Containment",
    "### Recovery",
    "### Verification",
    "### Keep out of tickets and logs",
)
PENDING_ISSUES = {"3574", "3575", "3577", "3580", "3600", "3603", "3699", "3703"}
LINKING_PAGES = (
    DOCS / "security" / "key-management.md",
    DOCS / "compliance" / "breach-notification-runbook.md",
    DOCS / "structured" / "local-journey-store.md",
)

ENV_VAR_RE = re.compile(r"\b(?:OPENMED|HF|TRANSFORMERS)_[A-Z0-9_]*[A-Z0-9]\b")
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
FENCE_RE = re.compile(r"^[ \t]*```([a-z]*)\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL)
MODULE_PATH_RE = re.compile(r"^openmed(?:\.[a-z_][a-z0-9_]*)+$")
IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
ROUTE_RE = re.compile(r"^(GET|POST|DELETE) (/[A-Za-z0-9_./{}-]*)$")
IMPORT_RE = re.compile(
    r"from\s+(openmed(?:\.[a-z_][a-z0-9_]*)*)\s+import\s+(\([^)]*\)|[^\n;\"]+)"
)
HELM_RE = re.compile(r"Helm `([A-Za-z0-9_.]+)`")
COMPOSE_RE = re.compile(r"Compose (service|volume) `([a-z0-9-]+)`")
ISSUE_RE = re.compile(r"#(\d{3,5})\b")
REASON_TABLE_ROW_RE = re.compile(r"^\| `([a-z][a-z0-9_]*)` \|", re.MULTILINE)

FORBIDDEN_CODE_PATTERNS = (
    re.compile(r"\bprintenv\b"),
    re.compile(r"(^|\|)\s*env\s*($|\|)", re.MULTILINE),
    re.compile(r"\becho\s+\"?\$"),
    re.compile(r"\bcat\s+"),
    re.compile(r"json\.tool"),
    re.compile(r"postgres(?:ql)?://"),
    re.compile(r"\bset\s+-x\b"),
    re.compile(r"(?i)x-api-key\s*:"),
    re.compile(r"(?i)authorization\s*:"),
    re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    re.compile(r"\b[0-9a-f]{40,}\b"),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fenced_blocks(text: str) -> Iterator[tuple[str, str]]:
    for match in FENCE_RE.finditer(text):
        yield match.group(1), match.group(2)


def _prose(text: str) -> str:
    return FENCE_RE.sub("", text)


def _inline_code(text: str) -> list[str]:
    return [token.strip() for token in INLINE_CODE_RE.findall(_prose(text))]


@lru_cache(maxsize=1)
def _source_text() -> str:
    parts: list[str] = []
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


@lru_cache(maxsize=1)
def _source_words() -> frozenset[str]:
    return frozenset(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", _source_text()))


def _module_file(module: str) -> Path | None:
    relative = Path(*module.split("."))
    for candidate in (ROOT / f"{relative}.py", ROOT / relative / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=None)
def _module_names(module: str) -> frozenset[str]:
    path = _module_file(module)
    if path is None:
        return frozenset()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name for alias in node.names)
    return frozenset(names)


def _class_members(module: str, class_name: str) -> frozenset[str]:
    path = _module_file(module)
    if path is None:
        return frozenset()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return frozenset(
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    return frozenset()


def missing_env_vars(text: str) -> set[str]:
    """Return cited environment variables that no source or deploy file uses."""
    source = _source_text()
    return {name for name in set(ENV_VAR_RE.findall(text)) if name not in source}


def missing_module_paths(text: str) -> set[str]:
    """Return cited ``openmed.*`` paths that resolve to no module or symbol."""
    missing: set[str] = set()
    for token in _inline_code(text):
        if not MODULE_PATH_RE.fullmatch(token):
            continue
        if _module_file(token) is not None:
            continue
        module, _, name = token.rpartition(".")
        if _module_file(module) is None or name not in _module_names(module):
            missing.add(token)
    return missing


def missing_imports(text: str) -> set[str]:
    """Return ``from openmed... import name`` targets that do not exist."""
    missing: set[str] = set()
    for _language, body in _fenced_blocks(text):
        for module, raw_names in IMPORT_RE.findall(body):
            names = raw_names.strip().strip("()").replace("\n", " ")
            for item in names.split(","):
                name = item.strip().split(" as ")[0].strip()
                if not name:
                    continue
                if name not in _module_names(module):
                    missing.add(f"{module}.{name}")
    return missing


def missing_identifiers(text: str) -> set[str]:
    """Return snake_case inline codes (reason codes, fields) absent from source."""
    words = _source_words()
    return {
        token
        for token in _inline_code(text)
        if IDENTIFIER_RE.fullmatch(token) and token not in words
    }


def reason_codes(text: str) -> list[str]:
    """Return the codes listed in the reason-code table."""
    section = text.split("## Reason codes", 1)[1].split("\n## ", 1)[0]
    return REASON_TABLE_ROW_RE.findall(section)


def missing_reason_codes(codes: Iterable[str]) -> set[str]:
    """Return reason codes that never occur as a string literal in source."""
    source = _source_text()
    return {code for code in codes if f'"{code}"' not in source}


def _route_policies() -> set[tuple[str, str]]:
    pytest.importorskip("fastapi")
    from openmed.service.auth import ROUTE_POLICIES

    return set(ROUTE_POLICIES)


def missing_routes(text: str) -> set[str]:
    """Return ``METHOD /path`` citations that the service does not declare."""
    routes = _route_policies()
    missing: set[str] = set()
    for token in _inline_code(text):
        match = ROUTE_RE.fullmatch(token)
        if match and (match.group(1), match.group(2)) not in routes:
            missing.add(token)
    return missing


def _helm_values() -> dict:
    return yaml.safe_load(_read(HELM_VALUES))


def missing_helm_values(text: str) -> set[str]:
    """Return ``Helm `a.b``` citations absent from the chart values file."""
    values = _helm_values()
    missing: set[str] = set()
    for dotted in HELM_RE.findall(text):
        node: object = values
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                missing.add(dotted)
                break
            node = node[part]
    return missing


def missing_compose_objects(text: str) -> set[str]:
    """Return Compose service or volume citations absent from the profile."""
    compose = yaml.safe_load(_read(COMPOSE))
    missing: set[str] = set()
    for kind, name in COMPOSE_RE.findall(text):
        if name not in (compose.get(f"{kind}s") or {}):
            missing.add(f"{kind}:{name}")
    return missing


def _command_lines(body: str) -> Iterator[str]:
    joined = body.replace("\\\n", " ")
    in_heredoc: str | None = None
    for raw_line in joined.splitlines():
        line = raw_line.strip()
        if in_heredoc is not None:
            if line == in_heredoc:
                in_heredoc = None
            continue
        heredoc = re.search(r"<<-?'?([A-Z]+)'?", line)
        if heredoc:
            in_heredoc = heredoc.group(1)
        if line and not line.startswith("#"):
            yield line


def _strip_assignments(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens) and re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", tokens[index]):
        index += 1
    return tokens[index:]


@lru_cache(maxsize=None)
def _parser_for(program: str) -> argparse.ArgumentParser:
    if program in {"openmed", "openmed.cli"}:
        from openmed.cli.main import build_parser

        return build_parser()
    if program in {"openmed-journey", "openmed.deploy.journey"}:
        from openmed.deploy.journey import _parser

        return _parser()
    raise KeyError(program)


def _subparser_choices(
    parser: argparse.ArgumentParser,
) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _validate_cli(program: str, arguments: list[str]) -> bool:
    try:
        parser = _parser_for(program)
    except KeyError:
        return False
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token.startswith("--"):
            option = token.split("=", 1)[0]
            action = parser._option_string_actions.get(option)
            if action is None:
                return False
            if action.nargs != 0 and "=" not in token:
                index += 1
        else:
            choices = _subparser_choices(parser)
            if token in choices:
                parser = choices[token]
            elif choices and not token.startswith("<"):
                return False
        index += 1
    return True


def cited_cli_commands(text: str) -> list[tuple[str, list[str]]]:
    """Return OpenMed CLI invocations from shell blocks."""
    commands: list[tuple[str, list[str]]] = []
    for language, body in _fenced_blocks(text):
        if language not in {"bash", "sh", "console"}:
            continue
        for line in _command_lines(body):
            tokens = _strip_assignments(shlex.split(line, comments=True))
            if not tokens:
                continue
            if tokens[0] in {"openmed", "openmed-journey"}:
                commands.append((tokens[0], tokens[1:]))
                continue
            for index, token in enumerate(tokens[:-1]):
                if (
                    re.fullmatch(r"python3?", token)
                    and tokens[index + 1] == "-m"
                    and index + 2 < len(tokens)
                    and tokens[index + 2].startswith("openmed")
                ):
                    commands.append((tokens[index + 2], tokens[index + 3 :]))
                    break
    return commands


def invalid_cli_commands(text: str) -> set[str]:
    """Return cited CLI invocations that the real argument parsers reject."""
    invalid: set[str] = set()
    for program, arguments in cited_cli_commands(text):
        if program == "openmed.service.logging":
            if _module_file(program) is None:
                invalid.add(program)
            continue
        if not _validate_cli(program, arguments):
            invalid.add(" ".join([program, *arguments]))
    return invalid


def unsafe_code_lines(text: str) -> list[str]:
    """Return code-block lines that could leak secrets, payloads or DSNs."""
    unsafe: list[str] = []
    for _language, body in _fenced_blocks(text):
        for line in body.splitlines():
            if any(pattern.search(line) for pattern in FORBIDDEN_CODE_PATTERNS):
                unsafe.append(line.strip())
    return unsafe


def unmarked_issue_references(text: str) -> list[str]:
    """Return lines that cite an issue without marking the capability pending."""
    return [
        line
        for line in text.splitlines()
        if ISSUE_RE.search(line) and "pending" not in line.lower()
    ]


def _block_containing(path: Path, marker: str) -> str:
    for _language, body in _fenced_blocks(_read(path)):
        if marker in body:
            return textwrap.dedent(body)
    raise AssertionError(f"no code block containing {marker!r}")


def _heredoc_python(block: str) -> str:
    match = re.search(r"<<'PY'\n(.*?)^PY$", block, re.MULTILINE | re.DOTALL)
    assert match, "runbook snippet must be a quoted PY heredoc"
    return match.group(1)


def _run_snippet(code: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), **env},
        timeout=120,
    )


# --- publication and structure -------------------------------------------------


def test_runbooks_exist_and_are_navigated_and_published() -> None:
    nav = _read(MKDOCS)
    publication = _read(PUBLICATION)
    for path in RUNBOOKS:
        assert path.is_file(), path
        relative = path.relative_to(DOCS).as_posix()
        assert f": {relative}" in nav
        assert f"    - {relative}" in publication


def test_overview_links_every_scenario_page_and_is_linked_from_related_docs() -> None:
    overview = _read(OVERVIEW)
    for name in SCENARIO_PAGES:
        assert f"]({name}" in overview
    for page in LINKING_PAGES:
        assert "operations/runbooks/service-runbooks.md" in _read(page), page


def test_each_scenario_has_every_required_section() -> None:
    for name, scenarios in SCENARIO_PAGES.items():
        text = _read(RUNBOOK_DIR / name)
        for heading in REQUIRED_SCENARIO_SECTIONS:
            assert text.count(f"{heading}\n") == scenarios, (name, heading)
        if name.endswith("compromise.md"):
            assert text.count("### Rotation\n") == scenarios, name


def test_pending_capabilities_are_marked_and_tabulated() -> None:
    overview = _read(OVERVIEW)
    pending = overview.split("## Pending capabilities", 1)[1].split("\n## ", 1)[0]
    assert set(ISSUE_RE.findall(pending)) == PENDING_ISSUES
    for path in RUNBOOKS:
        assert unmarked_issue_references(_read(path)) == [], path


# --- source-of-truth checks ---------------------------------------------------


def test_cited_environment_variables_exist_in_source() -> None:
    cited: set[str] = set()
    for path in RUNBOOKS:
        text = _read(path)
        cited.update(ENV_VAR_RE.findall(text))
        assert missing_env_vars(text) == set(), path
    assert {
        "OPENMED_SERVICE_AUTH_API_KEYS",
        "OPENMED_SERVICE_AUTH_JWKS_FILE",
        "OPENMED_SERVICE_MTLS_CA_BUNDLE",
        "OPENMED_SERVICE_MTLS_PRINCIPALS",
        "OPENMED_SERVICE_JOBS_STORE_PATH",
        "OPENMED_JOURNEY_ARTIFACT_ROOT",
        "OPENMED_JOURNEY_POSTGRES_DSN",
    } <= cited


def test_cited_module_paths_and_imports_exist() -> None:
    for path in RUNBOOKS:
        text = _read(path)
        assert missing_module_paths(text) == set(), path
        assert missing_imports(text) == set(), path


def test_cited_identifiers_and_reason_codes_exist_in_source() -> None:
    codes = reason_codes(_read(OVERVIEW))
    assert len(codes) == len(set(codes))
    assert {
        "invalid_credentials",
        "mtls_certificate_invalid",
        "artifact_write_failed",
        "schema_unsupported",
        "payload_hash_mismatch",
    } <= set(codes)
    assert missing_reason_codes(codes) == set()
    for path in RUNBOOKS:
        assert missing_identifiers(_read(path)) == set(), path


def test_cited_routes_are_declared_service_routes() -> None:
    for path in RUNBOOKS:
        assert missing_routes(_read(path)) == set(), path


def test_cited_helm_values_and_compose_objects_exist() -> None:
    helm = compose = 0
    for path in RUNBOOKS:
        text = _read(path)
        helm += len(HELM_RE.findall(text))
        compose += len(COMPOSE_RE.findall(text))
        assert missing_helm_values(text) == set(), path
        assert missing_compose_objects(text) == set(), path
    assert helm and compose


def test_cited_cli_commands_parse_with_the_real_parsers() -> None:
    cited = []
    for path in RUNBOOKS:
        text = _read(path)
        cited.extend(cited_cli_commands(text))
        assert invalid_cli_commands(text) == set(), path
    programs = {program for program, _ in cited}
    assert {"openmed", "openmed.deploy.journey"} <= programs


def test_code_blocks_keep_credentials_payloads_and_dsns_out_of_shell_output() -> None:
    for path in RUNBOOKS:
        assert unsafe_code_lines(_read(path)) == [], path


def job_store_messages(text: str) -> list[str]:
    """Return the job-store startup messages quoted by a runbook."""
    return [token for token in _inline_code(text) if token.startswith("Job metadata")]


def missing_job_store_messages(messages: Iterable[str]) -> set[str]:
    """Return quoted startup messages that the job store never raises."""
    source = _read(ROOT / "openmed" / "service" / "jobs.py")
    return {message for message in messages if f'"{message}"' not in source}


def test_cited_job_store_startup_messages_match_source() -> None:
    messages = job_store_messages(_read(RUNBOOK_DIR / "store-corruption.md"))
    assert len(messages) == 4
    assert missing_job_store_messages(messages) == set()
    assert missing_job_store_messages(["Job metadata store was repaired"]) == {
        "Job metadata store was repaired"
    }


def test_store_methods_cited_by_runbooks_exist() -> None:
    for class_name in ("SQLiteJourneyStore", "PostgresJourneyStore"):
        module = (
            "openmed.structured.store.local"
            if class_name.startswith("SQLite")
            else "openmed.structured.store.postgres"
        )
        assert {"open", "integrity_check", "close"} <= _class_members(
            module, class_name
        )
    assert "connect" in _class_members(
        "openmed.structured.store.postgres", "PostgresJourneyStore"
    )


# --- executable snippets ------------------------------------------------------


def test_job_store_check_snippet_reports_fixed_outcomes_without_content(
    tmp_path: Path,
) -> None:
    code = _heredoc_python(
        _block_containing(RUNBOOK_DIR / "store-corruption.md", "LocalJobStore(")
    )
    sentinel = "SENTINEL-PATIENT-MRN-4417"
    store = tmp_path / "jobs.json"
    env = {"OPENMED_SERVICE_JOBS_STORE_PATH": str(store)}

    absent = _run_snippet(code, env)
    assert absent.returncode == 0, absent.stderr
    assert absent.stdout.strip() == "job store: absent"

    store.write_text('{"jobs": {"' + sentinel + '": ', encoding="utf-8")
    invalid = _run_snippet(code, env)
    assert invalid.returncode == 0, invalid.stderr
    assert invalid.stdout.strip() == "job store: unreadable or invalid"
    assert sentinel not in invalid.stdout + invalid.stderr
    assert str(tmp_path) not in invalid.stdout + invalid.stderr

    store.write_text(json.dumps({"jobs": {}}), encoding="utf-8")
    valid = _run_snippet(code, env)
    assert valid.returncode == 0, valid.stderr
    assert valid.stdout.strip() == "job store: loadable"


def test_sqlite_journey_check_snippet_works_on_a_copy_without_content(
    tmp_path: Path,
) -> None:
    from openmed.structured.store import SQLiteJourneyStore

    code = _heredoc_python(
        _block_containing(RUNBOOK_DIR / "store-corruption.md", "reader.backup(")
    )
    source = tmp_path / "journey.sqlite3"
    SQLiteJourneyStore(source).close()
    before = source.read_bytes()
    env = {
        "JOURNEY_SQLITE_PATH": str(source),
        "JOURNEY_CHECK_COPY": str(tmp_path / "check.sqlite3"),
    }

    healthy = _run_snippet(code, env)
    assert healthy.returncode == 0, healthy.stderr
    assert healthy.stdout.splitlines()[0] == "integrity: success"
    assert source.read_bytes() == before

    sentinel = b"SENTINEL-PATIENT-NAME-9031"
    damaged = tmp_path / "damaged.sqlite3"
    damaged.write_bytes(b"not a database " + sentinel * 64)
    env["JOURNEY_SQLITE_PATH"] = str(damaged)
    env["JOURNEY_CHECK_COPY"] = str(tmp_path / "check-damaged.sqlite3")
    failed = _run_snippet(code, env)
    assert failed.returncode == 0, failed.stderr
    assert failed.stdout.strip() in {"copy: failed", "open: failure store_open_failed"}
    assert sentinel.decode() not in failed.stdout + failed.stderr


def test_webhook_overlap_verifier_accepts_both_secrets_then_only_the_new_one() -> None:
    from openmed.service.signing import InvalidSignatureError, NonceCache, sign_request

    namespace: dict[str, object] = {}
    exec(  # noqa: S102 - executes the reviewed runbook snippet under test
        _block_containing(
            RUNBOOK_DIR / "credential-compromise.md", "def verify_with_overlap"
        ),
        namespace,
    )
    verify = namespace["verify_with_overlap"]
    body = b'{"event":"job.done"}'
    old_secret, new_secret = "synthetic-old-secret", "synthetic-new-secret"
    cache = NonceCache()

    old_headers = sign_request("POST", "/hook", body, secret=old_secret)
    assert verify(
        "POST",
        "/hook",
        body,
        old_headers,
        secrets=(new_secret, old_secret),
        nonce_cache=cache,
    )
    new_headers = sign_request("POST", "/hook", body, secret=new_secret)
    assert verify(
        "POST",
        "/hook",
        body,
        new_headers,
        secrets=(new_secret,),
        nonce_cache=cache,
    )
    retired = sign_request("POST", "/hook", body, secret=old_secret)
    with pytest.raises(InvalidSignatureError) as excinfo:
        verify(
            "POST",
            "/hook",
            body,
            retired,
            secrets=(new_secret,),
            nonce_cache=cache,
        )
    assert old_secret not in str(excinfo.value)
    assert new_secret not in str(excinfo.value)


# --- negative controls --------------------------------------------------------


def test_negative_controls_reject_invented_citations() -> None:
    fake = "\n".join(
        [
            "Set OPENMED_SERVICE_AUTH_API_KEYZ before restart.",
            "Use `openmed.service.auth.rotate_api_keys_now` or `openmed.nope`.",
            "Watch for `api_key_revoked_forever` and `POST /jobz`.",
            "Resize Helm `persistence.sizeLimit` and Compose volume `openmed-tmp`.",
            "",
            "## Reason codes",
            "",
            "| `credential_hot_reloaded` | REST | 401 |",
            "",
            "```bash",
            "openmed models cache purge <repo-id> --yes",
            "openmed models verify --everything",
            "python -m openmed.deploy.journey repair",
            "printenv OPENMED_SERVICE_AUTH_API_KEYS",
            "curl -H 'X-API-Key: <key>' http://127.0.0.1:8080/jobs",
            "```",
            "",
            "```python",
            "from openmed.service.auth import reload_credentials",
            "```",
            "",
            "Hot reload ships in #3699 today.",
        ]
    )

    assert missing_env_vars(fake) == {"OPENMED_SERVICE_AUTH_API_KEYZ"}
    assert missing_module_paths(fake) == {
        "openmed.service.auth.rotate_api_keys_now",
        "openmed.nope",
    }
    assert missing_imports(fake) == {"openmed.service.auth.reload_credentials"}
    assert missing_identifiers(fake) >= {"api_key_revoked_forever"}
    assert missing_reason_codes(reason_codes(fake)) == {"credential_hot_reloaded"}
    assert missing_routes(fake) == {"POST /jobz"}
    assert missing_helm_values(fake) == {"persistence.sizeLimit"}
    assert missing_compose_objects(fake) == {"volume:openmed-tmp"}
    assert invalid_cli_commands(fake) == {
        "openmed models cache purge <repo-id> --yes",
        "openmed models verify --everything",
        "openmed.deploy.journey repair",
    }
    assert len(unsafe_code_lines(fake)) == 2
    assert unmarked_issue_references(fake) == ["Hot reload ships in #3699 today."]


def test_positive_controls_accept_real_citations() -> None:
    real = "\n".join(
        [
            "Set OPENMED_SERVICE_AUTH_API_KEYS and read `invalid_credentials`.",
            "Use `openmed.service.auth.hash_api_key` on `GET /jobs/{job_id}`.",
            "Resize Helm `persistence.size` and Compose volume `openmed-cache`.",
            "",
            "```bash",
            "openmed models cache clear <repo-id> --yes",
            "openmed models verify --all",
            "python -m openmed.deploy.journey probe --worker-url <url>",
            "```",
        ]
    )
    assert missing_env_vars(real) == set()
    assert missing_module_paths(real) == set()
    assert missing_identifiers(real) == set()
    assert missing_routes(real) == set()
    assert missing_helm_values(real) == set()
    assert missing_compose_objects(real) == set()
    assert invalid_cli_commands(real) == set()
    assert unsafe_code_lines(real) == []
