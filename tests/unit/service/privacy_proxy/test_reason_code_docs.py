"""Keep the documented privacy-proxy reason codes complete."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PROXY_DIR = REPO_ROOT / "openmed" / "service" / "privacy_proxy"
PROXY_MODULES = ("app.py", "inbound.py", "outbound.py")
GATEWAY_MODULE = REPO_ROOT / "openmed" / "service" / "privacy_gateway.py"
DOCS_PAGE = REPO_ROOT / "docs" / "service" / "privacy-proxy.md"
TABLE_START = "<!-- privacy-proxy-reason-codes:start -->"
TABLE_END = "<!-- privacy-proxy-reason-codes:end -->"
_ROW_CODE = re.compile(r"^\| `([a-z0-9_]+)` \|", re.MULTILINE)


def _str_constant(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def collect_reason_codes(source: str) -> set[str]:
    """Return reason-code literals from class attributes, keywords and getattr."""

    codes: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "reason_code"
                for target in node.targets
            ):
                codes.add(_str_constant(node.value) or "")
        elif isinstance(node, ast.keyword) and node.arg == "reason_code":
            codes.add(_str_constant(node.value) or "")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and _str_constant(node.args[1]) == "reason_code"
        ):
            codes.add(_str_constant(node.args[2]) or "")
    codes.discard("")
    return codes


def _module_codes(*paths: Path) -> set[str]:
    codes: set[str] = set()
    for path in paths:
        codes |= collect_reason_codes(path.read_text(encoding="utf-8"))
    return codes


def _proxy_codes() -> set[str]:
    return _module_codes(*(PROXY_DIR / name for name in PROXY_MODULES))


def _documented_table() -> str:
    text = DOCS_PAGE.read_text(encoding="utf-8")
    assert text.count(TABLE_START) == 1, f"missing {TABLE_START} marker"
    assert text.count(TABLE_END) == 1, f"missing {TABLE_END} marker"
    return text.split(TABLE_START, 1)[1].split(TABLE_END, 1)[0]


def _documented_codes() -> set[str]:
    return set(_ROW_CODE.findall(_documented_table()))


def _undocumented(codes: set[str], documented: set[str]) -> list[str]:
    return sorted(codes - documented)


def test_every_proxy_reason_code_is_documented() -> None:
    codes = _proxy_codes()
    missing = _undocumented(codes, _documented_codes())

    assert len(codes) > 20
    assert not missing, (
        "Undocumented privacy-proxy reason codes; add a row to "
        f"docs/service/privacy-proxy.md: {', '.join(missing)}"
    )


def test_new_reason_code_without_docs_is_reported_by_name() -> None:
    source = (
        "class NewError(Exception):\n"
        "    reason_code = 'synthetic_class_code'\n"
        "raise NewError(reason_code='synthetic_keyword_code')\n"
        "getattr(exc, 'reason_code', 'synthetic_default_code')\n"
        "def f(reason_code=None):\n"
        "    return reason_code\n"
    )

    missing = _undocumented(collect_reason_codes(source), _documented_codes())

    assert missing == [
        "synthetic_class_code",
        "synthetic_default_code",
        "synthetic_keyword_code",
    ]


def test_documented_rows_are_real_reason_codes() -> None:
    known = _proxy_codes() | _module_codes(GATEWAY_MODULE)

    assert sorted(_documented_codes() - known) == []


def test_reason_code_tables_contain_no_request_content() -> None:
    table = _documented_table()

    assert "OPENMED_PHI" not in table
    assert "{" not in table
    assert '"' not in table
