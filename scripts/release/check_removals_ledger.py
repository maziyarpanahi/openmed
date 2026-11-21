"""Validate the planned OpenMed 5.0 removals against the checked-in sources."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

LEDGER = Path("docs/migration/5.0-removals.json")


def _python_markers(path: Path) -> set[str]:
    markers: set[str] = set()

    def visit(node: ast.AST, parents: tuple[str, ...]) -> None:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            parents = (*parents, node.name)
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                name = ast.unparse(decorator.func).rsplit(".", 1)[-1]
                if name not in {"deprecated", "deprecated_parameter"} or not any(
                    keyword.arg == "remove_in"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "5.0"
                    for keyword in decorator.keywords
                ):
                    continue
                symbol = parents
                if name == "deprecated_parameter":
                    if (
                        not decorator.args
                        or not isinstance(decorator.args[0], ast.Constant)
                        or not isinstance(decorator.args[0].value, str)
                    ):
                        continue
                    symbol = (*parents, decorator.args[0].value)
                markers.add(".".join(symbol))
        for child in ast.iter_child_nodes(node):
            visit(child, parents)

    visit(ast.parse(path.read_bytes()), ())
    return markers


def _has_marker(path: Path, symbol: str, source: str) -> bool:
    name = re.escape(symbol)
    if path.suffix == ".kt":
        # Keep the annotation attached to this declaration. Its arguments can
        # contain nested ReplaceWith calls and span several lines.
        return bool(
            re.search(
                rf"@Deprecated\s*\((?:(?!\b(?:fun|class|val|var)\b|@Deprecated)[\s\S])*?\)\s*"
                rf"(?:(?:public|internal|suspend|override|enum|data|open)\s+)*(?:fun|class|val|var)\s+{name}\b",
                source,
            )
        )
    if path.suffix in {".ts", ".js", ".swift"}:
        if path.suffix == ".swift":
            return bool(
                re.search(
                    rf"@available\([^)]*\bdeprecated\b[^)]*\)\s*(?:(?:public|open|static|final|override)\s+)*(?:class|struct|enum|func|var|let)\s+{name}\b",
                    source,
                )
            )
        return bool(
            re.search(
                rf"/\*\*(?:(?!\*/)[\s\S])*?@deprecated\b(?:(?!\*/)[\s\S])*?\*/\s*"
                rf"(?:(?:export|declare|class|function|interface|type|const|readonly)\s+)*{name}\b",
                source,
            )
        )
    if path.suffix == ".go":
        return bool(
            re.search(
                rf"(?m)^// Deprecated:[^\n]*(?:\n//[^\n]*)*\n(?:type|func|const|var) {name}\b",
                source,
            )
        )
    if path.name == "openapi.json":
        method, route = symbol.split(" ", 1)
        operation = json.loads(source)["paths"].get(route, {}).get(method.lower(), {})
        return any(
            key.lower() == "deprecation"
            for status, response in operation.get("responses", {}).items()
            if str(status).startswith("2")
            for key in response.get("headers", {})
        )
    if path.name.endswith(".schema.json"):
        return (
            json.loads(source).get("properties", {}).get(symbol, {}).get("deprecated")
            is True
        )
    return False


def validate(root: Path) -> list[str]:
    """Return all missing markers, invalid entries and unlisted Python removals."""
    root = root.resolve()
    rows = json.loads((root / LEDGER).read_text())
    if not isinstance(rows, list) or not rows:
        return ["invalid removals ledger"]
    issues = []
    ids = set()
    listed = set()
    python = {
        p.relative_to(root).as_posix(): marks
        for p in (root / "openmed").rglob("*.py")
        if (marks := _python_markers(p))
    }
    for row in rows:
        if not isinstance(row, dict) or not all(
            row.get(k)
            for k in (
                "id",
                "surfaces",
                "symbols",
                "replacement",
                "status",
                "migration_anchor",
            )
        ):
            issues.append("missing ledger fields")
            continue
        key = row["id"]
        if key in ids or row["status"] not in {"planned", "deprecated"}:
            issues.append(f"{key}: duplicate id or invalid status")
        ids.add(key)
        if not isinstance(row["symbols"], dict):
            issues.append(f"{key}: invalid symbols")
            continue
        for filename, symbols in row["symbols"].items():
            path = (root / filename).resolve()
            if (
                not path.is_relative_to(root)
                or not path.is_file()
                or not isinstance(symbols, list)
                or not symbols
            ):
                issues.append(f"{key}: invalid source or symbols")
                continue
            source = path.read_text()
            for symbol in symbols:
                listed.add((filename, symbol))
                if row["status"] == "deprecated":
                    found = (
                        symbol in python.get(filename, set())
                        if path.suffix == ".py"
                        else _has_marker(path, symbol, source)
                    )
                    if not found:
                        issues.append(
                            f"{key}: missing deprecation marker for {filename}#{symbol}"
                        )
    for filename, symbols in python.items():
        for symbol in sorted(symbols):
            if (filename, symbol) not in listed:
                issues.append(f"unlisted 5.0 removal: {filename}#{symbol}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    issues = validate(parser.parse_args().root)
    for issue in issues:
        print(issue)
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
