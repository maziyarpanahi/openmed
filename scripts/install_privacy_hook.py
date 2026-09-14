#!/usr/bin/env python3
"""Install OpenMed's local deterministic privacy pre-push hook."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.guard.git_hook import ALLOWLIST_VERSION  # noqa: E402

HOOK_NAME = "pre-push"
ORIGINAL_HOOK_NAME = "pre-push.openmed-original"
HOOK_MARKER = f"# openmed-privacy-pre-push-hook:v{ALLOWLIST_VERSION}"


def _git_hooks_path(repo_root: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--git-path", "hooks"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        raise RuntimeError("git is unavailable") from None
    if completed.returncode != 0:
        raise RuntimeError("repository hooks directory could not be resolved")
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError("repository hooks directory could not be resolved")
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _hook_script(python_executable: str) -> str:
    quoted_python = shlex.quote(python_executable)
    return dedent(
        f"""\
        #!/bin/sh
        {HOOK_MARKER}
        # The scanner is local-only and receives Git's pre-push input on stdin.
        set -eu

        OPENMED_REPO_ROOT=$(git rev-parse --show-toplevel)
        OPENMED_HOOK_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
        OPENMED_PYTHON=${{OPENMED_PYTHON:-{quoted_python}}}
        OPENMED_INPUT=$(mktemp "${{TMPDIR:-/tmp}}/openmed-pre-push.XXXXXX")
        trap 'rm -f "$OPENMED_INPUT"' EXIT HUP INT TERM
        cat > "$OPENMED_INPUT"

        set +e
        "$OPENMED_PYTHON" -m openmed.guard.git_hook \\
          --repo "$OPENMED_REPO_ROOT" "$@" < "$OPENMED_INPUT"
        OPENMED_SCAN_STATUS=$?
        set -e
        if [ "$OPENMED_SCAN_STATUS" -ne 0 ]; then
          exit "$OPENMED_SCAN_STATUS"
        fi

        if [ -x "$OPENMED_HOOK_DIR/{ORIGINAL_HOOK_NAME}" ]; then
          "$OPENMED_HOOK_DIR/{ORIGINAL_HOOK_NAME}" "$@" < "$OPENMED_INPUT"
        fi
        """
    )


def install_hook(
    repo_root: str | Path,
    *,
    python_executable: str | None = None,
) -> Path:
    """Install or update the privacy hook while preserving an old hook."""

    root = Path(repo_root).resolve()
    hooks_dir = _git_hooks_path(root)
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / HOOK_NAME
    original_path = hooks_dir / ORIGINAL_HOOK_NAME

    if hook_path.is_symlink() or original_path.is_symlink():
        raise RuntimeError("a pre-push hook path is a symbolic link")

    preserve_existing = False
    if hook_path.exists():
        try:
            existing = hook_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise RuntimeError("existing pre-push hook could not be read") from None
        if HOOK_MARKER not in existing:
            if original_path.exists():
                raise RuntimeError("a preserved pre-push hook already exists")
            preserve_existing = True

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{HOOK_NAME}.openmed-",
        dir=hooks_dir,
    )
    temporary_path = Path(temporary_name)
    preserved = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(_hook_script(python_executable or sys.executable))
            stream.flush()
            os.fsync(stream.fileno())
            if hasattr(os, "fchmod"):
                os.fchmod(stream.fileno(), 0o755)
        if not hasattr(os, "fchmod"):
            os.chmod(temporary_path, 0o755)

        if preserve_existing:
            os.replace(hook_path, original_path)
            preserved = True
        try:
            os.replace(temporary_path, hook_path)
        except OSError:
            if preserved:
                try:
                    os.replace(original_path, hook_path)
                except OSError:
                    raise RuntimeError(
                        "privacy hook installation failed and rollback was incomplete"
                    ) from None
            raise RuntimeError("privacy hook could not be installed") from None
    except (OSError, UnicodeError):
        if preserved:
            try:
                os.replace(original_path, hook_path)
            except OSError:
                raise RuntimeError(
                    "privacy hook installation failed and rollback was incomplete"
                ) from None
        raise RuntimeError("privacy hook could not be installed") from None
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
    return hook_path


def main(argv: list[str] | None = None) -> int:
    """Install the hook and report only a high-level status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        help="Python executable used by the installed hook",
    )
    args = parser.parse_args(argv)
    try:
        install_hook(args.repo, python_executable=args.python_executable)
    except (OSError, RuntimeError, ValueError) as exc:
        del exc
        print("privacy hook installation failed", file=sys.stderr)
        return 1
    print("installed OpenMed privacy pre-push hook")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
