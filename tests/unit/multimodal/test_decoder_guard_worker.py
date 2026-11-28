"""The decoder guard's isolated worker must stay independent of the package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from openmed.core.errors import LegacyRootedError
from openmed.multimodal import decoder_guard

WORKER = str(Path(decoder_guard.__file__).resolve())


def test_worker_script_runs_isolated_without_importing_openmed() -> None:
    # The guard starts its worker as ``python -I <decoder_guard.py> --worker``.
    # Isolated mode drops the package path, so the script must not import
    # openmed: where the package is only on PYTHONPATH (the Nix dev shell),
    # every probe would otherwise fail before decoding.
    code = (
        "import runpy, sys\n"
        f"sys.argv = [{WORKER!r}, '--not-a-worker']\n"
        "try:\n"
        f"    runpy.run_path({WORKER!r}, run_name='__main__')\n"
        "except SystemExit as stop:\n"
        "    print(stop.code)\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'openmed'))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        capture_output=True,
        check=True,
        text=True,
    )

    isolation_exit = decoder_guard._EXIT[decoder_guard.DecodeStatus.ISOLATION]
    assert completed.stdout.split("\n")[:2] == [str(isolation_exit), "[]"]


def test_imported_guard_errors_keep_their_openmed_root() -> None:
    assert issubclass(decoder_guard.DecoderGuardError, LegacyRootedError)
    assert issubclass(decoder_guard.DecoderGuardError, ValueError)
