"""PyInstaller entry point that silences library logging before OpenMed import."""

import logging

logging.disable(logging.CRITICAL)

from openmed.service.sidecar import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
