"""Offline summary and NLI commands with value-free console projections."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import importlib
import io
import json
import logging
import math
import os
import re
import stat
import sys
import warnings
from collections.abc import Iterator, Sequence
from typing import Any

from ._output import EXIT_ERROR, EXIT_USAGE, CliError, command_path, emit, emit_error

_NOTE_BYTES = 16_384
_CLAIMS_BYTES = 65_536
_CLAIM_BYTES = 4_096
_MAX_CLAIMS = 128
_SUMMARY_BYTES = 8_192
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NLI_FIELDS = frozenset(
    {"claim_index", "label", "score", "backend_id", "contradicted", "review_required"}
)


def add_clinical_review_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register local summarization and claim verification commands.

    Args:
        subparsers: Root CLI command registrar.
    """
    summary = subparsers.add_parser(
        "summarize", help="Write a guarded local summary and separate metadata."
    )
    summary.add_argument("path", help="Local UTF-8 note file.")
    summary.add_argument(
        "--model",
        default="mlx",
        help="Local alias: mlx or explicit extractive baseline.",
    )
    summary.add_argument("--mode", default="bhc", help="Supported summary mode: bhc.")
    summary.add_argument(
        "--summary-output",
        required=True,
        help="New private file for protected summary text.",
    )
    summary.add_argument(
        "--metadata-output",
        required=True,
        help="Separate new private file for value-free metadata.",
    )
    summary.set_defaults(handler=handle_summarize)
    group = subparsers.add_parser(
        "nli", help="Inspect local clinical claim-verification evidence."
    )
    children = group.add_subparsers(dest="nli_command", required=True)
    verify = children.add_parser(
        "verify", help="Verify bounded claims without printing their text."
    )
    verify.add_argument("--source", required=True, help="Local UTF-8 source file.")
    verify.add_argument(
        "--claims", required=True, help="Local JSON array of nonempty claim strings."
    )
    verify.add_argument(
        "--backend",
        default="local",
        help="local or explicit heuristic development baseline.",
    )
    verify.add_argument(
        "--backend-factory",
        help="Trusted installed module:function returning a caller-supplied local backend.",
    )
    verify.set_defaults(handler=handle_nli_verify)


def parse_clinical_review_args(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> argparse.Namespace:
    """Parse protected commands without echoing private arguments on failure.

    Args:
        parser: Fully registered root parser.
        argv: Explicit arguments or None for the process arguments.

    Returns:
        Normal parsed arguments, or a controlled usage-error handler.
    """
    values = list(sys.argv[1:] if argv is None else argv)
    index = 0
    while index < len(values):
        value = values[index]
        if value == "--config-path":
            index += 2
        elif value.startswith("--config-path=") or value == "--json":
            index += 1
        else:
            break
    if index >= len(values) or values[index] not in {"summarize", "nli"}:
        # Unknown global options may precede a protected command. Respect the
        # first recognized root command so other command behavior is unchanged.
        choices = next(
            (
                action.choices
                for action in parser._actions
                if isinstance(action, argparse._SubParsersAction)
            ),
            {},
        )
        root = next((value for value in values[index:] if value in choices), None)
        if root is None:
            # An omitted global option value can consume the command token.
            # Still suppress the resulting error instead of echoing its paths.
            root = next((value for value in values if value in choices), None)
            index = 0
        if root not in {"summarize", "nli"}:
            return parser.parse_args(values)
        index = values.index(root, index)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stderr(sink):
            try:
                return parser.parse_args(values)
            except SystemExit as error:
                if error.code != EXIT_USAGE:
                    raise
    path = "summarize" if values[index] == "summarize" else "nli verify"
    return argparse.Namespace(
        command_path=path, json_output="--json" in values, handler=_invalid_arguments
    )


def _invalid_arguments(args: argparse.Namespace) -> int:
    code = (
        "summary_arguments_invalid"
        if command_path(args) == "summarize"
        else "nli_arguments_invalid"
    )
    return emit_error(args, _error(code, usage=True))


def _error(code: str, *, usage: bool = False) -> CliError:
    return CliError(
        "Local clinical request could not complete.",
        code=code,
        exit_code=EXIT_USAGE if usage else EXIT_ERROR,
    )


def _read(path: str, limit: int, prefix: str) -> bytes:
    descriptor = None
    failure = None
    try:
        if type(path) is not str or "://" in path:
            raise _error(prefix + "_input_invalid", usage=True)
        expected = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(expected.st_mode):
            raise _error(prefix + "_input_not_regular", usage=True)
        descriptor = os.open(
            path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            raise _error(prefix + "_input_not_regular", usage=True)
        if info.st_size > limit:
            raise _error(prefix + "_input_too_large", usage=True)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise _error(prefix + "_input_too_large", usage=True)
        return data
    except CliError as error:
        failure = _error(error.code, usage=True)
    except (OSError, ValueError):
        failure = _error(prefix + "_input_unavailable", usage=True)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    raise failure


def _text(raw: bytes, prefix: str) -> str:
    try:
        value = raw.decode("utf-8")
        if value.strip():
            return value
    except UnicodeError:
        pass
    raise _error(prefix + "_input_invalid", usage=True)


def _claims(raw: bytes) -> list[str]:
    failure = False
    try:
        values = json.loads(raw.decode("utf-8"))
        if type(values) is not list or not 1 <= len(values) <= _MAX_CLAIMS:
            raise ValueError()
        for value in values:
            if (
                type(value) is not str
                or not value.strip()
                or len(value.encode("utf-8")) > _CLAIM_BYTES
            ):
                raise ValueError()
        return values
    except (ValueError, TypeError, UnicodeError, RecursionError):
        failure = True
    if failure:
        raise _error("nli_claims_invalid", usage=True)
    raise AssertionError("unreachable")


def _flush_native() -> None:
    try:
        flush = ctypes.CDLL(None).fflush
        flush.argtypes = [ctypes.c_void_p]
        flush.restype = ctypes.c_int
        flush(None)
    except (AttributeError, OSError):
        pass


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Discard Python/native console chatter during trusted local processing."""
    previous_logging = logging.root.manager.disable
    saved = []
    with open(os.devnull, "w", encoding="utf-8") as sink:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            _flush_native()
            for descriptor in (1, 2):
                saved.append((descriptor, os.dup(descriptor)))
                os.dup2(sink.fileno(), descriptor)
            logging.disable(sys.maxsize)
            with (
                warnings.catch_warnings(),
                contextlib.redirect_stdout(sink),
                contextlib.redirect_stderr(sink),
            ):
                warnings.simplefilter("ignore")
                yield
        finally:
            sink.flush()
            _flush_native()
            for descriptor, backup in reversed(saved):
                os.dup2(backup, descriptor)
                os.close(backup)
            logging.disable(previous_logging)


def _summary_metadata(result: Any) -> dict[str, Any]:
    from openmed.clinical.summarize import LeakageCheck, SummarizationResult

    if (
        type(result) is not SummarizationResult
        or result.mode != "bhc"
        or result.backend not in {"local-mlx", "deterministic-extractive"}
    ):
        raise _error("summary_result_invalid")
    if (
        type(result.summary) is not str
        or not result.summary.strip()
        or len(result.summary.encode("utf-8")) > _SUMMARY_BYTES
    ):
        raise _error("summary_result_invalid")
    check = result.leakage_check
    if type(check) is not LeakageCheck:
        raise _error("summary_result_invalid")
    if check.passed is not True:
        raise _error("summary_leakage_rejected")
    if (
        type(check.checked_token_count) is not int
        or not 0 <= check.checked_token_count <= _NOTE_BYTES
        or check.leaked_token_count != 0
        or check.leaked_token_hashes
    ):
        raise _error("summary_result_invalid")
    if type(result.template_digest) is not str or not _DIGEST.fullmatch(
        result.template_digest
    ):
        raise _error("summary_result_invalid")
    return {
        "mode": "bhc",
        "backend_id": result.backend,
        "template_digest": result.template_digest,
        "leakage_check": check.to_dict(),
        "summary_characters": len(result.summary),
        "human_review_required": True,
    }


def _write_summary_pair(paths: tuple[str, str], contents: tuple[bytes, bytes]) -> None:
    created = []
    failure = False
    try:
        with contextlib.ExitStack() as stack:
            streams = []
            for path in paths:
                if type(path) is not str or "://" in path:
                    raise ValueError()
                descriptor = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                info = os.fstat(descriptor)
                created.append((path, info.st_dev, info.st_ino))
                try:
                    stream = os.fdopen(descriptor, "wb")
                except Exception:
                    os.close(descriptor)
                    raise
                streams.append(stack.enter_context(stream))
            for stream, content in zip(streams, contents, strict=True):
                stream.write(content)
    except Exception:
        failure = True
    if failure:
        for path, device, inode in created:
            try:
                info = os.stat(path, follow_symlinks=False)
                if (info.st_dev, info.st_ino) == (device, inode):
                    os.unlink(path)
            except OSError:
                pass
        raise _error("summary_output_failed")


def handle_summarize(args: argparse.Namespace) -> int:
    """Run the existing guarded API and write separate exclusive destinations.

    Args:
        args: Parsed local alias, bounded input and explicit output destinations.

    Returns:
        Zero after a passing leakage check and successful writes.
    """
    if args.model not in {"extractive", "mlx"}:
        raise _error("summary_model_invalid", usage=True)
    if args.mode != "bhc":
        raise _error("summary_mode_unsupported", usage=True)
    text = _text(_read(args.path, _NOTE_BYTES, "summary"), "summary")
    failure = None
    with _quiet():
        from openmed.clinical.summarize import (
            SummarizationLeakageError,
            SummarizationOrderError,
            summarize,
        )
        from openmed.clinical.summarize_backends import LocalSummarizerError
        from openmed.core.capabilities import MissingOptionalDependencyError
        from openmed.core.offline import network_blocked_if_offline

        try:
            with network_blocked_if_offline(local_only=True):
                result = summarize(text, model=args.model, mode=args.mode)
            metadata = _summary_metadata(result)
        except SummarizationLeakageError:
            failure = _error("summary_leakage_rejected")
        except SummarizationOrderError:
            failure = _error("summary_deidentification_rejected")
        except (MissingOptionalDependencyError, LocalSummarizerError):
            failure = _error("summary_backend_unavailable")
        except CliError as error:
            failure = _error(error.code)
        except Exception:
            failure = _error("summary_failed")
    if failure is not None:
        raise failure
    record = io.StringIO()
    emit(
        argparse.Namespace(command_path="summarize", json_output=True),
        metadata,
        stream=record,
    )
    _write_summary_pair(
        (args.summary_output, args.metadata_output),
        (result.summary.encode("utf-8"), record.getvalue().encode("utf-8")),
    )
    return emit(
        args,
        metadata,
        human="Summary files written; qualified clinical review is required.",
    )


def _nli_backend(args: argparse.Namespace) -> Any:
    from openmed.clinical.nli import HeuristicNLIBackend
    from openmed.clinical.nli_backends import resolve_nli_backend

    if args.backend not in {"local", "heuristic"}:
        raise _error("nli_backend_invalid", usage=True)
    if args.backend_factory is None:
        return resolve_nli_backend(args.backend)
    if args.backend != "local":
        raise _error("nli_backend_invalid", usage=True)
    parts = args.backend_factory.split(":")
    if (
        len(parts) != 2
        or not all(part.isidentifier() for part in parts[0].split("."))
        or not parts[1].isidentifier()
    ):
        raise _error("nli_backend_factory_invalid", usage=True)
    backend = getattr(importlib.import_module(parts[0]), parts[1])()
    if (
        isinstance(backend, HeuristicNLIBackend)
        or getattr(backend, "backend_id", None) == "heuristic"
    ):
        raise _error("nli_backend_invalid", usage=True)
    if not callable(backend) and not callable(getattr(backend, "predict", None)):
        raise _error("nli_backend_factory_invalid", usage=True)
    return backend


def _nli_projection(
    values: Any, count: int, *, caller_supplied: bool
) -> list[dict[str, Any]]:
    from openmed.clinical.nli import NLI_LABELS

    if type(values) is not list or len(values) != count:
        raise _error("nli_result_invalid")
    results = []
    for index, value in enumerate(values):
        if type(value) is not dict or value.keys() != _NLI_FIELDS:
            raise _error("nli_result_invalid")
        if (
            type(value["claim_index"]) is not int
            or value["claim_index"] != index
            or value["label"] not in NLI_LABELS
        ):
            raise _error("nli_result_invalid")
        score = value["score"]
        if (
            type(score) not in (int, float)
            or not 0 <= score <= 1
            or not math.isfinite(score)
        ):
            raise _error("nli_result_invalid")
        if (
            type(value["contradicted"]) is not bool
            or type(value["review_required"]) is not bool
            or value["contradicted"] != (value["label"] == "contradiction")
            or value["review_required"] != (value["label"] == "abstention")
        ):
            raise _error("nli_result_invalid")
        backend_id = "caller-supplied-local" if caller_supplied else value["backend_id"]
        if backend_id not in {"heuristic", "local-encoder", "caller-supplied-local"}:
            raise _error("nli_result_invalid")
        results.append({**value, "backend_id": backend_id})
    return results


def handle_nli_verify(args: argparse.Namespace) -> int:
    """Verify explicit bounded claims using an offline backend and safe metadata.

    Args:
        args: Parsed source/claim paths and explicit local backend options.

    Returns:
        Zero for all entailments, one when any claim is not entailed.
    """
    if args.backend not in {"local", "heuristic"}:
        raise _error("nli_backend_invalid", usage=True)
    source = _text(_read(args.source, _NOTE_BYTES, "nli"), "nli")
    claims = _claims(_read(args.claims, _CLAIMS_BYTES, "nli"))
    failure = None
    with _quiet():
        from openmed.clinical.nli import verify
        from openmed.clinical.nli_backends import LocalNLIError
        from openmed.core.capabilities import MissingOptionalDependencyError
        from openmed.core.offline import network_blocked_if_offline

        try:
            with network_blocked_if_offline(local_only=True):
                backend = _nli_backend(args)
                values = verify(claims, source, backend=backend)
            results = _nli_projection(
                values, len(claims), caller_supplied=args.backend_factory is not None
            )
        except CliError as error:
            failure = _error(error.code, usage=error.exit_code == EXIT_USAGE)
        except (
            LocalNLIError,
            MissingOptionalDependencyError,
            ImportError,
            AttributeError,
        ):
            failure = _error("nli_backend_unavailable")
        except Exception:
            failure = _error("nli_failed")
    if failure is not None:
        raise failure
    data = {"claims": results}
    emit(args, data, human=json.dumps(data, sort_keys=True))
    return int(any(result["label"] != "entailment" for result in results))


__all__ = [
    "add_clinical_review_commands",
    "handle_summarize",
    "handle_nli_verify",
    "parse_clinical_review_args",
]
