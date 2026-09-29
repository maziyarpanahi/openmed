"""Clinical brief command with separate explicit protected/audit destinations."""

import argparse
import importlib
import json
import os
from pathlib import Path

from openmed.service.brief import BRIEF_PROFILES, LOCAL_BRIEF_MODELS, brief_response

from ._output import CliError, add_json_flag, emit


def add_brief_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the guarded, local-only clinical brief command."""
    parser = subparsers.add_parser(
        "brief", help="Build a guarded local clinical brief."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", choices=LOCAL_BRIEF_MODELS, default="mlx")
    parser.add_argument("--profile", choices=BRIEF_PROFILES, default="bhc")
    parser.add_argument("--review-id")
    parser.add_argument(
        "--context-factory",
        help="Trusted installed module:function returning an application review provider.",
    )
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--review-output", required=True, type=Path)
    add_json_flag(parser)
    parser.set_defaults(handler=handle_brief)


def handle_brief(args: argparse.Namespace, *, context_provider=None) -> int:
    """Write two new private files; never overwrite an existing destination."""
    failed = False
    created = []
    try:
        factory = getattr(args, "context_factory", None)
        if factory is not None:
            module_name, function_name = factory.split(":")
            if (
                not all(part.isidentifier() for part in module_name.split("."))
                or not function_name.isidentifier()
            ):
                raise ValueError("invalid context factory")
            if context_provider is not None:
                raise ValueError("ambiguous context provider")
            context_provider = getattr(
                importlib.import_module(module_name), function_name
            )()
            if not callable(context_provider):
                raise ValueError("invalid context provider")
        with args.path.open("rb") as handle:
            raw = handle.read(16385)
        if len(raw) > 16384:
            raise ValueError("input limit")
        response = brief_response(
            raw.decode("utf-8"),
            model=args.model,
            profile=args.profile,
            review_id=args.review_id,
            context_provider=context_provider,
        )
        summary = response.pop("summary")
        audit = json.dumps(response, sort_keys=True, indent=2) + "\n"
        # Reserve both paths before writing content; exclusive creation also
        # rejects symlinks and identical destinations without destroying data.
        from contextlib import ExitStack

        with ExitStack() as stack:
            handles = []
            for path in (args.summary_output, args.review_output):
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                created.append(path)
                handles.append(
                    stack.enter_context(os.fdopen(fd, "w", encoding="utf-8"))
                )
            handles[0].write(summary)
            handles[1].write(audit)
    except Exception:
        failed = True
    if failed:
        for path in created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise CliError("Brief request or output failed.", code="brief_failed")
    emit(
        args,
        {
            "status": response["status"],
            "refusal_reason": response["refusal_reason"],
            "summary_characters": response["summary_characters"],
            "digest": response["digest"],
        },
        human="Clinical brief outputs written; human review is required.",
    )
    return 1 if response["status"] == "refused" else 0
