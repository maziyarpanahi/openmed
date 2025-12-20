"""Draft Typer-powered CLI for OpenMed.

This supplements the existing argparse CLI without breaking it. It is
optional: install extras `pip install .[cli]` or add Typer/Rich to your
environment, then run:

    python -m openmed.cli.typer_app --help
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List

try:  # soft dependency to avoid import errors in base installs
    import typer
    from rich import print as rprint
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover - optional surface
    typer = None
    Console = None
    Table = None
    rprint = print

from openmed import analyze_text, list_models, get_model_max_length
from openmed.core.config import (
    OpenMedConfig,
    get_config,
    set_config,
    resolve_config_path,
    load_config_from_file,
    save_config_to_file,
)
from openmed.ner import (
    build_index,
    write_index,
    infer as zs_infer,
    NerRequest,
    ensure_gliner_available,
    ensure_gliner2_available,
)


def _ensure_typer():
    if typer is None:
        raise RuntimeError(
            "Typer/Rich not installed. Install with `pip install .[cli]` or "
            "`pip install typer rich`."
        )


def _load_config(config_path: Optional[Path]) -> OpenMedConfig:
    if config_path:
        try:
            cfg = load_config_from_file(config_path)
            set_config(cfg)
            return cfg
        except FileNotFoundError:
            pass
    return get_config()


def _echo_json(payload: object) -> None:
    rprint(json.dumps(payload, indent=2, ensure_ascii=False))


def _render_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    if Console is None or Table is None:
        for row in rows:
            rprint(" | ".join(row))
        return
    table = Table(title=title)
    for head in headers:
        table.add_column(head)
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    Console().print(table)


def main() -> None:
    _ensure_typer()

    app = typer.Typer(help="OpenMed Typer CLI (draft).")
    models_app = typer.Typer(help="Model discovery commands.")
    cli_app = typer.Typer(help="Config utilities.")
    zero_app = typer.Typer(help="Zero-shot (GLiNER/GLiNER2) utilities.")

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------
    @app.command("analyze")
    def analyze(
        text: Optional[str] = typer.Option(
            None, "--text", "-t", help="Inline text to analyse."
        ),
        input_file: Optional[Path] = typer.Option(
            None, "--input-file", "-f", help="Path to a text file."
        ),
        model: str = typer.Option(
            "disease_detection_superclinical",
            "--model",
            "-m",
            help="Registry key or HF model id.",
        ),
        output_format: str = typer.Option(
            "dict", "--format", "-o", help="dict|json|html|csv"
        ),
        confidence_threshold: Optional[float] = typer.Option(
            None, "--threshold", "-c", help="Minimum confidence to keep entities."
        ),
        group_entities: bool = typer.Option(
            False, "--group", help="Merge adjacent spans of the same label."
        ),
        no_confidence: bool = typer.Option(
            False, "--no-confidence", help="Exclude confidence values."
        ),
        sentence_detection: bool = typer.Option(
            True, "--sentence-detection/--no-sentence-detection", help="Toggle sentence splitting."
        ),
        config_path: Optional[Path] = typer.Option(
            None, "--config-path", help="Override config path."
        ),
    ):
        """Analyse text with an OpenMed model and pretty-print the result."""
        cfg = _load_config(config_path)
        if text is None and input_file is None:
            raise typer.BadParameter("Provide --text or --input-file.")
        payload = text
        if input_file:
            payload = input_file.read_text(encoding="utf-8")

        result = analyze_text(
            payload,
            model_name=model,
            output_format=output_format,
            confidence_threshold=confidence_threshold,
            group_entities=group_entities,
            include_confidence=not no_confidence,
            sentence_detection=sentence_detection,
            config=cfg,
        )

        if hasattr(result, "to_dict"):
            _echo_json(result.to_dict())
        else:
            rprint(result)

    # ------------------------------------------------------------------
    # models
    # ------------------------------------------------------------------
    @models_app.command("list")
    def list_available_models(
        include_remote: bool = typer.Option(
            False, "--include-remote", help="Query Hugging Face Hub."
        ),
        config_path: Optional[Path] = typer.Option(
            None, "--config-path", help="Override config path."
        ),
    ):
        cfg = _load_config(config_path)
        models = list_models(include_registry=True, include_remote=include_remote, config=cfg)
        rows = [[m] for m in models]
        _render_table("Models", ["model_id"], rows)

    @models_app.command("info")
    def model_info(
        model_key: str = typer.Argument(..., help="Registry key or HF id."),
        config_path: Optional[Path] = typer.Option(
            None, "--config-path", help="Override config path."
        ),
    ):
        cfg = _load_config(config_path)
        max_len = get_model_max_length(model_key, config=cfg)
        _echo_json({"model_key": model_key, "max_length": max_len})

    app.add_typer(models_app, name="models")

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------
    @cli_app.command("show")
    def config_show(config_path: Optional[Path] = typer.Option(None, "--config-path")):
        cfg_path = resolve_config_path(config_path)
        try:
            cfg = load_config_from_file(cfg_path)
            source = str(cfg_path)
        except FileNotFoundError:
            cfg = get_config()
            source = "defaults (not yet saved)"
        payload = cfg.to_dict()
        payload["_source"] = source
        _echo_json(payload)

    @cli_app.command("set")
    def config_set(
        key: str,
        value: Optional[str] = typer.Argument(None),
        unset: bool = typer.Option(False, "--unset", help="Clear the value."),
        config_path: Optional[Path] = typer.Option(None, "--config-path"),
    ):
        cfg_path = resolve_config_path(config_path)
        try:
            cfg = load_config_from_file(cfg_path)
        except FileNotFoundError:
            cfg = get_config()
        cfg_dict = cfg.to_dict()
        if key not in cfg_dict:
            raise typer.BadParameter(f"Unknown key '{key}'. Valid: {', '.join(cfg_dict.keys())}")
        cfg_dict[key] = None if unset else value
        new_cfg = OpenMedConfig.from_dict(cfg_dict)
        set_config(new_cfg)
        save_config_to_file(new_cfg, cfg_path)
        rprint(f"[green]Updated {key} -> {cfg_dict[key]} in {cfg_path}[/green]")

    app.add_typer(cli_app, name="config")

    # ------------------------------------------------------------------
    # zero-shot (GLiNER/GLiNER2)
    # ------------------------------------------------------------------
    @zero_app.command("deps")
    def zero_deps():
        messages = []
        try:
            ensure_gliner_available()
            messages.append("GLiNER v1: ok")
        except Exception as exc:  # pragma: no cover
            messages.append(f"GLiNER v1: missing ({exc})")
        try:
            ensure_gliner2_available()
            messages.append("GLiNER v2: ok")
        except Exception as exc:  # pragma: no cover
            messages.append(f"GLiNER v2: missing ({exc})")
        for line in messages:
            rprint(line)

    @zero_app.command("index")
    def zero_index(
        models_dir: Path = typer.Argument(..., help="Root directory containing zero-shot models."),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to write index.json"),
        pretty: bool = typer.Option(True, "--pretty/--compact", help="Pretty-print JSON."),
    ):
        index = build_index(models_dir)
        out_path = output or (models_dir / "index.json")
        write_index(index, out_path, pretty=pretty)
        rprint(f"[green]Index written to {out_path}[/green]")

    @zero_app.command("infer")
    def zero_infer(
        text: str = typer.Argument(..., help="Input text for zero-shot NER."),
        model_id: str = typer.Option(..., "--model-id", "-m", help="Model id from index."),
        labels: Optional[str] = typer.Option(
            None, "--labels", "-l", help="Comma-separated label list (optional)."
        ),
        domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Domain hint."),
        threshold: float = typer.Option(0.5, "--threshold", "-c", help="Score threshold."),
        index_path: Optional[Path] = typer.Option(
            None, "--index-path", "-i", help="Path to index.json (defaults to models/index.json)."
        ),
    ):
        label_list = [label.strip() for label in labels.split(",")] if labels else None
        request = NerRequest(
            model_id=model_id,
            text=text,
            labels=label_list,
            domain=domain,
            threshold=threshold,
        )
        response = zs_infer(request, index_path=index_path)
        _echo_json(response.to_dict())

    app.add_typer(zero_app, name="zero")

    app()


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except RuntimeError as exc:
        rprint(f"[red]{exc}[/red]")
