from __future__ import annotations

import json

from openmed.cli import main_module
from openmed.eval.cost import load_cloud_prices


def test_cost_command_writes_json_and_markdown(tmp_path, capsys):
    perf_path = tmp_path / "perf.json"
    prices_path = tmp_path / "prices.json"
    output_dir = tmp_path / "reports"
    perf_path.write_text(
        json.dumps(
            {
                "model_name": "synthetic-model",
                "device": "cpu",
                "docs_per_second": 10.0,
                "metadata": {"chars_per_document": 1000},
                "hardware_cost_model": {
                    "purchase_price_usd": 1000.0,
                    "useful_life_hours": 10000.0,
                    "power_watts": 50.0,
                    "electricity_usd_per_kwh": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )
    prices_path.write_text(
        json.dumps(load_cloud_prices()),
        encoding="utf-8",
    )

    result = main_module.main(
        [
            "benchmark",
            "cost",
            "--perf",
            str(perf_path),
            "--prices",
            str(prices_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    payload = json.loads((output_dir / "cost-vs-cloud.json").read_text())
    assert payload["model_name"] == "synthetic-model"
    assert len(payload["comparisons"]) == 7
    assert (output_dir / "cost-vs-cloud.md").is_file()
    assert "Cost benchmark reports written" in capsys.readouterr().out
