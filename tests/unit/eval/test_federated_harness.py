from __future__ import annotations

import hashlib
from pathlib import Path

from openmed.eval.harness import (
    BenchmarkFixture,
    FederatedDetectorSpec,
    run_federated_leakage_eval,
)
from openmed.eval.metrics import EvalSpan
from openmed.eval.release_gates import evaluate_federated_boundary_gate

SIGNING_KEY = "unit-federated-key"


def _fixture(fixture_id: str, surface: str) -> BenchmarkFixture:
    text = f"Patient {surface} arrived for follow up."
    start = text.index(surface)
    end = start + len(surface)
    return BenchmarkFixture(
        fixture_id=fixture_id,
        text=text,
        gold_spans=(EvalSpan(start=start, end=end, label="PERSON"),),
    )


def _write_detector(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "detector.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_federated_harness_catches_stdout_and_file_leakage(tmp_path: Path) -> None:
    fixture = _fixture("leaky", "Jane Roe")
    detector = _write_detector(
        tmp_path,
        """
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["OPENMED_DETECTOR_INPUT"]).read_text())
text = payload["text"]
start = text.index("Jane Roe")
end = start + len("Jane Roe")
print(text[start:end])
Path(os.environ["OPENMED_DETECTOR_OUTPUT_DIR"], "leak.txt").write_text(
    text[start:end]
)
Path(os.environ["OPENMED_DETECTOR_OUTPUT"]).write_text(
    json.dumps({"spans": [{"start": start, "end": end, "label": "PERSON"}]})
)
""".lstrip(),
    )

    report = run_federated_leakage_eval(
        [fixture],
        detector=detector,
        generated_at="2026-06-29T00:00:00+00:00",
        signing_key=SIGNING_KEY,
    )

    assert report.boundary_leakage.rate > 0.0
    assert report.boundary_leakage.leaked_bytes == len("Jane Roe")
    assert report.gate_passed is False
    assert report.verify(SIGNING_KEY)
    assert evaluate_federated_boundary_gate(report).passed is False
    rendered = report.to_json()
    assert "Jane Roe" not in rendered
    assert report.boundary_leakage.findings[0].text_hash in rendered


def test_federated_harness_passes_clean_offset_only_detector(tmp_path: Path) -> None:
    fixture = _fixture("clean", "Morgan Lee")
    detector = _write_detector(
        tmp_path,
        """
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["OPENMED_DETECTOR_INPUT"]).read_text())
text = payload["text"]
start = text.index("Morgan Lee")
end = start + len("Morgan Lee")
Path(os.environ["OPENMED_DETECTOR_OUTPUT"]).write_text(
    json.dumps({"spans": [{"start": start, "end": end, "label": "PERSON"}]})
)
""".lstrip(),
    )

    report = run_federated_leakage_eval(
        [fixture],
        detector=FederatedDetectorSpec(script_path=detector),
        generated_at="2026-06-29T00:00:00+00:00",
        signing_key=SIGNING_KEY,
    )

    assert report.boundary_leakage.rate == 0.0
    assert report.boundary_leakage.findings == ()
    assert report.sandbox_violations == ()
    assert report.gate_passed is True
    assert evaluate_federated_boundary_gate(report.to_benchmark_report()).passed is True


def test_federated_harness_reports_network_and_filesystem_violations(
    tmp_path: Path,
) -> None:
    fixture = _fixture("sandbox", "Riley Chen")
    detector = _write_detector(
        tmp_path,
        """
import json
import os
import socket
from pathlib import Path

payload = json.loads(Path(os.environ["OPENMED_DETECTOR_INPUT"]).read_text())
text = payload["text"]
start = text.index("Riley Chen")
end = start + len("Riley Chen")
try:
    socket.socket()
except Exception:
    pass
try:
    Path(os.environ["OPENMED_DETECTOR_OUTPUT_DIR"], "..", "escape.txt").write_text(
        "outside"
    )
except Exception:
    pass
Path(os.environ["OPENMED_DETECTOR_OUTPUT"]).write_text(
    json.dumps({"spans": [{"start": start, "end": end, "label": "PERSON"}]})
)
""".lstrip(),
    )

    report = run_federated_leakage_eval(
        [fixture],
        detector=detector,
        generated_at="2026-06-29T00:00:00+00:00",
        signing_key=SIGNING_KEY,
    )

    kinds = {violation.kind for violation in report.sandbox_violations}
    assert {"network", "filesystem"}.issubset(kinds)
    assert report.gate_passed is False
    assert evaluate_federated_boundary_gate(report).passed is False


def test_federated_harness_flags_phi_timing_side_channel(tmp_path: Path) -> None:
    names = _side_channel_names()
    fixtures = [_fixture(f"timing-{index}", name) for index, name in enumerate(names)]
    detector = _write_detector(
        tmp_path,
        """
import hashlib
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["OPENMED_DETECTOR_INPUT"]).read_text())
text = payload["text"]
prefix = "Patient "
suffix = " arrived"
start = text.index(prefix) + len(prefix)
end = text.index(suffix)
surface = text[start:end]
secret_bit = hashlib.sha256(surface.encode("utf-8")).digest()[0] & 1
duration_ms = 100.0 if secret_bit else 1.0
Path(os.environ["OPENMED_DETECTOR_OUTPUT"]).write_text(
    json.dumps(
        {
            "spans": [{"start": start, "end": end, "label": "PERSON"}],
            "timings": [
                {"start": start, "end": end, "duration_ms": duration_ms}
            ],
        }
    )
)
""".lstrip(),
    )

    report = run_federated_leakage_eval(
        fixtures,
        detector=detector,
        generated_at="2026-06-29T00:00:00+00:00",
        side_channel_threshold_bits=0.10,
        side_channel_min_samples=4,
        signing_key=SIGNING_KEY,
    )

    assert report.boundary_leakage.rate == 0.0
    assert report.side_channel.flagged is True
    assert report.side_channel.estimate_bits >= 0.10
    assert report.gate_passed is False
    rendered = report.to_json()
    for name in names:
        assert name not in rendered


def _side_channel_names() -> list[str]:
    candidates = [
        "Alice Nova",
        "Brian Stone",
        "Carla Vega",
        "Derek Hale",
        "Elena Ross",
        "Frank Vale",
        "Grace Park",
        "Hana Mills",
    ]
    selected: list[str] = []
    seen_bits: set[int] = set()
    for candidate in candidates:
        selected.append(candidate)
        seen_bits.add(hashlib.sha256(candidate.encode("utf-8")).digest()[0] & 1)
        if len(selected) >= 4 and seen_bits == {0, 1}:
            return selected
    raise AssertionError("side-channel fixture names must cover both secret buckets")
