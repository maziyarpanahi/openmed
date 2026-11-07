"""Offline round trip for native-digit and native-calendar locale PHI dates."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.core.labels import DATE, DATE_OF_BIRTH
from openmed.core.pii_i18n import get_patterns_for_language
from openmed.core.safety_sweep import safety_sweep
from openmed.training.synthetic.locale_phi import (
    SUPPORTED_LOCALE_PHI_LANGUAGES,
    generate_locale_phi_examples,
)
from openmed.training.synthetic.native_dates import (
    DEFAULT_NATIVE_DATE_PROFILES,
    NATIVE_DATE_PROFILES,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("native date generation must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_default_profiles_round_trip_through_jsonl(no_network, tmp_path):
    examples = generate_locale_phi_examples(
        seed=3886, date_profiles=DEFAULT_NATIVE_DATE_PROFILES
    )
    path = tmp_path / "locale_phi_native_dates.jsonl"
    path.write_text(
        "".join(
            json.dumps(example.to_training_item(), ensure_ascii=False) + "\n"
            for example in examples
        ),
        encoding="utf-8",
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [row["language"] for row in rows] == list(SUPPORTED_LOCALE_PHI_LANGUAGES)
    profiled = {
        language
        for language in DEFAULT_NATIVE_DATE_PROFILES
        if language in SUPPORTED_LOCALE_PHI_LANGUAGES
    }
    for row, example in zip(rows, examples):
        for label in row["labels"]:
            assert row["text"][label["start"] : label["end"]] == label["text"]
        date_labels = [
            label for label in row["labels"] if label["label"] in {DATE, DATE_OF_BIRTH}
        ]
        if row["language"] not in profiled:
            assert "date_profile" not in row["metadata"]
            continue

        profile = NATIVE_DATE_PROFILES[row["metadata"]["date_profile"]]
        date_patterns = [
            pattern
            for pattern in get_patterns_for_language(row["language"], row["locale"])
            if pattern.entity_type == "date"
        ]
        detected = {
            (item.start, item.end)
            for item in safety_sweep(
                row["text"],
                [],
                lang=row["language"],
                locale=row["locale"],
                patterns=date_patterns,
            )
        }
        for label in date_labels:
            assert label["metadata"]["calendar"] == profile.calendar
            assert label["metadata"]["numeral_system"] == profile.numeral_system
            assert label["metadata"]["form"] == profile.form
            assert label["text"] not in json.dumps(
                label["metadata"], ensure_ascii=False
            )
            gap = label["metadata"]["detection_gap"]
            if gap is None:
                assert (label["start"], label["end"]) in detected
            else:
                field = label["metadata"]["field"]
                assert row["metadata"]["date_detection_gaps"][field] == gap
        assert example.text == row["text"]
