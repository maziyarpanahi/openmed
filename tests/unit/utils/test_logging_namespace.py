"""Keep documented module names in the configured OpenMed logger hierarchy."""

import logging

import pytest

from openmed.utils.logging import OpenMedLogger, get_logger


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("worker", "openmed.worker"),
        ("core.models", "openmed.core.models"),
        ("openmed", "openmed"),
        ("openmed.core.models", "openmed.core.models"),
        ("openmed_tools", "openmed.openmed_tools"),
        ("openmedicine", "openmed.openmedicine"),
        ("", "openmed."),
    ],
)
def test_get_logger_preserves_qualified_names_and_prefixes_short_names(name, expected):
    assert get_logger(name) is logging.getLogger(expected)


def test_short_and_qualified_names_resolve_to_the_same_logger():
    assert get_logger("core.synthetic") is get_logger("openmed.core.synthetic")


def test_module_specific_log_level_is_respected(caplog):
    parent_name = "openmed.namespace_regression"
    child = get_logger(parent_name + ".worker")
    previous_level = child.level
    child.setLevel(logging.NOTSET)
    try:
        with caplog.at_level(logging.DEBUG, logger="openmed"):
            with caplog.at_level(logging.WARNING, logger=parent_name):
                child.info("synthetic information")
                child.warning("synthetic warning")
        assert [record.message for record in caplog.records] == ["synthetic warning"]
    finally:
        child.setLevel(previous_level)


def test_context_logger_uses_the_configured_module_name(caplog):
    name = "openmed.namespace_regression.context"
    logger = OpenMedLogger(name)
    with caplog.at_level(logging.INFO, logger="openmed"):
        logger.log_predictions(2, "synthetic-model")
    assert logger.logger is logging.getLogger(name)
    assert [(record.name, record.message) for record in caplog.records] == [
        (name, "Model synthetic-model predicted 2 entities")
    ]
