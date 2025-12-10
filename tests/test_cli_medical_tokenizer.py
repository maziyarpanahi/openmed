import argparse

from openmed.cli.main import _load_and_apply_config
from openmed.core.config import get_config


def test_cli_override_medical_tokenizer():
    # start from default config (medical tokenizer enabled)
    base = get_config()
    assert base.use_medical_tokenizer is True

    args = argparse.Namespace(
        config_path=None,
        use_medical_tokenizer=False,
        medical_tokenizer_exceptions="COVID-19,MY-DRUG-123",
    )

    cfg = _load_and_apply_config(args)
    assert cfg.use_medical_tokenizer is False
    assert cfg.medical_tokenizer_exceptions == ["COVID-19", "MY-DRUG-123"]
