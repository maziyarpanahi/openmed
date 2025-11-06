from __future__ import annotations

import pytest

import scripts.smoke_gliner as smoke


@pytest.mark.slow
def test_smoke_script_requires_gliner(monkeypatch):
    monkeypatch.setattr(smoke, "is_gliner_available", lambda: False)
    with pytest.raises(SystemExit):
        smoke.main([])
