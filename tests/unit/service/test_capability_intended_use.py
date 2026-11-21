"""Check REST projection against installed declarations at the service boundary."""

from fastapi.testclient import TestClient

from openmed.core.capabilities import validate_capability_descriptor
from openmed.core.intended_use import load_intended_use_registry
from openmed.service.app import create_app


def test_registered_routes_publish_only_bundled_intended_use(monkeypatch):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    app = create_app()
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get(
            "/v1/capabilities", params={"regulatory_tier": "private-sentinel"}
        )
        assert response.status_code == 200 and "private-sentinel" not in response.text
        descriptor = response.json()
        validate_capability_descriptor(descriptor)
        codes = [row["code"] for row in descriptor["capabilities"]]
        assert codes == [
            "analyze_text",
            "brief",
            "decisions",
            "deidentify",
            "extract_pii",
            "ground",
            "journey",
        ]
        for row in descriptor["capabilities"]:
            decl = load_intended_use_registry().resolve(row["code"])
            assert row["intended_use_digest"] == decl.digest()
            assert row["regulatory_tier"] == decl.regulatory_tier.value
            assert row["obligations"] == sorted(decl.regulatory_tier.obligations)
            assert row["excluded_uses"] == list(decl.excluded_uses)
        app.router.routes[:] = [
            r for r in app.routes if getattr(r, "path", None) != "/ground"
        ]
        assert "ground" not in [
            r["code"] for r in client.get("/v1/capabilities").json()["capabilities"]
        ]
