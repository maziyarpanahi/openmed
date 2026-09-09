"""ASGI deployment entry point for the Health Universe PHI replacement agent."""

from __future__ import annotations

import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("DO_NOT_TRACK", "1")

import uvicorn  # noqa: E402
from health_universe_a2a import create_app  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

from main import agent  # noqa: E402

app = create_app(agent)


async def version_endpoint(_request):
    """Return non-sensitive runtime metadata."""

    return JSONResponse(
        {
            "agent": agent.get_agent_name(),
            "version": agent.get_agent_version(),
            "method": "replace",
            "platform_extraction": True,
            "offline_model_inference": True,
            "human_review_required": True,
        }
    )


app.add_route("/version", version_endpoint, methods=["GET"])


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
