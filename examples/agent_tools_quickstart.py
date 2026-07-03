"""Render OpenMed registry tools for common agent framework adapters."""

from __future__ import annotations

from openmed.interop import to_function_tools, to_tool_use_tools
from openmed.interop.langchain import create_tool_definitions as langchain_definitions
from openmed.interop.llamaindex import (
    create_tool_definitions as llamaindex_definitions,
)

SYNTHETIC_TEXT = (
    "Patient Alex Morgan received 75 mg clopidogrel after NSTEMI follow-up."
)


def main() -> None:
    """Print adapter definitions generated from the registry."""

    adapters = {
        "function": to_function_tools(),
        "tool_use": to_tool_use_tools(),
        "langchain": langchain_definitions(),
        "llamaindex": llamaindex_definitions(),
    }

    for adapter, tools in adapters.items():
        first = tools[0]
        name = first["function"]["name"] if adapter == "function" else first["name"]
        print(f"{adapter}: {len(tools)} tools; first={name}")

    payload = {
        "text": SYNTHETIC_TEXT,
        "model_name": "disease_detection_superclinical",
        "confidence_threshold": 0.0,
    }
    print(f"synthetic payload keys: {', '.join(sorted(payload))}")


if __name__ == "__main__":
    main()
