"""Root-level Health Universe PHI replacement agent entry point."""

from openmed.integrations.health_universe_agent import PhiReplacementAgent

agent = PhiReplacementAgent()

__all__ = ["PhiReplacementAgent", "agent"]
