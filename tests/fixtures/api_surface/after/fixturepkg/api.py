"""After-state public API fixture."""

from openmed.utils import deprecated

__all__ = [
    "LegacyClient",
    "added_later",
    "narrowed",
    "renamed_replacement",
    "retained",
]


class LegacyClient:
    """Fixture class with public attributes and methods."""

    stable_attribute = "stable"

    def request(self, value: str, optional: int) -> str:
        """Return a fixture response."""

        return f"{value}:{optional}"


def renamed_replacement(value: str) -> str:
    """Represent a renamed public function."""

    return value.casefold()


def narrowed(value: str, optional: int) -> str:
    """Represent a callable whose accepted calls later narrow."""

    return f"{value}:{optional}:0"


@deprecated(since="1.9", remove_in="2.0", replacement="fixturepkg.api.modern")
def retained(value: str) -> str:
    """Represent a callable that later becomes deprecated."""

    return value


def hidden_even_though_publicly_named() -> None:
    """Prove that a module ``__all__`` is authoritative."""


added_later = "now-public"
